//! Running a contraction program on one tenferro engine (CPU or CUDA).
//!
//! artensor's tensors are row-major (C order) and tenferro's are column-major. A row-major
//! array of shape `[d0, ..., dk]` is the same memory as a column-major tensor of shape
//! `[dk, ..., d0]`, so arrays cross the boundary without reordering: the executor keeps every
//! tensor with its axes reversed and maps each axis, label list and shape of a program onto
//! that order. Programs themselves are written in artensor's row-major terms.

use tenferro_einsum::{EinsumOps, EinsumSubscripts};
use tenferro_runtime::CoreOps;
use tenferro_tensor::{
    Engine, GatherConfig, MemoryKind, Tensor, TensorRead, TensorValue, TensorView,
};

use crate::{err, Result};

/// One instruction of a contraction program. Slots hold tensors; axes, label lists and shapes
/// are in artensor's row-major convention.
#[derive(Clone, Debug)]
pub enum Instr {
    /// Binary einsum: one label per axis of each operand, and the result's labels in order.
    Einsum {
        out: usize,
        lhs: usize,
        rhs: usize,
        lhs_labels: Vec<u32>,
        rhs_labels: Vec<u32>,
        out_labels: Vec<u32>,
    },
    /// Rows `positions` (which may repeat) along `axis`, like `tensor[positions]` on axis 0.
    IndexSelect {
        out: usize,
        src: usize,
        axis: usize,
        positions: Vec<usize>,
    },
    /// Row-major reshape; one extent may be `-1`.
    Reshape {
        out: usize,
        src: usize,
        shape: Vec<i64>,
    },
    /// Concatenation along `axis`.
    Concat {
        out: usize,
        srcs: Vec<usize>,
        axis: usize,
    },
    /// Release a slot.
    Free { slot: usize },
    /// Divide a slot by its largest magnitude and add that magnitude's log10 to the program's
    /// factor (artensor's `scientific_notation`).
    Normalize { slot: usize },
}

/// A program over `n_slots` slots, the first `n_inputs` of which hold the inputs.
#[derive(Clone, Debug)]
pub struct Program {
    pub n_inputs: usize,
    pub n_slots: usize,
    pub instrs: Vec<Instr>,
    pub result: usize,
}

impl Program {
    /// Every slot an instruction names exists.
    pub fn validate(&self) -> Result<()> {
        if self.n_inputs > self.n_slots {
            return Err(err(format!(
                "{} inputs do not fit in {} slots",
                self.n_inputs, self.n_slots
            )));
        }
        let check = |s: usize, k: usize| -> Result<()> {
            if s < self.n_slots {
                Ok(())
            } else {
                Err(err(format!(
                    "instruction {k} names slot {s}, but the program has {} slots",
                    self.n_slots
                )))
            }
        };
        for (k, instr) in self.instrs.iter().enumerate() {
            match instr {
                Instr::Einsum { out, lhs, rhs, .. } => {
                    check(*out, k)?;
                    check(*lhs, k)?;
                    check(*rhs, k)?;
                }
                Instr::IndexSelect { out, src, .. } | Instr::Reshape { out, src, .. } => {
                    check(*out, k)?;
                    check(*src, k)?;
                }
                Instr::Concat { out, srcs, .. } => {
                    check(*out, k)?;
                    for s in srcs {
                        check(*s, k)?;
                    }
                }
                Instr::Free { slot } | Instr::Normalize { slot } => check(*slot, k)?,
            }
        }
        check(self.result, self.instrs.len())
    }
}

/// A sliced bond: its extent, and every input carrying it as `(input slot, axis)`, with the
/// axis counted in that input's own (unsliced) row-major axes.
#[derive(Clone, Debug)]
pub struct SlicedBond {
    pub dim: usize,
    pub sites: Vec<(usize, usize)>,
}

/// The engine a backend runs on.
pub enum Device {
    Cpu(tenferro_cpu::CpuEngine),
    #[cfg(feature = "cuda")]
    Cuda(tenferro_gpu::cuda::CudaEngine),
}

/// A slot's tensor: an input shared by every slice, or a value the program owns.
enum Slot<'a> {
    Shared(&'a TensorValue),
    Owned(TensorValue),
}

impl Slot<'_> {
    fn value(&self) -> &TensorValue {
        match self {
            Slot::Shared(v) => v,
            Slot::Owned(v) => v,
        }
    }
}

fn get<'s>(slots: &'s [Option<Slot<'_>>], s: usize) -> Result<&'s TensorValue> {
    slots
        .get(s)
        .and_then(Option::as_ref)
        .map(Slot::value)
        .ok_or_else(|| err(format!("slot {s} is read before it is written")))
}

/// The column-major axis of row-major `axis` in a tensor of rank `rank`.
fn col(rank: usize, axis: usize) -> Result<usize> {
    if axis < rank {
        Ok(rank - 1 - axis)
    } else {
        Err(err(format!("axis {axis} is out of range for rank {rank}")))
    }
}

fn reversed<T: Copy>(v: &[T]) -> Vec<T> {
    v.iter().rev().copied().collect()
}

/// `shape` with its `-1` extent (if any) resolved for `numel` elements.
fn resolve_shape(shape: &[i64], numel: usize) -> Result<Vec<usize>> {
    let mut known = 1usize;
    let mut free = None;
    for (k, &d) in shape.iter().enumerate() {
        match d {
            -1 if free.is_none() => free = Some(k),
            d if d >= 0 => known *= d as usize,
            _ => return Err(err(format!("invalid reshape target {shape:?}"))),
        }
    }
    let mut out: Vec<usize> = shape.iter().map(|&d| d.max(0) as usize).collect();
    if let Some(k) = free {
        if known == 0 || numel % known != 0 {
            return Err(err(format!("cannot reshape {numel} elements to {shape:?}")));
        }
        out[k] = numel / known;
    } else if known != numel {
        return Err(err(format!("cannot reshape {numel} elements to {shape:?}")));
    }
    Ok(out)
}

/// A read of `value`.
fn read(value: &TensorValue) -> Result<TensorRead<'_>> {
    value
        .tensor_read()
        .ok_or_else(|| err("an external payload has no tensor read"))
}

/// `shape` with the axes before and after `axis` merged into one each.
fn around(shape: &[usize], axis: usize) -> [usize; 3] {
    [
        shape[..axis].iter().product(),
        shape[axis],
        shape[axis + 1..].iter().product(),
    ]
}

/// `value` read as `[prefix, extent, suffix]` around `axis`, which a compact column-major
/// layout takes as a relabeling, not a copy.
///
/// tenferro's CUDA gather and concatenation have fast kernels up to rank 8 only; past that a
/// general kernel decodes every axis of every element (tensor4all/tenferro-rs#2708), so every
/// gather and concatenation here runs on three merged axes.
fn merged_read(value: &TensorValue, axis: usize) -> Result<TensorRead<'_>> {
    let shape = around(value.shape(), axis);
    let view = value
        .tensor_view()
        .ok_or_else(|| err("an external payload has no tensor view"))?;
    // The dtype-erased view has no reshape of its own; each typed arm does.
    let view = match view {
        TensorView::F32(v) => TensorView::F32(v.reshape_view(&shape)?),
        TensorView::F64(v) => TensorView::F64(v.reshape_view(&shape)?),
        TensorView::I32(v) => TensorView::I32(v.reshape_view(&shape)?),
        TensorView::I64(v) => TensorView::I64(v.reshape_view(&shape)?),
        TensorView::Bool(v) => TensorView::Bool(v.reshape_view(&shape)?),
        TensorView::C32(v) => TensorView::C32(v.reshape_view(&shape)?),
        TensorView::C64(v) => TensorView::C64(v.reshape_view(&shape)?),
    };
    Ok(TensorRead::from_view(view))
}

/// Relabel a compact `[prefix, n, suffix]` result as `shape` with `n` on `axis`.
fn unmerge(t: Tensor, mut shape: Vec<usize>, axis: usize) -> Result<TensorValue> {
    shape[axis] = t.shape()[1];
    Ok(TensorValue::from_tensor(t).reshape_view(shape)?)
}

/// The first element of a host scalar, as f64.
fn scalar_f64(t: &Tensor) -> Result<f64> {
    if let Ok(s) = t.as_slice::<f64>() {
        return s.first().copied().ok_or_else(|| err("empty reduction"));
    }
    let s = t.as_slice::<f32>()?;
    s.first()
        .map(|&x| f64::from(x))
        .ok_or_else(|| err("empty reduction"))
}

impl Device {
    /// `"cpu"` (with `threads` worker threads), `"cuda"` or `"cuda:N"`.
    pub fn open(spec: &str, threads: usize) -> Result<Self> {
        if spec == "cpu" {
            return Self::cpu(threads);
        }
        if let Some(rest) = spec.strip_prefix("cuda") {
            let ordinal = if rest.is_empty() {
                Some(0)
            } else {
                rest.strip_prefix(':').and_then(|n| n.parse().ok())
            };
            if let Some(ordinal) = ordinal {
                return Self::cuda(ordinal);
            }
        }
        Err(err(format!(
            "unknown device {spec:?}: use \"cpu\", \"cuda\" or \"cuda:N\""
        )))
    }

    fn cpu(threads: usize) -> Result<Self> {
        use tenferro_cpu::{CpuEngine, Pool};
        let engine = if threads > 1 {
            CpuEngine::new(Pool::builder().threads(threads).build()?)
        } else {
            CpuEngine::serial()
        };
        Ok(Self::Cpu(engine))
    }

    #[cfg(feature = "cuda")]
    fn cuda(ordinal: usize) -> Result<Self> {
        use tenferro_gpu::cuda::{cuda_devices, gpu_available, CudaEngine};
        if !gpu_available() {
            return Err(err("no usable CUDA driver or device"));
        }
        let devices = cuda_devices()?;
        let visible = devices.len();
        let device = devices.get(ordinal).ok_or_else(|| {
            err(format!(
                "CUDA device {ordinal} is not visible ({visible} visible)"
            ))
        })?;
        Ok(Self::Cuda(CudaEngine::open(device.id())?))
    }

    #[cfg(not(feature = "cuda"))]
    fn cuda(_ordinal: usize) -> Result<Self> {
        Err(err(
            "this build of artensor_tenferro has no CUDA support; rebuild it with the `cuda` feature",
        ))
    }

    fn engine(&mut self) -> &mut dyn Engine {
        match self {
            Self::Cpu(engine) => engine,
            #[cfg(feature = "cuda")]
            Self::Cuda(engine) => engine,
        }
    }

    /// A host tensor on this device.
    pub fn upload(&mut self, host: Tensor) -> Result<Tensor> {
        match self {
            Self::Cpu(_) => Ok(host),
            #[cfg(feature = "cuda")]
            Self::Cuda(engine) => Ok(tenferro_gpu::cuda::upload_tensor(engine.runtime(), &host)?),
        }
    }

    /// A tensor of this device on the host. On CUDA this waits for all queued work.
    pub fn download(&mut self, tensor: Tensor) -> Result<Tensor> {
        match self {
            Self::Cpu(_) => Ok(tensor),
            #[cfg(feature = "cuda")]
            Self::Cuda(engine) => Ok(tenferro_gpu::cuda::download_tensor(
                engine.runtime(),
                &tensor,
            )?),
        }
    }

    /// A compact tensor holding `value`: its own buffer when the layout is the buffer's.
    fn compact(&mut self, value: TensorValue) -> Result<Tensor> {
        if value.as_tensor().is_some() {
            return Ok(value.into_tensor()?);
        }
        let shape = value.shape().to_vec();
        Ok(self.engine().reshape_read(read(&value)?, &shape)?)
    }

    fn einsum(
        &mut self,
        inputs: [&TensorValue; 2],
        labels: [&[u32]; 2],
        output: &[u32],
    ) -> Result<TensorValue> {
        let reads = inputs.iter().map(|v| read(v)).collect::<Result<Vec<_>>>()?;
        let subscripts = EinsumSubscripts::new(&labels, output);
        Ok(TensorValue::from_tensor(
            reads.einsum(&subscripts, self.engine())?,
        ))
    }

    /// `positions` along column-major `axis`.
    fn index_select(
        &mut self,
        t: &TensorValue,
        axis: usize,
        positions: &[usize],
    ) -> Result<TensorValue> {
        let [prefix, extent, suffix] = around(t.shape(), axis);
        if let Some(&p) = positions.iter().find(|&&p| p >= extent) {
            return Err(err(format!(
                "position {p} is out of range for an axis of extent {extent}"
            )));
        }
        let config = GatherConfig {
            offset_dims: vec![0, 2],
            collapsed_slice_dims: vec![1],
            start_index_map: vec![1],
            index_vector_dim: 1,
            slice_sizes: vec![prefix, 1, suffix],
        };
        let index: Vec<i64> = positions
            .iter()
            .map(|&p| i64::try_from(p).map_err(|_| err(format!("position {p} exceeds i64"))))
            .collect::<Result<_>>()?;
        let host = Tensor::from_vec_col_major(vec![positions.len(), 1], index)?;
        let operand = merged_read(t, axis)?;
        // A device engine refuses host indices rather than copy them implicitly.
        let uploaded = match operand.placement().memory_kind {
            MemoryKind::Device => Some(
                self.engine()
                    .upload_host_tensor(TensorRead::from_tensor(&host))?,
            ),
            _ => None,
        };
        let indices = TensorRead::from_tensor(uploaded.as_ref().unwrap_or(&host));
        let out = self.engine().gather_read(operand, indices, &config)?;
        unmerge(out, t.shape().to_vec(), axis)
    }

    /// Index `index` of column-major `axis`, with that axis dropped.
    fn select(&mut self, t: &TensorValue, axis: usize, index: usize) -> Result<TensorValue> {
        let picked = self.index_select(t, axis, &[index])?;
        let mut shape = picked.shape().to_vec();
        shape.remove(axis);
        Ok(picked.reshape_view(shape)?)
    }

    /// Column-major reshape.
    fn reshape(&mut self, t: &TensorValue, shape: &[usize]) -> Result<TensorValue> {
        let out = match t.as_tensor() {
            Some(tensor) => tensor.reshape(shape, self.engine())?,
            None => self.engine().reshape_read(read(t)?, shape)?,
        };
        Ok(TensorValue::from_tensor(out))
    }

    /// Concatenation along column-major `axis`.
    fn concatenate(&mut self, inputs: &[&TensorValue], axis: usize) -> Result<TensorValue> {
        let first = inputs
            .first()
            .ok_or_else(|| err("concatenate needs at least one input"))?;
        let reads = inputs
            .iter()
            .map(|v| merged_read(v, axis))
            .collect::<Result<Vec<_>>>()?;
        let out = self.engine().concatenate_read(&reads, 1)?;
        unmerge(out, first.shape().to_vec(), axis)
    }

    fn add(&mut self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        Ok(TensorRead::from_tensor(a).add(&TensorRead::from_tensor(b), self.engine())?)
    }

    /// `t` divided by its largest magnitude, and that magnitude's log10.
    fn normalize(&mut self, t: &TensorValue) -> Result<(TensorValue, f64)> {
        let magnitude = read(t)?.abs(self.engine())?;
        let peak = TensorRead::from_tensor(&magnitude).reduce_max(None, self.engine())?;
        let peak = scalar_f64(&self.download(peak)?)?;
        let scaled = read(t)?.scale_real(1.0 / peak, self.engine())?;
        Ok((TensorValue::from_tensor(scaled), peak.log10()))
    }

    /// Run `program` once on `operands`; returns the result and the summed log10 factors of its
    /// `Normalize` instructions.
    fn execute(&mut self, program: &Program, operands: Vec<Slot<'_>>) -> Result<(TensorValue, f64)> {
        let mut slots: Vec<Option<Slot<'_>>> = operands.into_iter().map(Some).collect();
        slots.resize_with(program.n_slots, || None);
        let mut log10_factor = 0.0;
        for (k, instr) in program.instrs.iter().enumerate() {
            let at = |e: crate::Error| err(format!("instruction {k} ({instr:?}): {e}"));
            match instr {
                Instr::Einsum {
                    out,
                    lhs,
                    rhs,
                    lhs_labels,
                    rhs_labels,
                    out_labels,
                } => {
                    let a = get(&slots, *lhs)?;
                    let b = get(&slots, *rhs)?;
                    if a.shape().len() != lhs_labels.len() || b.shape().len() != rhs_labels.len() {
                        return Err(at(err(format!(
                            "operand ranks {} and {} do not match {} and {} labels",
                            a.shape().len(),
                            b.shape().len(),
                            lhs_labels.len(),
                            rhs_labels.len()
                        ))));
                    }
                    let (la, lb, lo) = (reversed(lhs_labels), reversed(rhs_labels), reversed(out_labels));
                    let value = self.einsum([a, b], [&la, &lb], &lo).map_err(at)?;
                    slots[*out] = Some(Slot::Owned(value));
                }
                Instr::IndexSelect {
                    out,
                    src,
                    axis,
                    positions,
                } => {
                    let v = get(&slots, *src)?;
                    let c = col(v.shape().len(), *axis).map_err(at)?;
                    let value = self.index_select(v, c, positions).map_err(at)?;
                    slots[*out] = Some(Slot::Owned(value));
                }
                Instr::Reshape { out, src, shape } => {
                    let v = get(&slots, *src)?;
                    let numel = v.shape().iter().product();
                    let target = resolve_shape(shape, numel).map_err(at)?;
                    let value = self.reshape(v, &reversed(&target)).map_err(at)?;
                    slots[*out] = Some(Slot::Owned(value));
                }
                Instr::Concat { out, srcs, axis } => {
                    let vs = srcs
                        .iter()
                        .map(|&s| get(&slots, s))
                        .collect::<Result<Vec<_>>>()?;
                    let rank = vs.first().map_or(0, |v| v.shape().len());
                    let c = col(rank, *axis).map_err(at)?;
                    let value = self.concatenate(&vs, c).map_err(at)?;
                    slots[*out] = Some(Slot::Owned(value));
                }
                Instr::Free { slot } => slots[*slot] = None,
                Instr::Normalize { slot } => {
                    let v = get(&slots, *slot)?;
                    let (scaled, factor) = self.normalize(v).map_err(at)?;
                    log10_factor += factor;
                    slots[*slot] = Some(Slot::Owned(scaled));
                }
            }
        }
        let result = slots
            .get_mut(program.result)
            .and_then(Option::take)
            .ok_or_else(|| err(format!("the program left slot {} empty", program.result)))?;
        let value = match result {
            Slot::Owned(v) => v,
            Slot::Shared(v) => {
                let shape = v.shape().to_vec();
                TensorValue::from_tensor(self.engine().reshape_read(read(v)?, &shape)?)
            }
        };
        Ok((value, log10_factor))
    }

    /// Sum `program` over every value of the sliced bonds, slicing the inputs on this device.
    ///
    /// Slices run in artensor's order: bond 0 is the most significant digit. On an input
    /// carrying several sliced bonds the selects run from its highest axis down, so each axis
    /// (counted in the unsliced input) still names the same axis when it is applied.
    pub fn contract(
        &mut self,
        program: &Program,
        inputs: &[TensorValue],
        slicing: &[SlicedBond],
        scientific: bool,
    ) -> Result<(Tensor, f64)> {
        if inputs.len() != program.n_inputs {
            return Err(err(format!(
                "the program takes {} inputs, got {}",
                program.n_inputs,
                inputs.len()
            )));
        }
        let mut n_slices = 1usize;
        let mut sites: Vec<Vec<(usize, usize)>> = vec![Vec::new(); inputs.len()];
        for (k, bond) in slicing.iter().enumerate() {
            n_slices = n_slices
                .checked_mul(bond.dim)
                .ok_or_else(|| err("the number of slices overflows"))?;
            for &(slot, axis) in &bond.sites {
                let input = inputs
                    .get(slot)
                    .ok_or_else(|| err(format!("sliced bond {k} names input {slot}, which does not exist")))?;
                if axis >= input.shape().len() || input.shape()[input.shape().len() - 1 - axis] != bond.dim {
                    return Err(err(format!(
                        "sliced bond {k} (extent {}) does not match axis {axis} of input {slot} (row-major shape {:?})",
                        bond.dim,
                        reversed(input.shape())
                    )));
                }
                sites[slot].push((axis, k));
            }
        }
        for s in &mut sites {
            s.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            if s.windows(2).any(|w| w[0].0 == w[1].0) {
                return Err(err("two sliced bonds name the same axis of one input"));
            }
        }
        if scientific && n_slices > 1 {
            return Err(err("scientific notation is not supported together with slicing"));
        }

        let mut config = vec![0usize; slicing.len()];
        let mut acc: Option<Tensor> = None;
        let mut log10_factor = 0.0;
        for s in 0..n_slices {
            let mut rest = s;
            for k in (0..slicing.len()).rev() {
                config[k] = rest % slicing[k].dim;
                rest /= slicing[k].dim;
            }
            let mut operands = Vec::with_capacity(inputs.len());
            for (slot, input) in inputs.iter().enumerate() {
                if sites[slot].is_empty() {
                    operands.push(Slot::Shared(input));
                    continue;
                }
                let mut value: Option<TensorValue> = None;
                for &(axis, k) in &sites[slot] {
                    let current = value.as_ref().unwrap_or(input);
                    let c = col(current.shape().len(), axis)?;
                    value = Some(self.select(current, c, config[k])?);
                }
                operands.push(Slot::Owned(value.expect("a sliced input has sites")));
            }
            let (value, factor) = self.execute(program, operands)?;
            log10_factor = factor;
            let tensor = self.compact(value)?;
            acc = Some(match acc {
                None => tensor,
                Some(prev) => self.add(&prev, &tensor)?,
            });
        }
        let result = acc.ok_or_else(|| err("no slice was contracted"))?;
        Ok((result, log10_factor))
    }
}
