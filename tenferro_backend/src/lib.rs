//! `artensor_tenferro`: runs artensor's contraction schemes on tenferro, on CPU or CUDA.
//!
//! artensor compiles a scheme (dense or sparse-state) into a short program of binary einsums,
//! row selections, reshapes and concatenations over numbered slots (`artensor/tenferro_backend.py`),
//! and [`Backend::run`] executes it, summing over the sliced bonds on the device. Arrays are
//! NumPy arrays in C order; see `exec.rs` for how they map onto tenferro's column-major tensors.

mod exec;

use num_complex::{Complex32, Complex64};
use numpy::{
    Element, PyArray1, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use tenferro_tensor::{Tensor, TensorScalar, TensorValue};

use exec::{Device, Instr, Program, SlicedBond};

pub(crate) type Error = Box<dyn std::error::Error + 'static>;
pub(crate) type Result<T> = std::result::Result<T, Error>;

pub(crate) fn err(message: impl Into<String>) -> Error {
    message.into().into()
}

fn runtime_error(e: Error) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// One tenferro engine: `Backend("cpu", threads)`, `Backend("cuda")` or `Backend("cuda:N")`.
///
/// Keep one backend per device and reuse it: the engine owns tenferro's plan cache and buffer
/// pool.
#[pyclass(unsendable, module = "artensor_tenferro")]
struct Backend {
    device: Device,
    spec: String,
    threads: usize,
}

#[pymethods]
impl Backend {
    #[new]
    #[pyo3(signature = (device = "cpu", threads = 1))]
    fn new(device: &str, threads: usize) -> PyResult<Self> {
        let engine = Device::open(device, threads).map_err(runtime_error)?;
        Ok(Self {
            device: engine,
            spec: device.to_owned(),
            threads,
        })
    }

    /// The device this backend runs on.
    #[getter]
    fn device(&self) -> &str {
        &self.spec
    }

    /// CPU worker threads (1 runs on the calling thread).
    #[getter]
    fn threads(&self) -> usize {
        self.threads
    }

    fn __repr__(&self) -> String {
        format!("Backend({:?}, threads={})", self.spec, self.threads)
    }

    /// Run `program` on `inputs` and return `(log10_factor, result)`.
    ///
    /// `inputs` are C-contiguous NumPy arrays of one dtype (complex64, complex128, float32 or
    /// float64), in slot order. `program` is a list of tuples:
    ///
    /// - `("einsum", out, lhs, rhs, lhs_labels, rhs_labels, out_labels)`, labels as small ints;
    /// - `("index_select", out, src, axis, positions)`;
    /// - `("reshape", out, src, shape)`, with at most one `-1`;
    /// - `("concat", out, srcs, axis)`;
    /// - `("free", slot)`;
    /// - `("normalize", slot)`: divide by the largest magnitude, add its log10 to the factor.
    ///
    /// `slicing` lists the sliced bonds as `(extent, [(input slot, axis), ...])`, each axis
    /// counted in the unsliced input; the result is the sum over all their values. The factor
    /// is 0 unless the program normalizes, which slicing does not allow.
    #[pyo3(signature = (inputs, program, result, n_slots, slicing = None))]
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        inputs: Vec<Bound<'py, PyAny>>,
        program: Vec<Bound<'py, PyAny>>,
        result: usize,
        n_slots: usize,
        slicing: Option<Vec<(usize, Vec<(usize, usize)>)>>,
    ) -> PyResult<(f64, Py<PyAny>)> {
        let program = parse_program(&program, inputs.len(), n_slots, result)?;
        let scientific = program
            .instrs
            .iter()
            .any(|i| matches!(i, Instr::Normalize { .. }));
        let slicing: Vec<SlicedBond> = slicing
            .unwrap_or_default()
            .into_iter()
            .map(|(dim, sites)| SlicedBond { dim, sites })
            .collect();

        macro_rules! run_as {
            ($ty:ty) => {
                if let Some(arrays) = extract_all::<$ty>(&inputs) {
                    return self.run_typed::<$ty>(py, &arrays, &program, &slicing, scientific);
                }
            };
        }
        run_as!(Complex64);
        run_as!(Complex32);
        run_as!(f64);
        run_as!(f32);
        Err(PyTypeError::new_err(
            "inputs must be NumPy arrays of one dtype: complex64, complex128, float32 or float64",
        ))
    }
}

impl Backend {
    fn run_typed<'py, T: Element + TensorScalar + Copy>(
        &mut self,
        py: Python<'py>,
        arrays: &[PyReadonlyArrayDyn<'py, T>],
        program: &Program,
        slicing: &[SlicedBond],
        scientific: bool,
    ) -> PyResult<(f64, Py<PyAny>)> {
        let mut inputs = Vec::with_capacity(arrays.len());
        for (k, array) in arrays.iter().enumerate() {
            if !array.is_c_contiguous() {
                return Err(PyValueError::new_err(format!(
                    "input {k} is not C-contiguous"
                )));
            }
            let data = array
                .as_slice()
                .map_err(|e| PyValueError::new_err(format!("input {k}: {e}")))?;
            // Row-major [d0, ..., dk] is column-major [dk, ..., d0] over the same memory.
            let shape: Vec<usize> = array.shape().iter().rev().copied().collect();
            let host = Tensor::from_vec_col_major(shape, data.to_vec())
                .map_err(|e| runtime_error(e.into()))?;
            let tensor = self.device.upload(host).map_err(runtime_error)?;
            inputs.push(TensorValue::from_tensor(tensor));
        }
        let (tensor, log10_factor) = self
            .device
            .contract(program, &inputs, slicing, scientific)
            .map_err(runtime_error)?;
        drop(inputs);
        let host = self.device.download(tensor).map_err(runtime_error)?;
        let shape: Vec<usize> = host.shape().iter().rev().copied().collect();
        let data = host
            .as_slice::<T>()
            .map_err(|e| runtime_error(e.into()))?
            .to_vec();
        let array = PyArray1::from_vec(py, data).reshape(shape)?;
        Ok((log10_factor, array.into_any().unbind()))
    }
}

/// Every input as a read-only array of `T`, or `None` if any is not one.
fn extract_all<'py, T: Element>(
    inputs: &[Bound<'py, PyAny>],
) -> Option<Vec<PyReadonlyArrayDyn<'py, T>>> {
    inputs
        .iter()
        .map(|x| x.extract::<PyReadonlyArrayDyn<'py, T>>().ok())
        .collect()
}

fn parse_program(
    items: &[Bound<'_, PyAny>],
    n_inputs: usize,
    n_slots: usize,
    result: usize,
) -> PyResult<Program> {
    let mut instrs = Vec::with_capacity(items.len());
    for (k, item) in items.iter().enumerate() {
        let bad = |e: PyErr| PyValueError::new_err(format!("instruction {k}: {e}"));
        let name: String = item.get_item(0).and_then(|x| x.extract()).map_err(bad)?;
        let instr = match name.as_str() {
            "einsum" => {
                let (_, out, lhs, rhs, lhs_labels, rhs_labels, out_labels): (
                    String,
                    usize,
                    usize,
                    usize,
                    Vec<u32>,
                    Vec<u32>,
                    Vec<u32>,
                ) = item.extract().map_err(bad)?;
                Instr::Einsum {
                    out,
                    lhs,
                    rhs,
                    lhs_labels,
                    rhs_labels,
                    out_labels,
                }
            }
            "index_select" => {
                let (_, out, src, axis, positions): (String, usize, usize, usize, Vec<usize>) =
                    item.extract().map_err(bad)?;
                Instr::IndexSelect {
                    out,
                    src,
                    axis,
                    positions,
                }
            }
            "reshape" => {
                let (_, out, src, shape): (String, usize, usize, Vec<i64>) =
                    item.extract().map_err(bad)?;
                Instr::Reshape { out, src, shape }
            }
            "concat" => {
                let (_, out, srcs, axis): (String, usize, Vec<usize>, usize) =
                    item.extract().map_err(bad)?;
                Instr::Concat { out, srcs, axis }
            }
            "free" => {
                let (_, slot): (String, usize) = item.extract().map_err(bad)?;
                Instr::Free { slot }
            }
            "normalize" => {
                let (_, slot): (String, usize) = item.extract().map_err(bad)?;
                Instr::Normalize { slot }
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "instruction {k}: unknown operation {other:?}"
                )))
            }
        };
        instrs.push(instr);
    }
    let program = Program {
        n_inputs,
        n_slots,
        instrs,
        result,
    };
    program
        .validate()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(program)
}

/// Whether this build can run on CUDA and a usable CUDA device is present.
#[pyfunction]
fn cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    return tenferro_gpu::cuda::gpu_available();
    #[cfg(not(feature = "cuda"))]
    return false;
}

#[pymodule]
fn artensor_tenferro(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Backend>()?;
    m.add_function(wrap_pyfunction!(cuda_available, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("CUDA_BUILD", cfg!(feature = "cuda"))?;
    Ok(())
}
