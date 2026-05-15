# QEC Handoff: `d5_r05` Work Through 2026-05-15

## Scope

This note captures the state of the `artensor` QEC optimization work at the point where the current branch was cut for migration to another server.

The work in this branch focuses on:

- faster and better QEC benchmarking helpers
- a stronger `paper_skewed` greedy path for open-network QEC cases
- post-target slice replacement / restoration logic
- seed-search and multitrial tooling for `d5_r05`
- experimental beam-search restore variants

The benchmark input used throughout this work is now included in the repo:

- `examples/qec_inputs/d5_r05.txt`
- `examples/qec_inputs/d5_r05seed_01.json`

## Repository Layout

The actual git repo is this directory:

- `artensor/`

The surrounding parent directory on the original machine was not a git repo. Earlier logs and scratch outputs were produced in the parent `TensorContraction/` directory, but the important benchmark input files have been copied into this branch under `examples/qec_inputs/`.

## Current Best Known `artensor` Result

Best result found for `d5_r05` with the current new-design pipeline:

- seed: `308`
- greedy strategy: `paper_skewed`
- greedy alpha: `0.1`
- final slices: `7`
- `tc = 12.620759833191528`
- `sc = 33.0`
- `mc = 11.230418306321711`

This result was reproduced again in the targeted seed-cluster rerun, so it is stable.

## Reference Order

The provided reference order in `examples/qec_inputs/d5_r05seed_01.json` is still stronger than the best result found by `artensor`.

Important point:

- `artensor` can evaluate the provided order correctly
- the gap is in search, not in order evaluation

This means the remaining problem is still optimizer quality for open QEC networks, not correctness of tree scoring on a supplied good order.

## Main Findings So Far

### 1. `paper_skewed` was the major initializer breakthrough

Earlier greedy variants were much worse on `d5_r05`.

What worked:

- `paper_skewed`
- `greedy_alpha = 0.1` or `0.05`

What did not help:

- `min_dim`
- `paper_hybrid`
- `paper_tc` on this benchmark

### 2. Broad mixed-family multitrial search was not efficient on this machine

The multitrial runner can prescreen many candidates, but broad-family runs were poor use of time for `d5_r05`.

Reason:

- `paper_tc`, `paper_hybrid`, and `min_dim` prescreened into much worse unsliced trees
- full optimization on those candidates was expensive and did not beat the skewed family

### 3. The strong seed region is real

The best seeds found historically were concentrated in the old `v2` block:

- `298`
- `308`
- `322`
- nearby seeds in the same `201..400` region

Targeted reruns on that neighborhood confirmed:

- `308` stays best at `7` slices
- `298` stays at `9`
- `322` stays at `10`

### 4. More post-target depth was not the answer

Tested and rejected:

- deeper post-target rounds
- larger replacement candidate limits
- adding `iters=1`
- heavier annealing after hitting target

These typically made slice count worse, even when `tc` improved.

### 5. Stronger restore search is still unresolved

Two beam-search experiments were tried:

- `examples/benchmark_qec_beam_refine.py`
  - first prototype
  - not comparable to baseline because it changed the whole sliced trajectory
  - do not treat its result as meaningful against the baseline

- `examples/benchmark_qec_beam_refine_v2.py`
  - corrected wrapper
  - changes only the final restore stage
  - still did not beat baseline
  - result on seed `308`: `8` slices

So the current baseline restore logic still beats the beam prototype on the main objective.

## Scripts Added / Modified

### `examples/benchmark_qec_instance.py`

This is the main new-design QEC runner.

Important features:

- deterministic QEC parsing
- seed-scanned greedy candidate selection
- peak-subtree rebuild phase
- slicing stage with per-step logging
- post-target refine / replace logic
- single-bond restore refinement stage

This is still the main script to use for serious QEC experiments.

### `examples/benchmark_qec_multitrial.py`

Purpose:

- prescreen many `(seed, strategy, greedy_alpha)` candidates
- run full optimization on the selected pool

Status:

- useful for broad exploration
- not ideal for this machine when the selected pool is too large
- the script logs `trial_started`, `trial_done`, and `trial_failed`
- it is still an experimental helper rather than the most reliable path for `d5_r05`

Recommendation:

- use only skewed-heavy candidate pools on `d5_r05`
- keep worker count low on smaller machines

### `examples/benchmark_qec_find_order.py`

Purpose:

- explicit, logged legacy-style `find_order` path
- useful for comparing the older slicing loop against the new design

Status:

- kept mainly for comparison and diagnosis
- not the best route for the current QEC work

### `examples/benchmark_qec_beam_refine.py`

Purpose:

- first exploratory beam-search restore experiment

Status:

- exploratory only
- not directly comparable to baseline because it changed more than the final restore stage

### `examples/benchmark_qec_beam_refine_v2.py`

Purpose:

- wrap `benchmark_qec_instance.py`
- replace only the final `restore_slices_with_refinement(...)` step with a beam search

Status:

- correct as an isolated restore-stage experiment
- did not beat the baseline `7`-slice result

### `examples/profile_qec_stages.py`

Purpose:

- measure time spent in parse / greedy / build / anneal / slice / reduce phases

Use this when the next machine is available and you want to profile a new change before launching a long run.

## Reproduction Commands

### Best baseline run on `d5_r05`

Run from the repo root:

```bash
python3 -u examples/benchmark_qec_instance.py   examples/qec_inputs/d5_r05.txt   --seed 308   --sc-target 33   --iters 0   --beta-start 0.1   --beta-stop 10.0   --beta-steps 20   --alpha 64.0   --greedy-alpha 0.1   --greedy-strategy paper_skewed   --post-target-betas 4   --post-target-rounds 6   --slice-replace-rounds 8   --slice-candidate-limit 8   --replace-candidate-limit 8   --restore-refine-rounds 6   --restore-refine-betas 4
```

### Legacy comparison path

```bash
python3 -u examples/benchmark_qec_find_order.py   examples/qec_inputs/d5_r05.txt   --seed 10   --sc-target 33   --trials 5   --iters 6   --beta-start 0.1   --beta-stop 10.0   --beta-steps 20   --slicing-repeat 8   --alpha 64   --greedy-alpha 0.1   --greedy-strategy paper_skewed
```

### Focused multitrial search

Use cautiously on smaller machines:

```bash
python3 -u examples/benchmark_qec_multitrial.py   examples/qec_inputs/d5_r05.txt   --seed-start 201   --seed-count 200   --top-k 25   --max-workers 5   --sc-target 33   --iters 0   --beta-start 0.1   --beta-stop 10.0   --beta-steps 20   --alpha 64.0   --greedy-strategy paper_skewed   --greedy-alphas 0.1
```

For `d5_r05`, do not waste time on `min_dim` or `paper_hybrid`.

### Beam restore experiment

```bash
python3 -u examples/benchmark_qec_beam_refine_v2.py   --beam-width 6   --beam-expand-width 6   examples/qec_inputs/d5_r05.txt   --seed 308   --sc-target 33   --iters 0   --beta-start 0.1   --beta-stop 10.0   --beta-steps 20   --alpha 64.0   --greedy-alpha 0.1   --greedy-strategy paper_skewed   --post-target-betas 4   --post-target-rounds 6   --slice-replace-rounds 8   --slice-candidate-limit 8   --replace-candidate-limit 8   --restore-refine-rounds 6   --restore-refine-betas 4
```

## What To Try Next On The New Server

The current evidence says:

- seed search with the current baseline has plateaued
- schedule tuning plateaued
- beam restore prototype plateaued

So the next serious improvements should target search structure, not just more time.

Recommended next directions:

1. Improve unsliced tree quality before slicing starts

- the reference order is still fundamentally better in the unsliced tree
- the current pipeline usually reaches target only after slicing

2. Try richer initializer families, but only if they can produce good unsliced trees

- `paper_skewed` is still the only strong family on `d5_r05`
- any new family should be judged first by unsliced `tree_sc`, then by final slices

3. Improve the final restore move itself

- the baseline single-bond restore still beats the beam prototype
- next reasonable test would be:
  - exact two-bond restore neighborhoods
  - or restore plus constrained local subtree rebuild around the restored bond

4. Run the same focused seed region on a bigger server

- seeds worth keeping at the top of the queue:
  - `308`
  - `298`
  - `322`
  - `297`
  - `324`

5. Use `profile_qec_stages.py` before large reruns

- confirm where the next server is spending time
- do not assume the current bottleneck remains identical on different hardware

## Known Caveats

- `.venv/` and `build/` were intentionally not committed
- some long-run logs from the original machine live outside this repo and were not imported
- `benchmark_qec_multitrial.py` is useful but still experimental for very large candidate pools
- `benchmark_qec_beam_refine.py` is kept as a prototype record, not as a recommended benchmark path

## Migration Checklist

1. Clone this branch.
2. Install dependencies from `requirements.txt`.
3. Verify scripts compile:

```bash
python3 -m py_compile   examples/benchmark_qec_instance.py   examples/benchmark_qec_find_order.py   examples/benchmark_qec_multitrial.py   examples/benchmark_qec_beam_refine.py   examples/benchmark_qec_beam_refine_v2.py   examples/profile_qec_stages.py
```

4. Start with the seed `308` baseline run above.
5. Compare every new idea against the baseline `7`-slice result before scaling up.
