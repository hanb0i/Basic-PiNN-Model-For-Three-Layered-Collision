# Paper Reproducibility Guide

This repository contains the code and selected artifacts needed to reproduce
the result tables and figures for the manuscript.

## Environment

```bash
python3 -m pip install -r requirements.txt
```

For CPU reproduction:

```bash
export PINN_FORCE_CPU=1
export PYTHONPYCACHEPREFIX=.pycache
export MPLBACKEND=Agg
export MPLCONFIGDIR=.mplconfig
export XDG_CACHE_HOME=.cache
```

Timing values are hardware-dependent. Accuracy artifacts are tied to the saved
checkpoints and the scripts in this repository.

## Fast Paper Check

```bash
PYTHONPYCACHEPREFIX=.pycache python3 scripts/check_paper_reproducibility.py
```

The checker validates the paper-facing CSV/JSON artifacts, required checkpoints,
and selected figures.

## Paper Tables

### PINN-vs-FEM Verification

Artifact:

```text
graphs/data/paper_verification_results.csv
```

Expected:

| Configuration | Mean MAE (%) | Worst MAE (%) |
| --- | ---: | ---: |
| One-layer | 1.62 | 3.15 |
| Three-layer | 5.74 | 10.81 |

The three-layer verification row is reproduced by the final supervised
checkpoint and 100-case random-interior protocol:

```bash
PYTHONPYCACHEPREFIX=.pycache PINN_FORCE_CPU=1 \
python3 scripts/run_random_interior_generalization.py \
  --model-path three-layer-workflow/pinn_model_final.pth \
  --n-cases 100 \
  --seed 20260430 \
  --out-csv graphs/data/three_layer_final_supervised_random100.csv \
  --out-summary graphs/data/three_layer_final_supervised_random100_summary.json
```

The generated summary reports `5.7365%` mean top-surface MAE and `10.8073%`
worst top-surface MAE. The worst case is `random_interior_046` in the generated
CSV, not the extreme-grid case `E=[10,10,10], t=[0.02,0.10,0.02]`.

### One-Layer Ablation

Artifact:

```text
graphs/data/one_layer_paper_ablation.csv
```

Regenerate:

```bash
PYTHONPYCACHEPREFIX=.pycache python3 graphs/scripts/eval_one_layer_ablation.py
```

Current reproducible values with the checkpoints in this workspace:

| Variant | Mean MAE (%) | Worst MAE (%) |
| --- | ---: | ---: |
| Full framework | 1.62 | 3.15 |
| - Compliance-aware scaling | 10.76 | 17.73 |
| Vanilla PINN | 22.17 | 24.01 |

Note: an earlier manuscript draft used `19.45 / 23.34` for the vanilla PINN
row. That exact value requires restoring the external vanilla checkpoint used
for that run. The release package uses the available vanilla checkpoint under
`graphs/data/one_layer_ablation_400_from_scratch/vanilla_parameterized_pinn/`.

### Three-Layer Ablation

Artifact:

```text
graphs/data/three_layer_paper_ablation.csv
```

Expected:

| Variant | Mean MAE (%) | Worst MAE (%) |
| --- | ---: | ---: |
| Full framework | 5.74 | 10.81 |
| Vanilla PINN baseline | 58.59 | 277.37 |
| - Compliance-aware scaling | 14.86 | 19.00 |
| - Layerwise PDE decomposition | 5.47 | 9.63 |
| - Interface continuity | 5.46 | 9.57 |
| - Sparse FEM supervision | 10.93 | 16.12 |

Executable audit note: the release package now includes the final supervised
checkpoint that reproduces the `5.74 / 10.81` three-layer verification row.
However, it still does not include independent reverse-ablation checkpoints
needed to regenerate every three-layer ablation row from scratch. A direct spot
check of the default shipped three-layer checkpoint at the manuscript's stated
extreme case,

```bash
PINN_EVAL_E_VALUES=10,10 \
PINN_EVAL_T1_VALUES=0.02,0.02 \
PINN_EVAL_T2_VALUES=0.10,0.10 \
PINN_EVAL_T3_VALUES=0.02,0.02 \
PINN_EVAL_N_CASES=1 \
PYTHONPYCACHEPREFIX=.pycache \
python3 scripts/verify_three_layer_grid_sweep.py
```

currently reports `6.09%` top-surface MAE for
`E=[10,10,10], t=[0.02,0.10,0.02]`. The `10.81%` value comes from
`random_interior_046` in the 100-case final supervised random-interior
evaluation. Before journal release, update the manuscript wording so the
reported worst case matches this executable protocol.

### Efficiency Tables

Artifacts:

```text
graphs/data/one_layer_efficiency_timing_summary.json
graphs/data/efficiency_timing_summary.json
```

Regenerate:

```bash
PYTHONPYCACHEPREFIX=.pycache PINN_FORCE_CPU=1 python3 scripts/report_one_layer_efficiency_timing.py --repeats 200
PYTHONPYCACHEPREFIX=.pycache PINN_FORCE_CPU=1 python3 scripts/report_efficiency_timing.py --repeats 200
```

The current three-layer timing artifact records one-time training as 94.3 s
(1.57 min), while the manuscript text says 4.5 min. Either cite a log for
4.5 min or update the manuscript.

## Figures

Regenerate scripted figures:

```bash
PYTHONPYCACHEPREFIX=.pycache python3 graphs/make_all_graphs.py --mode plot-only
```

Primary figure artifacts are in `graphs/figures/`. The release package also
creates a top-level `Graphs/` directory matching the manuscript's LaTeX paths.

## Manuscript Cleanup Before Submission

- Remove the LaTeX `To-Do List` section.
- Make figure paths consistent (`Graphs/` in LaTeX, generated from `graphs/`).
- Align the one-layer vanilla ablation row with the checkpoint included in the
  release, or restore the exact earlier vanilla checkpoint.
- Add executable reproduction assets for the three-layer ablation table, or
  update the manuscript to match the released evaluation protocol.
- Correct the manuscript's three-layer worst-case description: `10.81%` is from
  the random-interior 100-case final supervised evaluation, not the stated
  all-stiffness extreme-grid case.
- Align the three-layer training time with the timing artifact or supply the
  supporting log for 4.5 min.
