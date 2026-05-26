# Release Checklist

- [ ] Run `PYTHONPYCACHEPREFIX=.pycache python3 scripts/check_paper_reproducibility.py`.
- [ ] Confirm the paper-facing tables:
  - `graphs/data/paper_verification_results.csv`
  - `graphs/data/one_layer_paper_ablation.csv`
  - `graphs/data/three_layer_paper_ablation.csv`
  - `graphs/data/one_layer_efficiency_timing_summary.json`
  - `graphs/data/efficiency_timing_summary.json`
- [ ] Confirm checkpoints:
  - `one-layer-workflow/pinn_model.pth`
  - `three-layer-workflow/pinn_model.pth`
  - `pinn-workflow/pinn_model.pth`
  - `graphs/data/one_layer_ablation_400_from_scratch/vanilla_parameterized_pinn/pinn_model.pth`
- [ ] Remove manuscript TODOs before submission.
- [ ] Decide whether MIT is the intended license. Change `LICENSE` if needed.
- [ ] Tag the release, e.g. `v1.0-paper`.
- [ ] Archive the GitHub release on Zenodo and cite the DOI.
