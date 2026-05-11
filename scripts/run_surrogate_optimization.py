"""Optimize three-layer designs with a balanced PINN surrogate screening metric."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from three_layer_experiment_utils import (
    GRAPHS_DATA_DIR,
    ThreeLayerCase,
    case_grid_top_surface_metrics,
    config,
    evaluate_case_grid,
    evaluate_case_top_surface,
    ensure_output_dirs,
    load_pinn,
    rows_to_csv,
    select_device,
    write_json,
)


BENCHMARK_NE_X = 16
BENCHMARK_NE_Y = 16
BENCHMARK_NE_Z = 8
OBJECTIVE_CHOICES = ("balanced_score", "peak_downward_abs", "mean_patch_abs")


def material_cost(case: ThreeLayerCase, alpha: float) -> float:
    e = np.asarray(case.e, dtype=float)
    t = np.asarray(case.t, dtype=float)
    return float(np.sum((e**alpha) * t))


def latin_hypercube_cases(n_cases: int, seed: int) -> list[ThreeLayerCase]:
    rng = np.random.default_rng(seed)
    lows = np.asarray(
        [
            config.E_RANGE[0],
            config.E_RANGE[0],
            config.E_RANGE[0],
            config.T1_RANGE[0],
            config.T2_RANGE[0],
            config.T3_RANGE[0],
        ],
        dtype=float,
    )
    highs = np.asarray(
        [
            config.E_RANGE[1],
            config.E_RANGE[1],
            config.E_RANGE[1],
            config.T1_RANGE[1],
            config.T2_RANGE[1],
            config.T3_RANGE[1],
        ],
        dtype=float,
    )
    unit = np.empty((n_cases, 6), dtype=float)
    for dim in range(6):
        unit[:, dim] = (rng.permutation(n_cases) + rng.random(n_cases)) / n_cases
    values = lows + unit * (highs - lows)
    return [
        ThreeLayerCase(f"lhs_{idx:03d}", *[float(v) for v in values[idx]])
        for idx in range(n_cases)
    ]


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - values.min()) / (values.max() - values.min() + 1e-8)


def _score_candidates(results: list[dict], lambda_cost: float, cost_alpha: float) -> None:
    disp = np.asarray([float(r["peak_downward_abs"]) for r in results], dtype=float)
    cost = np.asarray([material_cost(r["case"], cost_alpha) for r in results], dtype=float)
    disp_norm = _normalize(disp)
    cost_norm = _normalize(cost)
    score = disp_norm + float(lambda_cost) * cost_norm
    for idx, result in enumerate(results):
        result["material_cost"] = float(cost[idx])
        result["disp_norm"] = float(disp_norm[idx])
        result["cost_norm"] = float(cost_norm[idx])
        result["balanced_score"] = float(score[idx])


def _objective_value(result: dict, objective: str) -> float:
    return float(result[objective])


def _candidate_row(result: dict, rank: int | None = None) -> dict:
    case = result["case"]
    row = {
        "case_id": case.case_id,
        "e1": f"{case.e1:.8g}",
        "e2": f"{case.e2:.8g}",
        "e3": f"{case.e3:.8g}",
        "t1": f"{case.t1:.8g}",
        "t2": f"{case.t2:.8g}",
        "t3": f"{case.t3:.8g}",
        "total_thickness": f"{case.thickness:.8g}",
        "peak_downward_uz": f"{result['peak_downward_uz']:.10g}",
        "peak_downward_abs": f"{result['peak_downward_abs']:.10g}",
        "mean_patch_uz": f"{result['mean_patch_uz']:.10g}",
        "mean_patch_abs": f"{result['mean_patch_abs']:.10g}",
        "material_cost": f"{result['material_cost']:.10g}",
        "disp_norm": f"{result['disp_norm']:.10g}",
        "cost_norm": f"{result['cost_norm']:.10g}",
        "balanced_score": f"{result['balanced_score']:.10g}",
        "pinn_eval_seconds": f"{result['pinn_eval_seconds']:.6f}",
        "n_eval_points": str(result["n_eval_points"]),
    }
    if rank is not None:
        row["rank"] = str(rank)
    return row


def _confirmation_row(rank: int, pinn_result: dict, fem_result: dict, cost_alpha: float) -> dict:
    case = pinn_result["case"]
    cost = material_cost(case, cost_alpha)
    return {
        "rank": str(rank),
        "case_id": case.case_id,
        "e1": f"{case.e1:.8g}",
        "e2": f"{case.e2:.8g}",
        "e3": f"{case.e3:.8g}",
        "t1": f"{case.t1:.8g}",
        "t2": f"{case.t2:.8g}",
        "t3": f"{case.t3:.8g}",
        "total_thickness": f"{case.thickness:.8g}",
        "material_cost": f"{cost:.10g}",
        "pinn_balanced_score": f"{pinn_result['balanced_score']:.10g}",
        "pinn_disp_norm": f"{pinn_result['disp_norm']:.10g}",
        "pinn_cost_norm": f"{pinn_result['cost_norm']:.10g}",
        "pinn_peak_downward_uz": f"{pinn_result['peak_downward_uz']:.10g}",
        "pinn_peak_downward_abs": f"{pinn_result['peak_downward_abs']:.10g}",
        "fem_peak_downward_uz": f"{fem_result['peak_downward_uz']:.10g}",
        "fem_peak_downward_abs": f"{fem_result['peak_downward_abs']:.10g}",
        "abs_gap_peak_downward_abs": f"{abs(pinn_result['peak_downward_abs'] - fem_result['peak_downward_abs']):.10g}",
        "rel_gap_peak_downward_pct": (
            f"{100.0 * abs(pinn_result['peak_downward_abs'] - fem_result['peak_downward_abs']) / max(fem_result['peak_downward_abs'], 1e-12):.6f}"
        ),
        "pinn_mean_patch_uz": f"{pinn_result['mean_patch_uz']:.10g}",
        "pinn_mean_patch_abs": f"{pinn_result['mean_patch_abs']:.10g}",
        "fem_mean_patch_uz": f"{fem_result['mean_patch_uz']:.10g}",
        "fem_mean_patch_abs": f"{fem_result['mean_patch_abs']:.10g}",
        "top_uz_mae_pct": f"{fem_result['top_uz_mae_pct']:.6f}",
        "top_uz_max_pct": f"{fem_result['top_uz_max_pct']:.6f}",
        "volume_mae_pct": f"{fem_result['volume_mae_pct']:.6f}",
        "volume_max_pct": f"{fem_result['volume_max_pct']:.6f}",
        "pinn_eval_seconds": f"{pinn_result['pinn_eval_seconds']:.6f}",
        "fem_seconds": f"{fem_result['fem_seconds']:.6f}",
        "n_eval_points": str(fem_result["n_eval_points"]),
    }


def _candidate_fieldnames(include_rank: bool = False) -> list[str]:
    fields = [
        "case_id",
        "e1",
        "e2",
        "e3",
        "t1",
        "t2",
        "t3",
        "total_thickness",
        "peak_downward_uz",
        "peak_downward_abs",
        "mean_patch_uz",
        "mean_patch_abs",
        "material_cost",
        "disp_norm",
        "cost_norm",
        "balanced_score",
        "pinn_eval_seconds",
        "n_eval_points",
    ]
    return ["rank", *fields] if include_rank else fields


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--calibration-json", default=None)
    parser.add_argument("--objective", choices=OBJECTIVE_CHOICES, default="balanced_score")
    parser.add_argument("--n-candidates", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260423)
    parser.add_argument("--lambda-cost", type=float, default=0.25)
    parser.add_argument("--cost-alpha", type=float, default=1.5)
    parser.add_argument("--surrogate-ne-x", type=int, default=20)
    parser.add_argument("--surrogate-ne-y", type=int, default=20)
    parser.add_argument("--fem-ne-x", type=int, default=BENCHMARK_NE_X)
    parser.add_argument("--fem-ne-y", type=int, default=BENCHMARK_NE_Y)
    parser.add_argument("--fem-ne-z", type=int, default=BENCHMARK_NE_Z)
    parser.add_argument("--out-candidates-csv", default=str(GRAPHS_DATA_DIR / "surrogate_optimization_balanced_candidates.csv"))
    parser.add_argument("--out-topk-csv", default=str(GRAPHS_DATA_DIR / "surrogate_optimization_balanced_topk.csv"))
    parser.add_argument("--out-confirmation-csv", default=str(GRAPHS_DATA_DIR / "surrogate_optimization_balanced_confirmation.csv"))
    parser.add_argument("--out-summary", default=str(GRAPHS_DATA_DIR / "surrogate_optimization_balanced_summary.json"))
    args = parser.parse_args()

    ensure_output_dirs()
    calibration_path = args.calibration_json or os.getenv("PINN_CALIBRATION_JSON")
    if args.calibration_json:
        os.environ["PINN_CALIBRATION_JSON"] = args.calibration_json

    device = select_device()
    pinn, model_path = load_pinn(device, args.model_path)

    print(f"Screening {args.n_candidates} Latin-hypercube designs with the PINN surrogate...")
    candidate_results = []
    for idx, case in enumerate(latin_hypercube_cases(args.n_candidates, args.seed), start=1):
        result = evaluate_case_top_surface(pinn, device, case, args.surrogate_ne_x, args.surrogate_ne_y)
        candidate_results.append(result)
        if idx == 1 or idx == args.n_candidates or idx % 50 == 0:
            print(f"  evaluated {idx}/{args.n_candidates}", flush=True)

    _score_candidates(candidate_results, args.lambda_cost, args.cost_alpha)
    candidate_results.sort(key=lambda r: (_objective_value(r, args.objective), r["peak_downward_abs"], r["material_cost"]))
    top_k = candidate_results[: max(1, min(args.top_k, len(candidate_results)))]

    rows_to_csv(
        Path(args.out_candidates_csv),
        _candidate_fieldnames(),
        [_candidate_row(result) for result in candidate_results],
    )
    rows_to_csv(
        Path(args.out_topk_csv),
        _candidate_fieldnames(include_rank=True),
        [_candidate_row(result, rank=rank) for rank, result in enumerate(top_k, start=1)],
    )

    print(f"Confirming the top {len(top_k)} balanced designs with FEM...")
    confirmation_rows = []
    fem_confirmations = []
    for rank, pinn_result in enumerate(top_k, start=1):
        grid_result = evaluate_case_grid(
            pinn,
            device,
            pinn_result["case"],
            args.fem_ne_x,
            args.fem_ne_y,
            args.fem_ne_z,
        )
        top_metrics = case_grid_top_surface_metrics(grid_result)
        fem_result = {
            "peak_downward_uz": top_metrics["fem_top_metrics"]["peak_downward_uz"],
            "peak_downward_abs": top_metrics["fem_top_metrics"]["peak_downward_abs"],
            "mean_patch_uz": top_metrics["fem_top_metrics"]["mean_patch_uz"],
            "mean_patch_abs": top_metrics["fem_top_metrics"]["mean_patch_abs"],
            "top_uz_mae_pct": grid_result["top_uz_mae_pct"],
            "top_uz_max_pct": grid_result["top_uz_max_pct"],
            "volume_mae_pct": grid_result["volume_mae_pct"],
            "volume_max_pct": grid_result["volume_max_pct"],
            "fem_seconds": grid_result["fem_seconds"],
            "n_eval_points": int(top_metrics["x_grid"].size),
        }
        fem_confirmations.append(fem_result)
        confirmation_rows.append(_confirmation_row(rank, pinn_result, fem_result, args.cost_alpha))
        print(
            f"  rank {rank}: score={pinn_result['balanced_score']:.6g}, "
            f"PINN peak={pinn_result['peak_downward_abs']:.6g}, FEM peak={fem_result['peak_downward_abs']:.6g}",
            flush=True,
        )

    rows_to_csv(
        Path(args.out_confirmation_csv),
        [
            "rank",
            "case_id",
            "e1",
            "e2",
            "e3",
            "t1",
            "t2",
            "t3",
            "total_thickness",
            "material_cost",
            "pinn_balanced_score",
            "pinn_disp_norm",
            "pinn_cost_norm",
            "pinn_peak_downward_uz",
            "pinn_peak_downward_abs",
            "fem_peak_downward_uz",
            "fem_peak_downward_abs",
            "abs_gap_peak_downward_abs",
            "rel_gap_peak_downward_pct",
            "pinn_mean_patch_uz",
            "pinn_mean_patch_abs",
            "fem_mean_patch_uz",
            "fem_mean_patch_abs",
            "top_uz_mae_pct",
            "top_uz_max_pct",
            "volume_mae_pct",
            "volume_max_pct",
            "pinn_eval_seconds",
            "fem_seconds",
            "n_eval_points",
        ],
        confirmation_rows,
    )

    surrogate_objectives = np.asarray([_objective_value(r, args.objective) for r in candidate_results], dtype=float)
    pinn_times = np.asarray([r["pinn_eval_seconds"] for r in candidate_results], dtype=float)
    fem_peak = np.asarray([r["peak_downward_abs"] for r in fem_confirmations], dtype=float)
    top_mae = np.asarray([r["top_uz_mae_pct"] for r in fem_confirmations], dtype=float)
    vol_mae = np.asarray([r["volume_mae_pct"] for r in fem_confirmations], dtype=float)
    fem_gaps = np.asarray(
        [abs(p["peak_downward_abs"] - f["peak_downward_abs"]) for p, f in zip(top_k, fem_confirmations)],
        dtype=float,
    )
    fem_best_idx = int(np.argmin(fem_peak))
    summary = {
        "model_path": str(model_path),
        "seed": int(args.seed),
        "n_candidates": int(args.n_candidates),
        "top_k": int(len(top_k)),
        "calibration_json": calibration_path,
        "benchmark_protocol": "latin_hypercube_pinn_screening_fem_confirmation",
        "optimization_protocol": {
            "method": "latin_hypercube_screening_on_pinn_surrogate",
            "n_candidates": int(args.n_candidates),
            "selection": f"rank by {args.objective}",
        },
        "objective": {
            "name": "balanced_peak_displacement_material_cost",
            "formula": "score = normalized_peak_downward_abs + lambda_cost * normalized_material_cost",
            "displacement_metric": "peak_downward_abs",
            "material_cost": "sum_i E_i^alpha * t_i",
            "lambda_cost": float(args.lambda_cost),
            "cost_alpha": float(args.cost_alpha),
            "reported_surrogate_metric": args.objective,
        },
        "surrogate_grid": {"ne_x": int(args.surrogate_ne_x), "ne_y": int(args.surrogate_ne_y)},
        "fem_confirmation_mesh": {"ne_x": int(args.fem_ne_x), "ne_y": int(args.fem_ne_y), "ne_z": int(args.fem_ne_z)},
        "surrogate_screening": {
            "best_objective_value": float(surrogate_objectives.min()),
            "median_objective_value": float(np.median(surrogate_objectives)),
            "worst_objective_value": float(surrogate_objectives.max()),
            "mean_pinn_eval_seconds": float(pinn_times.mean()),
            "total_pinn_eval_seconds": float(pinn_times.sum()),
        },
        "best_surrogate_design": {
            "rank": 1,
            "case_id": top_k[0]["case"].case_id,
            "e": [float(v) for v in top_k[0]["case"].e],
            "t": [float(v) for v in top_k[0]["case"].t],
            "balanced_score": float(top_k[0]["balanced_score"]),
            "disp_norm": float(top_k[0]["disp_norm"]),
            "cost_norm": float(top_k[0]["cost_norm"]),
            "material_cost": float(top_k[0]["material_cost"]),
            "peak_downward_uz": float(top_k[0]["peak_downward_uz"]),
            "peak_downward_abs": float(top_k[0]["peak_downward_abs"]),
            "mean_patch_uz": float(top_k[0]["mean_patch_uz"]),
            "mean_patch_abs": float(top_k[0]["mean_patch_abs"]),
        },
        "best_fem_peak_design_among_top_k": {
            "rank_within_top_k": fem_best_idx + 1,
            "case_id": top_k[fem_best_idx]["case"].case_id,
            "e": [float(v) for v in top_k[fem_best_idx]["case"].e],
            "t": [float(v) for v in top_k[fem_best_idx]["case"].t],
            "fem_peak_downward_uz": float(fem_confirmations[fem_best_idx]["peak_downward_uz"]),
            "fem_peak_downward_abs": float(fem_confirmations[fem_best_idx]["peak_downward_abs"]),
        },
        "fem_confirmation": {
            "mean_abs_gap_peak_downward_abs": float(fem_gaps.mean()),
            "worst_abs_gap_peak_downward_abs": float(fem_gaps.max()),
            "mean_rel_gap_peak_downward_pct": float(
                np.mean(
                    [
                        100.0 * abs(p["peak_downward_abs"] - f["peak_downward_abs"]) / max(f["peak_downward_abs"], 1e-12)
                        for p, f in zip(top_k, fem_confirmations)
                    ]
                )
            ),
            "top_uz_mae_pct_mean": float(top_mae.mean()),
            "top_uz_mae_pct_worst": float(top_mae.max()),
            "volume_mae_pct_mean": float(vol_mae.mean()),
            "volume_mae_pct_worst": float(vol_mae.max()),
        },
    }
    write_json(Path(args.out_summary), summary)

    print(f"Wrote {args.out_candidates_csv}")
    print(f"Wrote {args.out_topk_csv}")
    print(f"Wrote {args.out_confirmation_csv}")
    print(f"Wrote {args.out_summary}")


if __name__ == "__main__":
    main()
