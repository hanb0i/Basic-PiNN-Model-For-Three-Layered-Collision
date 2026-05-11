"""Run one-layer and three-layer PINN-vs-FEM random generalization studies."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "graphs" / "generalized_study"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _format_params(row: dict[str, str], study: str) -> str:
    if study == "one_layer":
        return f"E={_float(row, 'E'):.4g}, t={_float(row, 'thickness'):.4g}"
    return (
        f"E1={_float(row, 'e1'):.4g}, E2={_float(row, 'e2'):.4g}, E3={_float(row, 'e3'):.4g}, "
        f"t1={_float(row, 't1'):.4g}, t2={_float(row, 't2'):.4g}, t3={_float(row, 't3'):.4g}"
    )


def _representative_rows(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda r: _float(r, "top_uz_mae_pct"))
    indices = [0, len(ordered) // 2, len(ordered) - 1]
    picked = []
    seen = set()
    for idx in indices:
        row = ordered[idx]
        case_id = row.get("case_id", "")
        if case_id not in seen:
            picked.append(row)
            seen.add(case_id)
    for row in ordered:
        if len(picked) >= limit:
            break
        case_id = row.get("case_id", "")
        if case_id not in seen:
            picked.append(row)
            seen.add(case_id)
    return picked[:limit]


def _write_table(path: Path, studies: dict[str, dict], representative_limit: int) -> None:
    lines = ["# PINN Generalization Study", ""]
    for label, study_key in [("One-layer", "one_layer"), ("Three-layer", "three_layer")]:
        info = studies[study_key]
        summary = info["summary"]
        rows = info["rows"]
        lines.extend(
            [
                f"## {label} random samples, n={summary.get('n_cases', len(rows))}",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Top uz MAE mean (%) | {summary.get('top_uz_mae_pct_mean', 0.0):.3f} |",
                f"| Top uz MAE worst (%) | {summary.get('top_uz_mae_pct_worst', 0.0):.3f} |",
                f"| Volume MAE mean (%) | {summary.get('volume_mae_pct_mean', 0.0):.3f} |",
                f"| Volume MAE worst (%) | {summary.get('volume_mae_pct_worst', 0.0):.3f} |",
                f"| Avg displacement relative error mean (%) | {summary.get('avg_displacement_relative_error_pct_mean', 0.0):.3f} |",
                "",
                "| Representative case | Top uz MAE (%) | Volume MAE (%) | Peak FEM uz | Peak PINN uz | Parameters |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in _representative_rows(rows, representative_limit):
            lines.append(
                "| {case_id} | {top:.3f} | {vol:.3f} | {peak_fem:.6g} | {peak_pinn:.6g} | {params} |".format(
                    case_id=row.get("case_id", ""),
                    top=_float(row, "top_uz_mae_pct"),
                    vol=_float(row, "volume_mae_pct"),
                    peak_fem=_float(row, "peak_fem_uz"),
                    peak_pinn=_float(row, "peak_pinn_uz"),
                    params=_format_params(row, study_key),
                )
            )
        lines.append("")
    path.write_text("\n".join(lines))


def _run_script(script: str, n_cases: int, seed: int, mesh: tuple[int, int, int], out_csv: Path, out_summary: Path, env: dict) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / script),
        "--n-cases",
        str(n_cases),
        "--seed",
        str(seed),
        "--ne-x",
        str(mesh[0]),
        "--ne-y",
        str(mesh[1]),
        "--ne-z",
        str(mesh[2]),
        "--out-csv",
        str(out_csv),
        "--out-summary",
        str(out_summary),
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def _run_count(args: argparse.Namespace, n_cases: int, env: dict) -> dict:
    mesh = (args.ne_x, args.ne_y, args.ne_z)
    seed = args.seed
    outputs = {
        "one_layer": {
            "csv": OUT_DIR / f"one_layer_random_{n_cases}.csv",
            "summary": OUT_DIR / f"one_layer_random_{n_cases}_summary.json",
        },
        "three_layer": {
            "csv": OUT_DIR / f"three_layer_random_{n_cases}.csv",
            "summary": OUT_DIR / f"three_layer_random_{n_cases}_summary.json",
        },
    }

    if not args.skip_one_layer:
        _run_script(
            "run_one_layer_generalization.py",
            n_cases,
            seed,
            mesh,
            outputs["one_layer"]["csv"],
            outputs["one_layer"]["summary"],
            env,
        )
    if not args.skip_three_layer:
        _run_script(
            "run_random_interior_generalization.py",
            n_cases,
            seed,
            mesh,
            outputs["three_layer"]["csv"],
            outputs["three_layer"]["summary"],
            env,
        )

    studies = {}
    for key, paths in outputs.items():
        studies[key] = {
            "csv": str(paths["csv"]),
            "summary_path": str(paths["summary"]),
            "summary": _read_json(paths["summary"]),
            "rows": _read_csv(paths["csv"]),
        }

    combined = {
        "seed": seed,
        "n_cases": n_cases,
        "mesh": {"ne_x": args.ne_x, "ne_y": args.ne_y, "ne_z": args.ne_z},
        "one_layer": {
            "csv": studies["one_layer"]["csv"],
            "summary": studies["one_layer"]["summary"],
        },
        "three_layer": {
            "csv": studies["three_layer"]["csv"],
            "summary": studies["three_layer"]["summary"],
        },
    }
    combined_path = OUT_DIR / f"generalization_{n_cases}_summary.json"
    table_path = OUT_DIR / f"generalization_{n_cases}_table.md"
    _write_json(combined_path, combined)
    _write_table(table_path, studies, args.representative_cases)
    print(f"Wrote {combined_path}", flush=True)
    print(f"Wrote {table_path}", flush=True)
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-counts", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--ne-x", type=int, default=16)
    parser.add_argument("--ne-y", type=int, default=16)
    parser.add_argument("--ne-z", type=int, default=8)
    parser.add_argument("--representative-cases", type=int, default=5)
    parser.add_argument("--force-cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-one-layer", action="store_true")
    parser.add_argument("--skip-three-layer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONPYCACHEPREFIX", str(REPO_ROOT / ".pycache"))
    if args.force_cpu:
        env["PINN_FORCE_CPU"] = "1"

    combined_runs = {}
    for n_cases in args.case_counts:
        combined_runs[str(n_cases)] = _run_count(args, int(n_cases), env)

    index_path = OUT_DIR / "generalization_study_index.json"
    _write_json(index_path, combined_runs)
    print(f"Wrote {index_path}", flush=True)


if __name__ == "__main__":
    main()
