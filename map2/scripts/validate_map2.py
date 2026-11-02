"""Recompute Downtown West metrics and route-clustered uncertainty from frozen CSV flights.

Run from the repository root: python map2/scripts/validate_map2.py
No simulator, model API, or goal search is invoked. Outputs are written to
map2/validation/.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT / "map2"
OUT = MAP / "validation"
ROUTES = json.loads((MAP / "formal_20260909_170427" / "routes.json").read_text(encoding="utf-8"))
GOALS = json.loads((MAP / "goaldest" / "goals.json").read_text(encoding="utf-8"))
MODELS = {
    "Qwen3.5-35B-A3B + CLIP": "qwen3.5-35b-a3b_CLIP_",
    "Qwen3.5-35B-A3B": "qwen3.5-35b-a3b_",
    "AerialVLA": "AerialVLA_INT8_Adapted_",
    "FineCog-Nav": "FineCogNav_Qwen_Adapted_",
    "CityNavAgent": "CityNavAgent_Adapted_internvl2-26b_",
}
METRICS = ("SR", "NE", "SPL", "OR", "TL")
PCT = (0, 2, 3)
REPEATS = 10
SEED = 20261014
RESAMPLES = 10_000


def points(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    values = np.asarray([[float(row[axis]) for axis in ("X", "Y")] for row in rows], dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) == 0:
        raise ValueError(f"Invalid trajectory: {path}")
    return values


def score(path: Path, start: np.ndarray, goal: np.ndarray, radius: float) -> np.ndarray:
    xy = points(path)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    nav_end = np.asarray(metadata["navigation_end_position"][:2], dtype=float)
    matches = np.flatnonzero(np.linalg.norm(xy - nav_end, axis=1) < 1e-6)
    if not len(matches):
        raise ValueError(f"Navigation endpoint absent from trajectory: {path}")
    # The CSV may also record a subsequent landing drift. Score only navigation.
    xy = xy[:matches[-1] + 1]
    ne = float(np.linalg.norm(xy[-1] - goal))
    tl = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    oracle = float(np.linalg.norm(xy - goal, axis=1).min())
    length = float(np.linalg.norm(start - goal))
    success = float(ne <= radius)
    spl = success * length / max(length, tl) if max(length, tl) else 0.0
    return np.asarray([success, ne, spl, float(oracle <= radius), tl], dtype=float)


def load(radius: float) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for model, prefix in MODELS.items():
        matrix = np.empty((30, REPEATS, len(METRICS)), dtype=float)
        for route_number in range(1, 31):
            key = f"{route_number:02d}"
            folder = MAP / f"data_{route_number}"
            matches = [d for d in folder.iterdir() if d.is_dir() and d.name.startswith(prefix)
                       and (model != "Qwen3.5-35B-A3B" or "_CLIP_" not in d.name)]
            if len(matches) != 1:
                raise ValueError(f"Expected one {model} folder for {key}: {matches}")
            flights = sorted(matches[0].glob("flight_*.csv"))
            if len(flights) != REPEATS or [int(p.name.split("_")[1]) for p in flights] != list(range(1, 11)):
                raise ValueError(f"Incomplete flight set: {model}, {key}")
            start = np.asarray(ROUTES[key]["start_position"][:2], dtype=float)
            goal = np.asarray(GOALS[key][:2], dtype=float)
            if not np.isfinite(goal).all():
                raise ValueError(f"Invalid goal for route {key}")
            matrix[route_number - 1] = np.stack([score(p, start, goal, radius) for p in flights])
        values[model] = matrix
    return values


def mean_and_ci(matrix: np.ndarray, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    route_means = matrix.mean(axis=1)
    estimate = route_means.mean(axis=0)
    bootstrap = route_means[samples].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    return estimate, low, high


def cell(x: float, i: int) -> str:
    scale = 100 if i in PCT else 1
    return f"{x * scale:.1f}" if i in PCT else f"{x:.2f}"


def summary(radius: float, samples: np.ndarray) -> tuple[dict, dict[str, np.ndarray]]:
    matrices = load(radius)
    data = {"radius_m": radius, "routes": 30, "episodes_per_route_per_method": 10,
            "bootstrap_resamples": RESAMPLES, "seed": SEED, "method": "route-clustered percentile bootstrap; paired by route for differences",
            "methods": {}, "differences": {}}
    for model, matrix in matrices.items():
        estimate, low, high = mean_and_ci(matrix, samples)
        data["methods"][model] = {METRICS[i]: {"mean": float(estimate[i]), "low": float(low[i]), "high": float(high[i])}
                                   for i in range(len(METRICS))}
    primary = matrices["Qwen3.5-35B-A3B + CLIP"]
    for control in list(MODELS)[1:]:
        delta = primary - matrices[control]
        estimate, low, high = mean_and_ci(delta, samples)
        data["differences"][control] = {METRICS[i]: {"mean": float(estimate[i]), "low": float(low[i]), "high": float(high[i])}
                                        for i in range(len(METRICS))}
    return data, matrices


def write_table(path: Path, data: dict, difference: bool = False) -> None:
    entries = data["differences" if difference else "methods"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["comparison" if difference else "method", *[f"{m}_{field}" for m in METRICS for field in ("mean", "ci_low", "ci_high")]])
        for name, metrics in entries.items():
            writer.writerow([name, *[cell(metrics[m][field], i) for i, m in enumerate(METRICS)
                                     for field in ("mean", "low", "high")]])


def main(argv=None) -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args(argv)
    OUT = args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    samples = rng.integers(0, 30, (RESAMPLES, 30))
    result, matrices = summary(6.0, samples)
    (OUT / "bootstrap_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_table(OUT / "method_ci.csv", result)
    write_table(OUT / "paired_route_difference_ci.csv", result, True)

    with (OUT / "routewise_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["route", "method", "episodes", "SR_percent", "NE_m", "SPL_percent", "OR_percent", "TL_m"])
        for route in range(30):
            for name, matrix in matrices.items():
                means = matrix[route].mean(axis=0)
                writer.writerow([f"{route + 1:02d}", name, REPEATS, *[cell(means[i], i) for i in range(5)]])

    primary = matrices["Qwen3.5-35B-A3B + CLIP"]
    control = matrices["Qwen3.5-35B-A3B"]
    route_diffs = primary.mean(axis=1) - control.mean(axis=1)
    loo = np.asarray([(route_diffs.sum(axis=0) - route_diffs[i]) / 29 for i in range(30)])
    with (OUT / "leave_one_route_out.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["omitted_route", *[f"delta_{m}" for m in METRICS]])
        for i, row in enumerate(loo):
            writer.writerow([f"{i+1:02d}", *[cell(row[j], j) for j in range(5)]])

    sensitivities = []
    for radius in (4.0, 6.0, 8.0):
        analysis, _ = (result, matrices) if radius == 6.0 else summary(radius, samples)
        diff = analysis["differences"]["Qwen3.5-35B-A3B"]
        sensitivities.append({"radius_m": radius, "SR_delta_pp": round(diff["SR"]["mean"] * 100, 3),
                              "SR_ci_low": round(diff["SR"]["low"] * 100, 3), "SR_ci_high": round(diff["SR"]["high"] * 100, 3),
                              "SPL_delta_pp": round(diff["SPL"]["mean"] * 100, 3),
                              "SPL_ci_low": round(diff["SPL"]["low"] * 100, 3), "SPL_ci_high": round(diff["SPL"]["high"] * 100, 3)})
    with (OUT / "radius_sensitivity.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sensitivities[0]))
        writer.writeheader()
        writer.writerows(sensitivities)

    print("methods", list(MODELS), "episodes", sum(x.shape[0]*x.shape[1] for x in matrices.values()))
    for name in MODELS:
        print(name, [cell(result["methods"][name][m]["mean"], i) for i, m in enumerate(METRICS)])
    print("35B difference", [cell(result["differences"]["Qwen3.5-35B-A3B"][m]["mean"], i) for i, m in enumerate(METRICS)])
    print("radius sensitivity", sensitivities)
    print("LOO SR delta range", (loo[:, 0].min()*100, loo[:, 0].max()*100))
    print("LOO SPL delta range", (loo[:, 2].min()*100, loo[:, 2].max()*100))


if __name__ == "__main__":
    main()
