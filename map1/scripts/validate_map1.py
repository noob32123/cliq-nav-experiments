"""Offline City Park metrics and route-clustered 95% confidence intervals.

Run: python map1/scripts/validate_map1.py
Original trajectories and manuscript text are read-only inputs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np

try:
    from . import evaluate_nav_metrics as metrics
except ImportError:
    import evaluate_nav_metrics as metrics

MAP = Path(__file__).resolve().parents[1]
ROOT = MAP.parent
METRICS = ("SR", "NE", "SPL", "OR", "TL")
PCT = (0, 2, 3)
SEED = 20260823
RESAMPLES = 10_000
REPEATS = 10
MODELS = {
    "Qwen-VL-Max": ("", "qwen-vl-max"),
    "Qwen-VL-Plus": ("", "qwen-vl-plus"),
    "Qwen3-VL-Flash": ("", "qwen3-vl-flash"),
    "Qwen3.5-35B-A3B + CLIP": ("", "qwen3.5-35b-a3b_CLIP"),
    "Qwen3.5-35B-A3B": ("", "qwen3.5-35b-a3b"),
    "Qwen3.5-Flash + CLIP": ("", "qwen3.5-flash_CLIP"),
    "Qwen3.5-Flash": ("", "qwen3.5-flash"),
    "CityNavAgent": ("baselines/city_nav", "CityNavAgent"),
    "AerialVLA": ("baselines/aerialvla", "AerialVLA"),
    "FineCog-Nav": ("baselines/finecog_nav", "FineCogNav"),
}
PAIRS = {
    "35B (CLIP minus no CLIP)": ("Qwen3.5-35B-A3B + CLIP", "Qwen3.5-35B-A3B"),
    "Flash (CLIP minus no CLIP)": ("Qwen3.5-Flash + CLIP", "Qwen3.5-Flash"),
}


def points(path: Path) -> list[tuple[float, float, float]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"X", "Y", "Z"}.issubset(reader.fieldnames or ()):
            raise ValueError(f"Invalid trajectory header: {path}")
        values = []
        for line, row in enumerate(reader, 2):
            try:
                point = tuple(float(row[axis]) for axis in ("X", "Y", "Z"))
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"Invalid coordinate at {path}:{line}") from exc
            if not np.isfinite(point).all():
                raise ValueError(f"Non-finite coordinate at {path}:{line}")
            values.append(point)
    if not values:
        raise ValueError(f"Empty trajectory: {path}")
    return values


def flights(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise ValueError(f"Missing flight folder: {folder}")
    paths = sorted(folder.glob("flight_*.csv"))
    indices = []
    for path in paths:
        match = re.match(r"flight_(\d{2})_", path.name)
        if not match:
            raise ValueError(f"Invalid flight filename: {path}")
        indices.append(int(match.group(1)))
    if len(paths) != REPEATS or sorted(indices) != list(range(1, REPEATS + 1)):
        raise ValueError(f"Incomplete or duplicate flight set: {folder}; indices={indices}")
    return paths


def load(data_root: Path) -> dict[str, np.ndarray]:
    goals = json.loads((data_root / "goaldest/goals.json").read_text(encoding="utf-8"))
    if set(goals) != set(metrics.SCENARIO_CONFIG):
        raise ValueError("Expected the 14 frozen City Park goals")
    matrices = {}
    for model, (subdir, prefix) in MODELS.items():
        route_scores = []
        for number in range(1, 15):
            key = f"data_{number}"
            cfg = metrics.SCENARIO_CONFIG[key]
            stored_goal = np.asarray(goals[key], dtype=float)
            if stored_goal.shape != (3,) or not np.isfinite(stored_goal).all():
                raise ValueError(f"Invalid goal for {key}")
            # The pre-existing coordinate JSON differs on routes 7 and 13.
            # Preserve the evaluator used by the existing published summaries.
            goal = cfg["goal_position"]
            folder = data_root / subdir / key / f"{prefix}_{cfg['name']}"
            episodes = []
            for path in flights(folder):
                score = metrics.compute_episode_metrics(points(path), cfg["start_position"], goal,
                                                        cfg["success_threshold"])
                episodes.append([score[name] for name in METRICS])
            route_scores.append(episodes)
        matrix = np.asarray(route_scores, dtype=float)
        if matrix.shape != (14, REPEATS, len(METRICS)) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid metric matrix: {model}")
        matrices[model] = matrix
    return matrices


def mean_and_ci(matrix: np.ndarray, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    route_means = matrix.mean(axis=1)
    estimate = route_means.mean(axis=0)
    distribution = route_means[samples].mean(axis=1)
    low, high = np.quantile(distribution, [0.025, 0.975], axis=0)
    return estimate, low, high


def metric_record(matrix: np.ndarray, samples: np.ndarray) -> dict:
    estimate, low, high = mean_and_ci(matrix, samples)
    return {name: {"mean": float(estimate[i]), "low": float(low[i]), "high": float(high[i])}
            for i, name in enumerate(METRICS)}


def analyze(matrices: dict[str, np.ndarray], samples: np.ndarray) -> dict:
    return {
        "routes": 14, "episodes_per_route_per_method": REPEATS,
        "bootstrap_resamples": RESAMPLES, "seed": SEED, "confidence_level": 0.95,
        "method": "route-clustered percentile bootstrap; identical route samples across methods and paired differences",
        "units": {"SR": "fraction", "NE": "m", "SPL": "fraction", "OR": "fraction", "TL": "m"},
        "success_radii_m": {key: cfg["success_threshold"] for key, cfg in metrics.SCENARIO_CONFIG.items()},
        "endpoint_policy": "last recorded CSV point; preserves the existing map1 evaluator",
        "goal_source": "map1/scripts/evaluate_nav_metrics.py:SCENARIO_CONFIG",
        "methods": {name: metric_record(matrix, samples) for name, matrix in matrices.items()},
        "differences": {name: metric_record(matrices[cue] - matrices[control], samples)
                        for name, (cue, control) in PAIRS.items()},
    }


def cell(value: float, index: int) -> str:
    return f"{value * 100:.1f}" if index in PCT else f"{value:.2f}"


def write_ci(path: Path, entries: dict, label: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([label, *[f"{m}_{field}" for m in METRICS for field in ("mean", "ci_low", "ci_high")]])
        for name, record in entries.items():
            writer.writerow([name, *[cell(record[m][field], i) for i, m in enumerate(METRICS)
                                     for field in ("mean", "low", "high")]])


def manuscript_comparison(result: dict) -> list[dict]:
    source = ROOT / "paper/eaai_submission/paper_eaai.tex"
    if not source.exists():
        return []  # Manuscript deliberately excluded from the public experiment archive.
    text = source.read_text(encoding="utf-8")
    table = text.split(r"\label{tab:ci_results}", 1)[1].split(r"\end{table}", 1)[0]
    comparisons = []
    pair = None
    for line in table.splitlines():
        if r"\multicolumn{4}" in line:
            pair = "35B (CLIP minus no CLIP)" if "35B" in line else "Flash (CLIP minus no CLIP)"
        match = re.match(r"(SR|NE|SPL|OR|TL)\s", line)
        if not match or pair is None:
            continue
        metric = match.group(1)
        index = METRICS.index(metric)
        cue, control = PAIRS[pair]
        for column, (kind, name) in zip(line.split("&")[1:], [("methods", cue), ("methods", control), ("differences", pair)]):
            published = [float(x) for x in re.findall(r"[+-]?\d+(?:\.\d+)?", column)]
            if len(published) != 3:
                raise ValueError(f"Cannot parse published CI cell: {column}")
            recomputed = [float(cell(result[kind][name][metric][field], index)) for field in ("mean", "low", "high")]
            comparisons.append({"kind": kind, "name": name, "metric": metric,
                                "published": published, "recomputed": recomputed,
                                "matches_at_published_precision": published == recomputed})
    if len(comparisons) != 30:
        raise ValueError("Expected 30 published map1 CI cells")
    return comparisons


def write_report(out: Path, result: dict, comparisons: list[dict]) -> None:
    mismatches = [row for row in comparisons if not row["matches_at_published_precision"]]
    paired = [row for row in comparisons if row["kind"] == "differences"]
    report = ["# map1 95% 置信验证与论文核对", "",
              "原始轨迹、目标坐标和论文文字均保持原样。本验证覆盖 10 个方法、14 条路线、每条 10 次实验，共 1,400 次实验。", "",
              f"路线级 bootstrap：{RESAMPLES:,} 次，NumPy default_rng，种子 {SEED}，百分位区间 [2.5%, 97.5%]。",
              "同一次抽样保留所选路线的全部 10 次实验；所有方法及两组配对差值共用抽样路线索引。", "",
              "指标沿用现有 map1 评估器：XY 平面、CSV 最后一个记录点、原有逐路线成功半径及 SPL 公式。",
              "map1 CSV 不含 map2 的 navigation_end_position 元数据；本次没有推断或裁剪落地轨迹，也没有改动已有指标口径。", "",
              "## 原有目标坐标差异", "",
              "原始 goals.json 与现有指标配置在部分路线不一致。本验证沿用现有指标配置，保持两种来源的原始数值，实际计算配置见 evaluation_config.json。", "",
              "| 路线 | 现有指标采用坐标 | goals.json 保存坐标 |", "|---|---|---|"]
    for row in result["coordinate_file_differences"]:
        report.append(f"| {row['route']} | {row['evaluation_goal']} | {row['stored_goal']} |")
    report.extend(["",
              "JSON 保留完整浮点精度；CSV 的 SR/SPL/OR 以百分数、NE/TL 以米显示。", "",
              "## 论文核对", "",
              "核对来源：paper/eaai_submission/paper_eaai.tex 中 tab:ci_results。",
              f"两组配对差值的 {len(paired)} 个单元格中，{sum(r['matches_at_published_precision'] for r in paired)} 个与论文显示精度一致。",
              f"全部 {len(comparisons)} 个单元格中，有 {len(mismatches)} 个存在差异；原始脚本缺失，不能据此确定历史单模型区间的抽样实现。",
              "保持论文数值，以下列出可复现结果的差异；没有更改种子或挑选抽样结果以匹配原表。", "",
              "| 方法 | 指标 | 论文 mean [low, high] | 复算 mean [low, high] |",
              "|---|---|---|---|"])
    for row in mismatches:
        report.append(f"| {row['name']} | {row['metric']} | {row['published']} | {row['recomputed']} |")
    report.extend(["", "机器可读核对结果见 manuscript_comparison.json。", ""])
    (out / "verification_report.md").write_text("\n".join(report), encoding="utf-8")
    (out / "manuscript_comparison.json").write_text(json.dumps(comparisons, indent=2, allow_nan=False), encoding="utf-8")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=MAP)
    parser.add_argument("--output-dir", type=Path, default=MAP / "validation")
    args = parser.parse_args(argv)
    matrices = load(args.data_root)
    samples = np.random.default_rng(SEED).integers(0, 14, (RESAMPLES, 14))
    result = analyze(matrices, samples)
    stored_goals = json.loads((args.data_root / "goaldest/goals.json").read_text(encoding="utf-8"))
    result["coordinate_file_differences"] = [
        {"route": key, "evaluation_goal": list(cfg["goal_position"]), "stored_goal": stored_goals[key]}
        for key, cfg in metrics.SCENARIO_CONFIG.items()
        if list(cfg["goal_position"]) != stored_goals[key]
    ]
    comparisons = manuscript_comparison(result)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "bootstrap_results.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    (out / "evaluation_config.json").write_text(json.dumps(metrics.SCENARIO_CONFIG, indent=2), encoding="utf-8")
    write_ci(out / "method_ci.csv", result["methods"], "method")
    write_ci(out / "paired_route_difference_ci.csv", result["differences"], "comparison")
    with (out / "routewise_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["route", "method", "episodes", "SR_percent", "NE_m", "SPL_percent", "OR_percent", "TL_m"])
        for route in range(14):
            for name, matrix in matrices.items():
                means = matrix[route].mean(axis=0)
                writer.writerow([f"{route + 1:02d}", name, REPEATS, *[cell(means[i], i) for i in range(5)]])
    write_report(out, result, comparisons)
    print(f"Verified {len(matrices)} methods, 14 routes, {len(matrices) * 14 * REPEATS} episodes; outputs: {out}")


if __name__ == "__main__":
    main()
