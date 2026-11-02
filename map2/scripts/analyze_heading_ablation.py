"""Offline 30 x 10 heading-ablation analysis; whole-route paired bootstrap."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from map2.scripts import heading_ablation as h
from map1.scripts import validate_map1 as validation
from map2.scripts import validate_map2 as legacy
from map2.scripts.monitor_heading_ablation import confirmed_reset_valid, inspect_arm


def analyze_batch(directory: Path) -> dict:
    directory = directory.resolve()
    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    if state.get("exclude_from_statistics"):
        raise ValueError("Archived invalid scene/workflow batch is excluded from statistics")
    routes = json.loads((directory / "routes.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "input_manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["sha256"].items():
        if h.sha256(ROOT / name) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    for name, digest in manifest.get("external_sha256", {}).items():
        if h.sha256(Path(name)) != digest:
            raise ValueError(f"Frozen scene map changed: {name}")
    if h.sha256(Path(manifest["global_settings_path"])) != manifest["global_settings_sha256"]:
        raise ValueError("Global AirSim settings changed")
    formal = [j for j in state["jobs"] if j["kind"] == "formal"]
    expected = {(r, k) for r in range(1, 31) for k in range(1, 11)}
    if len(formal) != 300 or {(j["route"], j["repeat"]) for j in formal} != expected or any(j["status"] != "complete" for j in formal):
        raise ValueError("Expected 300 unique complete formal experiments (30 routes x 10)")
    baseline, ablated, episodes, checks = [], [], [], []
    for r in range(1, 31):
        route = routes[str(r)]
        baseline_folder = h.MAP / f"data_{r}" / f"qwen3.5-35b-a3b_CLIP_{route['name']}"
        full_scores, ablation_scores = [], []
        for k, path in enumerate(validation.flights(baseline_folder), 1):
            # Existing map2 CSVs include landing rows; use the original saved navigation endpoint.
            validation.points(path)  # Reject NaNs and malformed rows before original scorer.
            values = legacy.score(path, np.asarray(route["start_position"][:2]),
                                  np.asarray(route["goal_position"][:2]), route["success_threshold"])
            score = dict(zip(validation.METRICS, map(float, values)))
            full_scores.append(values.tolist())
            episodes.append({"method": "existing_full", "route": r, "repeat": k, **score})
        for job in sorted((j for j in formal if j["route"] == r), key=lambda j: j["repeat"]):
            path = directory / job["result_path"]
            record = json.loads(path.read_text(encoding="utf-8"))
            trajectory = path.parent / "trajectory.csv"
            h.validate_recent_history(route)
            arms = {}
            for arm in ("baseline", "ablation"):
                arms[arm] = json.loads((path.parent / f"{arm}_initialization.json").read_text(encoding="utf-8"))
                if inspect_arm(route, arms[arm], arm)["issues"]:
                    raise ValueError(f"Independent initialization check failed: {job['id']} {arm}")
            recalculated_check = h.heading_check(route, arms["baseline"], arms["ablation"])
            if not recalculated_check["passed"]:
                raise ValueError(f"Recalculated heading delta failed: {job['id']}")
            check = record["heading_check"]
            if state["config"].get("scene_frame_reference"):
                for arm in ("baseline", "ablation"):
                    initialization = json.loads((path.parent / f"{arm}_reset_takeoff.json").read_text(encoding="utf-8"))
                    if not confirmed_reset_valid(route, initialization):
                        raise ValueError(f"Invalid scene frame/start position: {job['id']} {arm}")
            if (record["id"] != job["id"] or not record["complete"] or not check["passed"] or
                    record["reason"] not in {"model_stop", "max_iterations", "collision"} or
                    record["first_frame_yaw_error_deg"] > 1 or record["first_frame_camera_yaw_error_deg"] > 1 or
                    h.sha256(trajectory) != record["trajectory_sha256"]):
                raise ValueError(f"Invalid completed flight: {job['id']}")
            score = h.metrics.compute_episode_metrics(validation.points(trajectory), route["start_position"],
                                                      route["goal_position"], route["success_threshold"])
            if any(abs(score[m] - record["metrics"][m]) > 1e-10 for m in validation.METRICS):
                raise ValueError(f"Recorded metric mismatch: {job['id']}")
            ablation_scores.append([score[m] for m in validation.METRICS])
            episodes.append({"method": "last_segment_delta", "route": r, "repeat": job["repeat"],
                             **{m: score[m] for m in validation.METRICS}})
            checks.append(check)
        baseline.append(full_scores)
        ablated.append(ablation_scores)
    full, ablation = np.asarray(baseline), np.asarray(ablated)
    if full.shape != (30, 10, 5) or ablation.shape != full.shape or not np.isfinite([full, ablation]).all():
        raise ValueError("Invalid metric matrices")
    samples = np.random.default_rng(20260823).integers(0, 30, (10000, 30))
    methods = {"existing_full": validation.metric_record(full, samples),
               "last_segment_delta": validation.metric_record(ablation, samples)}
    # Pair route means only: repeat numbers from different batches are not matched experimental pairs.
    difference = full.mean(axis=1) - ablation.mean(axis=1)
    result = {"routes": 30, "repeats": 10, "seed": 20260823, "bootstrap_resamples": 10000,
              "confidence_level": 0.95, "bootstrap_unit": "whole route; all 10 repeats retained",
              "pairing": "same route indices across both methods; no episode pairing",
              "difference_direction": "existing_full minus last_segment_delta",
              "units": {"SR": "fraction", "NE": "m", "SPL": "fraction", "OR": "fraction", "TL": "m"},
              "methods": methods, "differences": {"full_minus_ablation": validation.metric_record(difference[:, None, :], samples)},
              "heading_verification": {"passed": len(checks), "required": 300,
                                       "max_actual_delta_error_deg": max(c["actual_delta_error_deg"] for c in checks),
                                       "max_initial_yaw_error_deg": max(c["initial_yaw_error_deg"] for c in checks)},
              "config": state["config"], "protected_inputs_verified": True}
    out = directory / "analysis"
    out.mkdir(exist_ok=True)
    h.atomic_json(out / "bootstrap_results.json", result)
    np.save(out / "bootstrap_route_indices.npy", samples)
    np.savez(out / "episode_metric_matrices.npz", existing_full=full, last_segment_delta=ablation)
    h.write_csv(out / "episode_metrics.csv", episodes)
    rows = []
    for name, entries in {**methods, **result["differences"]}.items():
        rows.append({"method_or_difference": name, **{f"{metric}_{field}": entries[metric][field]
                    for metric in validation.METRICS for field in ("mean", "low", "high")}})
    h.write_csv(out / "method_and_difference_ci.csv", rows)
    route_rows = []
    for r in range(30):
        for name, matrix in (("existing_full", full), ("last_segment_delta", ablation)):
            route_rows.append({"method": name, "route": r + 1,
                               **dict(zip(validation.METRICS, map(float, matrix[r].mean(axis=0))))})
    h.write_csv(out / "routewise_metrics.csv", route_rows)
    report = ["# map2 Heading Prediction 消融核验", "",
              "正式实验 30 条路线 × 10 次；冒烟实验不进入统计。", "",
              "消融初始化转角 = map2 固定路线的原初始化转角 + wrap(末段 XY 朝向 − 加权 XY 朝向)。",
              "未将最终 yaw 强制设置为末段绝对朝向；初始化反馈仅修正相对转角执行误差。", "",
              "每次独立复现原初始化配置，再重置运行消融配置；这不是旧基线的 yaw 遥测。",
              f"300 次朝向差核验通过，最大角差误差 {result['heading_verification']['max_actual_delta_error_deg']:.6f}°。", "",
              "指标公式、XY 距离与逐路线成功半径沿用 map2 评估器。",
              "10,000 次路线级配对 bootstrap，种子 20260823；同一抽样路线索引用于两个方法及差值，不配对单次重复。",
              "JSON/CSV 保留完整精度；SR/SPL/OR 为比例，NE/TL 为米。", "",
              "## 历史比较限制", "",
              *[f"- {warning}" for warning in state["config"]["legacy_warnings"]],
              "- 本次为固定种子合成历史航迹，按原路线朝向构建；未实飞验证这些历史点。",
              "- 历史完整配置没有运行本次合成航迹估计器；此比较检验合成初始朝向扰动，不能直接认定独立测量航迹下模块因果效果。",
              "- 旧基线、历史航迹、论文及配置的哈希核对通过；论文文字和数值未调整。", ""]
    (out / "verification_report.md").write_text("\n".join(report), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    analyze_batch(parser.parse_args().batch)
