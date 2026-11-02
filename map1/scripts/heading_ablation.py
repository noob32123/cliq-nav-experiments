"""Frozen heading-ablation definitions; no simulator or network dependencies."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

try:
    from . import evaluate_nav_metrics as metrics
except ImportError:
    import evaluate_nav_metrics as metrics

ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT / "map1"
WEIGHTS = (0.1, 0.15, 0.2, 0.25, 0.3)
BASE_TURNS = (160, 0, 70, 0, -90, -90, 25, 90, -120, 170, -40, -30, 60, -90)
ALTITUDES = (1, 5, 8, 8, 12, 8, 3, 12, 1, 18, 3, 3, 8, 3)
CLIP_TEXTS = ("orange and white striped umbrella", "statue", "restaurant", "hotel front door",
              "blue amusement park", "fountain", "house", "playground", "park bench",
              "umbrella", "house", "fountain", "sculpture", "sculpture")
TARGETS = ("前方橘白色条纹遮阳伞的附近", "前方人物雕塑的附近", "河对岸的餐厅门口附近",
           "前方酒店正门门口", "前方蓝色游乐设施的附近", "前方喷泉的附近",
           "前方房屋门口的附近", "前方游乐设施的附近", "前方公园长椅的附近",
           "前方蓝色建筑物右边遮阳伞区域的附近", "前方房屋门口的附近", "前方喷泉的附近",
           "前方艺术雕塑的附近", "前方艺术雕塑的附近")
MODEL = "qwen3.5-35b-a3b"
ENDPOINT = "https://api.desire-drive.com/v1"
SMOKE_ROUTES = (1, 3, 10)
NORMAL_ENDS = {"model_stop", "collision", "max_iterations", "smoke_step_limit"}


def wrap(angle: float) -> float:
    if not math.isfinite(angle):
        raise ValueError("Non-finite heading")
    return (angle + 180.0) % 360.0 - 180.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for attempt in range(21):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 20:
                raise
            time.sleep(0.1)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_history(path: Path) -> list[tuple[float, float, float]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"Point_Index", "X", "Y", "Z"}.issubset(reader.fieldnames or ()):
            raise ValueError(f"Invalid history header: {path}")
        try:
            rows = [(int(row["Point_Index"]), tuple(float(row[k]) for k in ("X", "Y", "Z")))
                    for row in reader]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid history coordinate/index: {path}") from exc
    rows.sort(key=lambda row: row[0])
    if [row[0] for row in rows] != list(range(6)):
        raise ValueError(f"Expected exactly six unique history indices 0..5: {path}")
    if not all(math.isfinite(x) for _, point in rows for x in point):
        raise ValueError(f"Non-finite history coordinate: {path}")
    return [row[1] for row in rows]


def heading_definition(points, baseline_turn: float) -> dict:
    if len(points) != 6 or not all(len(p) == 3 and all(math.isfinite(x) for x in p) for p in points):
        raise ValueError("Expected six finite XYZ history points")
    segments = [[points[i + 1][k] - points[i][k] for k in range(2)] for i in range(5)]
    weighted = [sum(w * v[k] for w, v in zip(WEIGHTS, segments)) for k in range(2)]
    last = segments[-1]
    if min(math.hypot(*weighted), math.hypot(*last)) < 1e-5:
        raise ValueError("Zero-length weighted or last-segment XY vector")
    full_yaw = math.degrees(math.atan2(weighted[1], weighted[0]))
    last_yaw = math.degrees(math.atan2(last[1], last[0]))
    delta = wrap(last_yaw - full_yaw)
    cross = weighted[0] * last[1] - weighted[1] * last[0]
    dot = weighted[0] * last[0] + weighted[1] * last[1]
    independent_delta = wrap(math.degrees(math.atan2(cross, dot)))
    if abs(wrap(delta - independent_delta)) > 1e-8:
        raise ValueError("Independent signed angle calculations disagree")
    turn = baseline_turn + delta
    return {"weights": list(WEIGHTS), "history_points": points, "segments_xy": segments,
            "weighted_vector_xy": weighted, "last_vector_xy": last,
            "weighted_yaw_deg": full_yaw, "last_segment_yaw_deg": last_yaw,
            "baseline_turn_deg": float(baseline_turn), "delta_deg": delta,
            "cross_dot_delta_deg": independent_delta, "ablation_turn_deg": turn,
            "ablation_action": "turn_right" if turn > 0 else "turn_left" if turn < 0 else "none",
            "ablation_action_value_deg": abs(turn)}


def routes() -> dict[str, dict]:
    result = {}
    for n in range(1, 15):
        key = f"data_{n}"
        history = MAP / "history_tracks" / f"{key}_history.csv"
        definition = heading_definition(read_history(history), BASE_TURNS[n - 1])
        with history.open(encoding="utf-8-sig", newline="") as stream:
            stored = float(next(csv.DictReader(stream))["Turn_Angle_Deg"])
        if not math.isfinite(stored) or abs(stored - abs(definition["delta_deg"])) > 0.000051:
            raise ValueError(f"Stored unsigned history angle disagrees: {key}")
        result[str(n)] = {"route": n, **metrics.SCENARIO_CONFIG[key],
                          "altitude": ALTITUDES[n - 1], "target": TARGETS[n - 1],
                          "clip_text": CLIP_TEXTS[n - 1], "history_path": str(history.relative_to(ROOT)),
                          "history_sha256": sha256(history), "stored_unsigned_delta_deg": stored,
                          **definition}
    return result


def heading_check(route: dict, baseline: dict, ablation: dict) -> dict:
    delta = route["delta_deg"]
    actual = wrap(ablation["end"]["yaw_deg"] - baseline["end"]["yaw_deg"])
    initial_error = abs(wrap(ablation["start"]["yaw_deg"] - baseline["start"]["yaw_deg"]))
    error = abs(wrap(actual - delta))
    command_error = abs((route["ablation_turn_deg"] - route["baseline_turn_deg"]) - delta)
    return {"route": route["route"], "baseline_turn_deg": route["baseline_turn_deg"],
            "weighted_yaw_deg": route["weighted_yaw_deg"], "last_segment_yaw_deg": route["last_segment_yaw_deg"],
            "delta_deg": delta, "ablation_turn_deg": route["ablation_turn_deg"],
            "baseline_start_yaw_deg": baseline["start"]["yaw_deg"],
            "baseline_end_yaw_deg": baseline["end"]["yaw_deg"],
            "ablation_start_yaw_deg": ablation["start"]["yaw_deg"],
            "ablation_end_yaw_deg": ablation["end"]["yaw_deg"],
            "actual_delta_deg": actual, "actual_delta_error_deg": error,
            "initial_yaw_error_deg": initial_error, "command_delta_error_deg": command_error,
            "passed": initial_error <= 0.5 and error <= 1.0 and command_error <= 1e-8}


def jobs() -> list[dict]:
    smoke = [{"id": f"smoke_{r:02d}_01", "kind": "smoke", "route": r, "repeat": 1,
              "status": "pending", "attempts": 0} for r in SMOKE_ROUTES]
    return smoke + [{"id": f"formal_{r:02d}_{k:02d}", "kind": "formal", "route": r, "repeat": k,
                     "status": "pending", "attempts": 0} for r in range(1, 15) for k in range(1, 11)]


def done_ids(state: dict) -> set[str]:
    return {j["id"] for j in state["jobs"] if j["status"] == "complete"}


def initial_turn_duration(angle: float, rate: float = 30.0) -> float:
    if not math.isfinite(angle) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("Invalid initialization angle/rate")
    return abs(angle) / rate  # Deliberately no navigation minimum of 0.5 s.


def new_collision(record: dict, navigation_epoch_ns: int) -> bool:
    """Ignore saved takeoff contacts, but retain every new navigation contact."""
    return bool(record["has_collided"] and int(record["time_stamp_ns"]) > navigation_epoch_ns)


def smoke_acceptance(result: dict) -> dict:
    motion = result.get("max_action_position_change_m", 0) >= 0.05 or result.get("max_action_yaw_change_deg", 0) >= 0.2
    checks = {"heading_difference": result["heading_check"]["passed"],
              "first_frame_vehicle": result["first_frame_yaw_error_deg"] <= 1,
              "first_frame_camera": result["first_frame_camera_yaw_error_deg"] <= 1,
              "real_model_response": result["model"] == MODEL and result["decisions"] >= 1,
              "action_executed": result.get("actions_executed", 0) >= 1,
              "measurable_pose_response": motion}
    return {"passed": all(checks.values()), "checks": checks,
            "actions_executed": result.get("actions_executed", 0),
            "max_action_position_change_m": result.get("max_action_position_change_m", 0),
            "max_action_yaw_change_deg": result.get("max_action_yaw_change_deg", 0)}
