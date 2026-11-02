"""Map2 synthetic-history protocol. No simulator or network imports."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from map1.scripts.heading_ablation import (WEIGHTS, MODEL, ENDPOINT, NORMAL_ENDS, wrap, sha256,
    atomic_json, write_csv, read_history, heading_definition, heading_check, initial_turn_duration,
    new_collision, smoke_acceptance, metrics)
from map1.scripts.validate_map1 import flights, points

ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT / "map2"
HISTORY = MAP / "history_tracks"
FROZEN_ROUTES = MAP / "formal_20260909_170427/routes.json"
SEED = 20260823
SMOKE_ROUTES = (1, 10, 30)


def synthetic_points(route, amplitude):
    """Oldest H0 to newest H5, with H5 three metres before navigation start.

    The weighted vector equals three metres along the frozen baseline heading.
    No trajectory outcome or goal coordinate is used to select the perturbation.
    These points are mathematical inputs, not physically flown GNSS observations.
    """
    yaw = math.radians(route["heading_deg"])
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.])
    side = np.array([-forward[1], forward[0], 0.])
    shape = np.array([2., 1., 0., -1., -2.])
    shape -= np.dot(WEIGHTS, shape)
    vectors = 3 * forward + (shape * amplitude)[:, None] * side
    newest = np.array([*route["start_position"][:2], -route["altitude"]]) - 3 * forward
    result = [newest]
    for vector in reversed(vectors):
        result.append(result[-1] - vector)
    return np.array(result[::-1]).tolist()


def prepare_histories():
    source = json.loads(FROZEN_ROUTES.read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    amplitudes = rng.uniform(.5, 3., 30) * rng.permutation(np.array([1.] * 15 + [-1.] * 15))
    HISTORY.mkdir(parents=True, exist_ok=True)
    metadata = {"synthetic": True, "seed": SEED, "source_routes_sha256": sha256(FROZEN_ROUTES),
                "protocol": "S-curve: 3m longitudinal segments; [2,1,0,-1,-2] lateral shape centered by weights; abs amplitude U(0.5,3)m; balanced seeded signs",
                "time_order": "indices 0..5 = History_-6..History_-1, oldest to newest; ordinal labels, not measured timestamps",
                "last_segment": "H5-H4, newest two historical points",
                "limitations": ["Synthetic histories constructed to match frozen route heading; not independent measured GNSS tracks",
                                "Historical complete configuration did not read these histories; comparison is a synthetic initial-heading perturbation against historical control",
                                "History points were not physically flown or collision-validated; they are estimator inputs only"],
                "routes": {}}
    for number in range(1, 31):
        key = f"{number:02d}"
        route = source[key]
        path = HISTORY / f"data_{number}_history.csv"
        values = synthetic_points(route, float(amplitudes[number - 1]))
        if path.exists():
            if not np.allclose(read_history(path), values, atol=1e-12, rtol=0):
                raise ValueError(f"Existing synthetic history differs: {path}")
        else:
            write_csv(path, [{"Point_Index": i, "X": p[0], "Y": p[1], "Z": p[2],
                             "Description": f"History_{i-6}", "Synthetic": True} for i, p in enumerate(values)])
        metadata["routes"][key] = {"amplitude_m": float(amplitudes[number - 1]), "sha256": sha256(path)}
    dest = HISTORY / "provenance.json"
    if dest.exists() and json.loads(dest.read_text(encoding="utf-8")) != metadata:
        raise ValueError("Existing synthetic provenance differs")
    if not dest.exists():
        atomic_json(dest, metadata)
    return metadata


def baseline_folder(number):
    matches = list((MAP / f"data_{number}").glob("qwen3.5-35b-a3b_CLIP_*"))
    if len(matches) != 1:
        raise ValueError(f"Expected one existing complete Qwen+CLIP group: route {number}")
    return matches[0]


def routes():
    provenance = prepare_histories()
    source = json.loads(FROZEN_ROUTES.read_text(encoding="utf-8"))
    goals = json.loads((MAP / "goaldest/goals.json").read_text(encoding="utf-8"))
    result = {}
    for number in range(1, 31):
        key = f"{number:02d}"
        route = source[key]
        turn = route["turn_degrees"] * (-1 if route["turn_direction"] == "turn_left" else 1)
        if abs(wrap(route["initial_yaw_deg"] + turn - route["heading_deg"])) > 1e-7:
            raise ValueError(f"Frozen initial heading disagrees: {key}")
        path = HISTORY / f"data_{number}_history.csv"
        definition = heading_definition(read_history(path), turn)
        if abs(wrap(definition["weighted_yaw_deg"] - route["heading_deg"])) > 1e-8:
            raise ValueError(f"Weighted synthetic heading disagrees: {key}")
        result[str(number)] = {**route, "route": number, "display_name": route["name"],
            "name": baseline_folder(number).name.removeprefix("qwen3.5-35b-a3b_CLIP_"),
            "goal_position": goals[key], "success_threshold": 6.,
            "history_path": str(path.relative_to(ROOT)), "history_sha256": sha256(path),
            "synthetic_history": True, "synthetic_amplitude_m": provenance["routes"][key]["amplitude_m"],
            **definition}
    return result


def jobs():
    return ([{"id": f"smoke_{r:02d}_01", "kind": "smoke", "route": r, "repeat": 1,
              "status": "pending", "attempts": 0} for r in SMOKE_ROUTES] +
            [{"id": f"formal_{r:02d}_{k:02d}", "kind": "formal", "route": r, "repeat": k,
              "status": "pending", "attempts": 0} for r in range(1, 31) for k in range(1, 11)])


def validate_recent_history(route):
    """Independent newest-point audit, also used by offline result verification."""
    values = read_history(ROOT / route["history_path"])
    newest = [values[5][k] - values[4][k] for k in (0, 1)]
    yaw = math.degrees(math.atan2(newest[1], newest[0]))
    if abs(wrap(yaw - route["last_segment_yaw_deg"])) > 1e-9:
        raise ValueError("Latest two history points disagree with recorded heading")
    return {"indices": [4, 5], "direction": "H5-H4", "vector_xy": newest, "yaw_deg": yaw}
