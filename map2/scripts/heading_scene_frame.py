"""Downtown scene check against the original September route survey."""
import json
import os
import math
from pathlib import Path

from map1.scripts.heading_scene_frame import compare_anchors
from map2.scripts import heading_ablation as h

MAP_PATH = Path(r"H:\airsimdowntown\airsimdowntown\Content\maps\airsim_downtown.umap")
REFERENCE = h.MAP / "heading_ablation/downtown_scene_frame.json"


def prepare_reference():
    if REFERENCE.exists():
        frozen = json.loads(REFERENCE.read_text(encoding="utf-8"))
        asset = os.environ.get("DOWNTOWN_MAP", frozen["map_path"])
        if h.sha256(asset) != frozen["map_sha256"]:
            raise ValueError("Downtown map differs from the frozen experiment asset")
        return REFERENCE
    source = json.loads(h.FROZEN_ROUTES.read_text(encoding="utf-8"))
    # September survey used simGetObjectPose, whose frame is global NED.
    record = {"map_path": str(MAP_PATH), "map_sha256": h.sha256(MAP_PATH),
              "provenance": "Original formal routes and September simGetObjectPose scene survey; existing PlayerStart and frozen settings",
              "vehicle_spawn_settings": {}, "anchor_tolerance_m": .05,
              "anchors_global_ned": {r["actor"]: r["target_position"] for r in source.values()},
              "expected_vehicle_local_origin_global_ned": [0., 0., .14649658203125],
              "origin_provenance": "Original PlayerStart at global 0 and upright SimpleFlight collision half-height; AirSim NedTransform.cpp ground-offset convention",
              "settings_source": "map2/formal_20260909_170427/settings.json"}
    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    if REFERENCE.exists():
        if json.loads(REFERENCE.read_text(encoding="utf-8")) != record:
            raise ValueError("Existing Downtown reference changed; inspect before creating another batch")
    else:
        h.atomic_json(REFERENCE, record)
    return REFERENCE


def sample(client, reference):
    paused = client.simIsPause()
    client.simPause(True)
    try:
        local = list(client.simGetVehiclePose().position)
        world = list(client.simGetObjectPose("Drone1").position)
        origin = [a - b for a, b in zip(world, local)]
        measured = {name: list(client.simGetObjectPose(name).position)
                    for name in reference["anchors_global_ned"]}
    finally:
        client.simPause(paused)
    check = compare_anchors(reference["anchors_global_ned"], measured, reference["anchor_tolerance_m"])
    error = math.dist(origin, reference["expected_vehicle_local_origin_global_ned"])
    return {**check, "passed": check["passed"] and math.isfinite(error) and error <= .05,
            "max_anchor_error_m": max(check["max_anchor_error_m"], error),
            "global_landmark_max_error_m": check["max_anchor_error_m"], "local_origin_error_m": error,
            "vehicle_local_position": local, "vehicle_global_ned_position": world,
            "global_ned_origin_of_vehicle_local_frame": origin, "anchors_global_ned": measured}
