"""Verify scene landmarks in vehicle-local NED, independently of vehicle XYZ."""
from __future__ import annotations

import math


def compare_anchors(expected, actual, tolerance_m=.05):
    errors = {}
    for name, target in expected.items():
        measured = actual.get(name, ())
        if (len(target) != 3 or len(measured) != 3 or
                not all(math.isfinite(v) for v in [*target, *measured])):
            raise ValueError(f"Missing/non-finite scene anchor: {name}")
        errors[name] = math.dist(target, measured)
    if not errors:
        raise ValueError("No scene anchors supplied")
    return {"passed": max(errors.values()) <= tolerance_m,
            "max_anchor_error_m": max(errors.values()), "anchor_errors_m": errors,
            "tolerance_m": tolerance_m}


def sample(client, reference):
    paused = client.simIsPause()
    client.simPause(True)
    try:
        local = list(client.simGetVehiclePose().position)
        world = list(client.simGetObjectPose("Drone1").position)
        origin = [a - b for a, b in zip(world, local)]
        anchors = {name: [v - d for v, d in zip(client.simGetObjectPose(name).position, origin)]
                   for name in reference["anchors_local_ned"]}
    finally:
        client.simPause(paused)
    return {**compare_anchors(reference["anchors_local_ned"], anchors, reference["anchor_tolerance_m"]),
            "vehicle_local_position": local, "vehicle_global_ned_position": world,
            "global_ned_origin_of_vehicle_local_frame": origin,
            "anchors_local_ned": anchors}
