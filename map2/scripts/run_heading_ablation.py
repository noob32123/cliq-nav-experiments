"""Resumable map2 synthetic-history heading ablation with per-flight measured heading gates.

python map2/scripts/run_heading_ablation.py --preflight-only --batch-dir PATH
python map2/scripts/run_heading_ablation.py --resume PATH --smoke-test
python map2/scripts/run_heading_ablation.py --resume PATH --formal-only
"""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
from datetime import datetime, timezone, timedelta
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from map2.scripts import heading_ablation as h
from map1.scripts import validate_map1 as validation
from map2.scripts import heading_scene_frame as scene_frame

ENGINE = Path(os.environ.get("UE4_EDITOR", r"H:\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe"))
PROJECT = Path(os.environ.get("DOWNTOWN_PROJECT", r"H:\airsimdowntown\airsimdowntown\airsimdowntown.uproject"))
GLOBAL_SETTINGS = Path.home() / "Documents/AirSim/settings.json"
CODE = ("map2/scripts/heading_ablation.py", "map2/scripts/run_heading_ablation.py",
        "map2/scripts/analyze_heading_ablation.py", "map2/scripts/heading_scene_frame.py",
        "map2/scripts/monitor_heading_ablation.py", "map2/scripts/validate_map2.py",
        "map1/scripts/heading_ablation.py", "map1/scripts/heading_scene_frame.py", "map1/scripts/evaluate_nav_metrics.py",
        "map1/scripts/validate_map1.py", "nav_qwen_zgc.py", "fly_on_keyboard.py", "setup_path.py")
RUNNER_SOURCE = "map2/scripts/run_heading_ablation.py"
SCHEDULER_VERSION = 2
RECOVERY_COOLDOWN_SECONDS = 30


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def port_open(port):
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


class PauseBatch(RuntimeError):
    pass


class HeadingGateError(PauseBatch):
    pass


class SceneFrameError(PauseBatch):
    """A batch-wide coordinate error cannot be handled by retrying other routes."""
    pass


class Batch:
    def __init__(self, args):
        self.args = args
        self.directory = (args.resume or args.batch_dir or
                          h.MAP / "heading_ablation" / datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")).resolve()
        self.engine = self.drone = self.ai = None
        self.context = {}
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.directory / "state.json"
        if args.resume:
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state.get("exclude_from_statistics"):
                raise ValueError("Archived invalid workflow batch cannot resume; use a corrected separate batch")
            self.routes = json.loads((self.directory / "routes.json").read_text(encoding="utf-8"))
            if getattr(args, "upgrade_scheduler", False):
                self.upgrade_scheduler()
            self.verify_manifest()
            # A crash after writing result.json but before checkpoint publication is recoverable.
            for job in self.state["jobs"]:
                if job["status"] != "complete":
                    for path in sorted((self.directory / "attempts" / job["id"]).glob("*/result.json")):
                        record = json.loads(path.read_text(encoding="utf-8"))
                        trajectory = path.parent / "trajectory.csv"
                        if (record.get("complete") and record["id"] == job["id"] and
                                trajectory.is_file() and h.sha256(trajectory) == record["trajectory_sha256"]):
                            job.update(status="complete", result_path=str(path.relative_to(self.directory)))
                    if job["status"] != "complete":
                        # Tried but unfinished jobs belong at the back, including a
                        # protection failure saved by the older stop-on-error runner.
                        job["status"] = "deferred" if job.get("attempts", 0) else "pending"
                        if job["status"] == "deferred":
                            old_attempt = self.directory / "attempts" / job["id"] / f"{job['attempts']:03d}"
                            failure_path = old_attempt / "failure.json"
                            failure = json.loads(failure_path.read_text(encoding="utf-8")) if failure_path.is_file() else {}
                            job.setdefault("deferrals", 1)
                            job.setdefault("last_failed_attempt", str(old_attempt.relative_to(self.directory)))
                            job.setdefault("last_error_type", failure.get("error_type", "InterruptedAttempt"))
                            job.setdefault("last_error_message", failure.get("message", "Previous attempt did not finish"))
                if job["status"] == "complete":
                    path = self.directory / job["result_path"]
                    record = json.loads(path.read_text(encoding="utf-8"))
                    if (not record["complete"] or not record["heading_check"]["passed"] or
                            (job["kind"] == "smoke" and not h.smoke_acceptance(record)["passed"]) or
                            h.sha256(path.parent / "trajectory.csv") != record["trajectory_sha256"]):
                        raise ValueError(f"Invalid completed result: {job['id']}")
            frozen = self.state["config"]
            if (frozen["endpoint"], frozen["rpc_port"]) != (args.endpoint, args.rpc_port):
                raise ValueError("Resume endpoint/port must match frozen batch configuration")
        else:
            if self.state_path.exists():
                raise ValueError("Batch exists; use --resume")
            self.routes = h.routes()
            frame_path = args.scene_frame_reference.resolve()
            frame = json.loads(frame_path.read_text(encoding="utf-8"))
            config = {"model": h.MODEL, "endpoint": args.endpoint, "rpc_port": args.rpc_port,
                      "scene_frame_reference": str(frame_path.relative_to(ROOT)),
                      "scene_frame_reference_sha256": h.sha256(frame_path),
                      "vehicle_spawn_settings": frame["vehicle_spawn_settings"],
                      "clock_speed": 5, "max_iterations": 30, "smoke_iterations": 2,
                      "horizontal_speed": 2, "vertical_speed": 1.5, "yaw_rate": 30,
                      "action_min_seconds": 0.5, "post_action_wall_seconds": 0.8, "pause_during_inference": True,
                      "max_tokens": 256, "enable_thinking": False, "request_timeout_seconds": 150,
                      "request_attempts": 3, "seed": 20260823, "bootstrap_resamples": 10000,
                      "engine": str(args.engine), "project": str(args.project),
                      "heading_policy": "original signed initialization turn + wrap(last XY yaw - weighted XY yaw)",
                      "collision_policy": "record and consume initialization contacts; count only navigation contacts newer than takeover timestamp",
                      "smoke_acceptance": "heading/image/CLIP/real response gates plus >=1 executed action and >=0.05m displacement or >=0.2deg yaw response",
                      "reset_initial_yaw_deg": -45,
                      "initial_orientation_note": "upright -45deg yaw from original map2 route; measured in both calibration arms",
                      "baseline_yaw_limit": "historical full configuration reused; new calibration compares initialization commands",
                      "history_provenance": "history_tracks; synthetic deterministic S-curves aligned with original route headings; newest segment H5-H4",
                      "legacy_warnings": ["Synthetic histories, not measured GNSS. Historical full group did not execute the estimator on these histories",
                                          "Interpret as a synthetic initial-heading perturbation against a historical Qwen+CLIP control",
                                          "Same endpoint/model parameters requested; remote model implementation and temporal run conditions may differ",
                                          "Original baseline endpoint excludes landing via its saved navigation_end_position" ]}
            self.state = {"batch_id": self.directory.name, "created_at": now(), "status": "created",
                          "config": config, "jobs": h.jobs()}
            h.atomic_json(self.directory / "routes.json", self.routes)
            angle_rows = [{k: route[k] for k in ("route", "name", "weighted_yaw_deg", "last_segment_yaw_deg",
                                                "baseline_turn_deg", "delta_deg", "ablation_turn_deg",
                                                "ablation_action", "ablation_action_value_deg", "history_sha256")}
                          for route in self.routes.values()]
            h.write_csv(self.directory / "heading_angles.csv", angle_rows)
            self.freeze_manifest()
            self.save()
        self.write_resume_script()

    def freeze_manifest(self):
        baseline_paths = []
        for n in range(1, 31):
            route = self.routes[str(n)]
            folder = h.MAP / f"data_{n}" / f"qwen3.5-35b-a3b_CLIP_{route['name']}"
            paths = validation.flights(folder)
            for path in paths:
                validation.points(path)
            baseline_paths.extend(paths)
        protected = baseline_paths + list(h.HISTORY.rglob("*")) + list((ROOT / "paper").rglob("*"))
        protected += [h.MAP / "goaldest/goals.json", h.FROZEN_ROUTES, h.MAP / "formal_20260909_170427/settings.json", h.MAP / "downtown_routes/object_poses.json"]
        protected += [p.with_suffix(".json") for p in baseline_paths]
        protected += [ROOT / self.state["config"]["scene_frame_reference"]]
        hashes = {str(path.relative_to(ROOT)): h.sha256(path) for path in protected if path.is_file()}
        hashes.update({name: h.sha256(ROOT / name) for name in CODE})
        snap = self.directory / "source_snapshot"
        for name in CODE:
            target = snap / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        self.manifest = {"sha256": hashes, "baseline_csv_count": len(baseline_paths),
                         "routes_sha256": h.sha256(self.directory / "routes.json"),
                         "heading_angles_sha256": h.sha256(self.directory / "heading_angles.csv"),
                         "config_sha256": hashlib.sha256(json.dumps(self.state["config"], sort_keys=True).encode()).hexdigest(),
                         "global_settings_path": str(GLOBAL_SETTINGS), "global_settings_sha256": h.sha256(GLOBAL_SETTINGS)}
        frame = json.loads((ROOT / self.state["config"]["scene_frame_reference"]).read_text(encoding="utf-8"))
        if h.sha256(Path(os.environ.get("DOWNTOWN_MAP", frame["map_path"]))) != frame["map_sha256"]:
            raise ValueError("Downtown map changed after default-origin reference was recorded")
        self.manifest["external_sha256"] = {os.environ.get("DOWNTOWN_MAP", frame["map_path"]): frame["map_sha256"]}
        h.atomic_json(self.directory / "input_manifest.json", self.manifest)

    def verify_manifest(self, allow_scheduler_upgrade=False):
        manifest = json.loads((self.directory / "input_manifest.json").read_text(encoding="utf-8"))
        changed = [name for name, digest in manifest["sha256"].items()
                   if (not (ROOT / name).is_file() or h.sha256(ROOT / name) != digest)
                   and not (allow_scheduler_upgrade and name == RUNNER_SOURCE)]
        changed.extend(name for name, digest in manifest.get("external_sha256", {}).items()
                       if not Path(name).is_file() or h.sha256(Path(name)) != digest)
        if h.sha256(GLOBAL_SETTINGS) != manifest["global_settings_sha256"]:
            changed.append("global AirSim settings")
        for name, key in (("routes.json", "routes_sha256"), ("heading_angles.csv", "heading_angles_sha256")):
            if h.sha256(self.directory / name) != manifest[key]:
                changed.append(name)
        if hashlib.sha256(json.dumps(self.state["config"], sort_keys=True).encode()).hexdigest() != manifest["config_sha256"]:
            changed.append("frozen configuration")
        if changed:
            raise ValueError(f"Frozen inputs/code changed: {changed[:10]}")
        self.manifest = manifest

    def upgrade_scheduler(self):
        raise ValueError("Map2 source changes require an explicitly reviewed new batch; scheduler upgrade is disabled")

    def save(self):
        self.state["updated_at"] = now()
        self.state["smoke_completed"] = sum(j["kind"] == "smoke" and j["status"] == "complete" for j in self.state["jobs"])
        self.state["formal_completed"] = sum(j["kind"] == "formal" and j["status"] == "complete" for j in self.state["jobs"])
        deferred = [j for j in self.state["jobs"] if j["status"] == "deferred"]
        self.state["formal_deferred"] = sum(j["kind"] == "formal" for j in deferred)
        self.state["deferred_jobs"] = [j["id"] for j in deferred]
        h.atomic_json(self.state_path, self.state)
        h.atomic_json(self.directory / "retry_queue.json", {"updated_at": self.state["updated_at"],
                      "scheduler_phase": self.state.get("scheduler_phase"),
                      "recovery_round": self.state.get("recovery_round", 0),
                      "jobs": [{k: j.get(k) for k in ("id", "kind", "route", "repeat", "attempts", "deferrals", "last_error_type", "last_error_message", "last_failed_attempt")}
                               for j in deferred]})

    def event(self, event, **fields):
        record = {"time": now(), "event": event, **self.context, **fields}
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    def write_resume_script(self):
        # No credentials written to disk; background process inherits the authorized environment.
        quote = lambda s: "'" + str(s).replace("'", "''") + "'"
        script = ("$ErrorActionPreference = 'Stop'\n"
                  f"$batchPath = {quote(self.directory)}\n"
                  f"$arguments = @('-B', '-u', '\"{ROOT / CODE[1]}\"', '--resume', '\"{self.directory}\"', '--formal-only')\n"
                  f"$p = Start-Process -FilePath {quote(sys.executable)} -ArgumentList $arguments -WorkingDirectory {quote(ROOT)} "
                  "-WindowStyle Hidden -RedirectStandardOutput (Join-Path $batchPath 'formal.stdout.log') "
                  "-RedirectStandardError (Join-Path $batchPath 'formal.stderr.log') -PassThru\n"
                  "$p.Id | Set-Content (Join-Path $batchPath 'background.pid')\nWrite-Output $p.Id\n")
        (self.directory / "start_formal.ps1").write_text(script, encoding="utf-8-sig")

    def load_runtime(self):
        import nav_qwen_zgc as nav
        import cv2
        import numpy as np
        from openai import OpenAI
        self.nav, self.cv2, self.np = nav, cv2, np
        self.airsim = nav.airsim
        if not os.environ.get("OPENAI_API_KEY"):
            raise PauseBatch("OPENAI_API_KEY is absent")
        self.ai = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=self.args.endpoint,
                         timeout=150, max_retries=0)

    def start_engine(self):
        if port_open(self.args.rpc_port):
            raise PauseBatch(f"RPC {self.args.rpc_port} occupied; existing simulation preserved")
        cfg = json.loads((h.MAP / "formal_20260909_170427/settings.json").read_text(encoding="utf-8-sig"))
        cfg.update(ClockSpeed=5.0, ApiServerPort=self.args.rpc_port)
        cfg["Vehicles"]["Drone1"].update(self.state["config"]["vehicle_spawn_settings"])
        # Preserve September camera resolutions/FOVs; apply the production camera poses.
        cameras = cfg["Vehicles"]["Drone1"]["Cameras"]
        cameras["bottom_center"].update(X=0, Y=0, Z=.7, Pitch=-90, Roll=0, Yaw=0)
        settings = self.directory / "settings.json"
        h.atomic_json(settings, cfg)
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
        command = [self.state["config"]["engine"], self.state["config"]["project"],
                   "/Game/maps/airsim_downtown", "-game", "-windowed",
                   "-ResX=1280", "-ResY=720", "-nosound", "-unattended",
                   f"-settings={settings}", f"-abslog={self.directory / 'unreal.log'}"]
        self.engine = subprocess.Popen(command, startupinfo=startup,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.state["engine_pid"] = self.engine.pid
        self.save()
        self.event("engine_started", pid=self.engine.pid, command=command)
        deadline = time.monotonic() + 3600  # First launch also builds large merged Downtown meshes.
        while not port_open(self.args.rpc_port):
            if self.engine.poll() is not None or time.monotonic() > deadline:
                raise PauseBatch("Downtown startup failed; see unreal.log")
            time.sleep(1)
        client = self.airsim.MultirotorClient(port=self.args.rpc_port, timeout_value=60)
        self.drone = self.nav.FixedDroneController(client=client)
        settings_read = json.loads(client.getSettingsString())
        if settings_read["ClockSpeed"] != 5 or settings_read["ApiServerPort"] != self.args.rpc_port:
            raise PauseBatch("Simulator settings do not match batch")
        frame_check = self.check_scene_frame()
        h.atomic_json(self.directory / "scene_frame_check.json", {"time": now(), **frame_check})
        t0 = client.getMultirotorState().timestamp
        start = time.monotonic()
        time.sleep(1)
        measured = (client.getMultirotorState().timestamp - t0) / 1e9 / (time.monotonic() - start)
        self.event("clock_verified", configured=5, measured=measured)

    def snapshot(self):
        pose = self.drone.client.simGetVehiclePose()
        q = pose.orientation
        values = [q.x_val, q.y_val, q.z_val, q.w_val]
        position = list(pose.position)
        if not all(math.isfinite(x) for x in values + position):
            raise HeadingGateError("Non-finite vehicle pose")
        yaw = math.degrees(self.airsim.to_eularian_angles(q)[2])
        return {"wall_time": now(), "yaw_deg": yaw, "orientation_xyzw": values, "position": position}

    def collision(self):
        info = self.drone.client.simGetCollisionInfo()
        return {"has_collided": bool(info.has_collided), "time_stamp_ns": int(info.time_stamp),
                "object_name": info.object_name, "object_id": int(info.object_id),
                "position": list(info.position), "impact_point": list(info.impact_point),
                "normal": list(info.normal), "penetration_depth": float(info.penetration_depth)}

    def check_scene_frame(self):
        path = ROOT / self.state["config"]["scene_frame_reference"]
        reference = json.loads(path.read_text(encoding="utf-8"))
        check = scene_frame.sample(self.drone.client, reference)
        if not check["passed"]:
            h.atomic_json(self.directory / "scene_frame_failure.json", {"time": now(), **check})
            raise SceneFrameError("Scene coordinate frame disagrees with original Downtown survey and PlayerStart")
        return check

    def initialize(self, route, diagnostic_path):
        c = self.drone.client
        c.simPause(False)
        c.reset()
        c.enableApiControl(True)
        c.armDisarm(True)
        c.simSetWind(self.airsim.Vector3r(0, 0, 0))
        self.drone.is_flying = False
        # AirSim sets the physics body immediately but queues its render pose.
        # Freeze physics and read ground-truth kinematics before controller
        # settling can change height/pitch; retain the separate render read too.
        c.simPause(True)
        try:
            c.simSetVehiclePose(self.airsim.Pose(self.airsim.Vector3r(*route["start_position"]),
                                               self.airsim.to_quaternion(0, 0, math.radians(route["initial_yaw_deg"]))), True)
            reset_return = self.snapshot()
            kinematics = c.simGetGroundTruthKinematics()
            q = kinematics.orientation
            pitch, roll, yaw = self.airsim.to_eularian_angles(q)
            reset = {"wall_time": now(), "source": "simGetGroundTruthKinematics while paused",
                     "position": list(kinematics.position), "yaw_deg": math.degrees(yaw),
                     "orientation_xyzw": [q.x_val, q.y_val, q.z_val, q.w_val]}
            confirmed = (all(math.isfinite(v) for v in reset["position"] + reset["orientation_xyzw"]) and
                         math.dist(reset["position"], route["start_position"]) <= .1 and
                         abs(h.wrap(reset["yaw_deg"] - route["initial_yaw_deg"])) <= .5 and
                         max(abs(pitch), abs(roll)) <= math.radians(.5))
        finally:
            c.simPause(False)
        reset_samples = [reset]
        if not confirmed:
            h.atomic_json(diagnostic_path, {"reset_command_position": route["start_position"],
                "reset_rpc_return_pose": reset_return, "reset_application_samples": reset_samples,
                "passed": False, "failure": "Physics reset was not confirmed"})
            raise HeadingGateError("Physics reset pose did not reach commanded XYZ and upright route initial yaw")
        h.atomic_json(diagnostic_path, {"reset_command_position": route["start_position"],
                      "reset_rpc_return_pose": reset_return, "reset_confirmed_pose": reset,
                      "reset_application_samples": reset_samples})
        c.moveByVelocityAsync(0, 0, 0, 2.0).join()
        c.simSetCameraPose("front_center", self.airsim.Pose(self.airsim.Vector3r(.5,0,0), self.airsim.to_quaternion(0,0,0)))
        c.simSetCameraPose("bottom_center", self.airsim.Pose(self.airsim.Vector3r(0,0,.7), self.airsim.to_quaternion(-math.pi/2,0,0)))
        self.drone.takeoff(route["altitude"], settle_seconds=.2)
        c.moveByVelocityZAsync(0, 0, -route["altitude"], 3).join()
        c.hoverAsync().join()
        time.sleep(0.3)
        takeoff = self.snapshot()
        check = self.check_scene_frame()
        xy_error = math.dist(takeoff["position"][:2], route["start_position"][:2])
        z_error = abs(takeoff["position"][2] + route["altitude"])
        passed = xy_error <= .5 and z_error <= .5
        h.atomic_json(diagnostic_path, {"reset_command_position": route["start_position"],
                      "reset_rpc_return_pose": reset_return, "reset_confirmed_pose": reset,
                      "reset_application_samples": reset_samples, "post_takeoff_pose": takeoff,
                      "scene_frame": check, "xy_error_m": xy_error, "altitude_error_m": z_error,
                      "passed": passed})
        if not passed:
            raise HeadingGateError("Post-takeoff start position/height disagrees with route configuration")
        return takeoff

    def precise_turn(self, angle, path):
        c = self.drone.client
        start = self.snapshot()
        target = h.wrap(start["yaw_deg"] + angle)
        record = {"requested_relative_turn_deg": angle, "target_relative_yaw_deg": target,
                  "start": start, "commands": [], "samples": []}
        h.atomic_json(path, record)
        duration = h.initial_turn_duration(angle)
        if duration:
            command = {"yaw_rate_deg_s": math.copysign(30, angle), "duration_sim_seconds": duration}
            record["commands"].append(command)
            h.atomic_json(path, record)
            c.moveByVelocityAsync(0, 0, 0, duration,
                                  yaw_mode=self.airsim.YawMode(True, command["yaw_rate_deg_s"])).join()
        record["open_loop_end"] = self.snapshot()
        # Fix rate-controller execution error only; target is relative initialization, not history absolute yaw.
        for correction in range(3):
            if abs(h.wrap(target - self.snapshot()["yaw_deg"])) <= 0.25:
                break
            record["commands"].append({"correction": correction + 1, "rotate_to_relative_target_deg": target})
            h.atomic_json(path, record)
            c.rotateToYawAsync(target, timeout_sec=30, margin=0.15).join()
        c.hoverAsync().join()
        deadline, consecutive = time.monotonic() + 10, 0
        while time.monotonic() < deadline:
            sample = self.snapshot()
            record["samples"].append(sample)
            consecutive = consecutive + 1 if abs(h.wrap(sample["yaw_deg"] - target)) <= 0.35 else 0
            if consecutive >= 5:
                record["end"] = sample
                record["execution_error_deg"] = abs(h.wrap(sample["yaw_deg"] - target))
                h.atomic_json(path, record)
                return record
            time.sleep(0.1)
        record["end"] = self.snapshot()
        record["failed"] = True
        h.atomic_json(path, record)
        raise HeadingGateError("Relative initialization rotation did not stabilize")

    def calibrate(self, route, folder):
        self.initialize(route, folder / "baseline_reset_takeoff.json")
        baseline = self.precise_turn(route["baseline_turn_deg"], folder / "baseline_initialization.json")
        self.initialize(route, folder / "ablation_reset_takeoff.json")
        ablation = self.precise_turn(route["ablation_turn_deg"], folder / "ablation_initialization.json")
        check = h.heading_check(route, baseline, ablation)
        h.atomic_json(folder / "heading_check.json", check)
        self.event("heading_checked", **check)
        if not check["passed"]:
            raise HeadingGateError("Measured initialization difference disagrees with theoretical delta")
        return check, ablation

    def capture(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        c = self.drone.client
        requests = [self.airsim.ImageRequest("front_center", self.airsim.ImageType.Scene, False, False),
                    self.airsim.ImageRequest("front_center", self.airsim.ImageType.DepthVis, False, False),
                    self.airsim.ImageRequest("bottom_center", self.airsim.ImageType.Scene, False, False)]
        for attempt in range(3):
            responses = c.simGetImages(requests)
            if len(responses) == 3 and all((r.width, r.height) == size and len(r.image_data_uint8) in
                                          (r.width * r.height, r.width * r.height * 3) for r, size in zip(responses, ((640,480),(640,480),(2560,1440)))):
                break
            time.sleep(1)
        else:
            raise PauseBatch("Three complete original-resolution observations unavailable")
        images, camera_records, urls = [], [], []
        for i, response in enumerate(responses):
            data = self.np.frombuffer(response.image_data_uint8, dtype=self.np.uint8)
            if data.size == response.width * response.height:
                image = self.cv2.cvtColor(data.reshape(response.height, response.width), self.cv2.COLOR_GRAY2RGB)
            else:
                image = data.reshape(response.height, response.width, 3)
            self.cv2.imwrite(str(folder / f"{i}.png"), image)
            url = self.nav.encode_numpy_to_data_url(image)
            (folder / f"{i}.jpeg").write_bytes(base64.b64decode(url.split(",", 1)[1]))
            images.append(image)
            urls.append({"type": "image_url", "image_url": {"url": url}})
            q = response.camera_orientation
            camera_records.append({"view": i, "camera": "front_center" if i < 2 else "bottom_center",
                                   "width": response.width, "height": response.height,
                                   "sim_timestamp_ns": int(response.time_stamp),
                                   "orientation_xyzw": [q.x_val, q.y_val, q.z_val, q.w_val],
                                   "yaw_deg": math.degrees(self.airsim.to_eularian_angles(q)[2]),
                                   "position": list(response.camera_position)})
        observation = {"vehicle": self.snapshot(), "cameras": camera_records}
        h.atomic_json(folder / "observation.json", observation)
        return urls, observation

    def prompt(self, route, clip_position):
        return self.nav.build_target_prompt(route["target"]) + f"为辅助你的判断：你的导航目标有很大的可能性在你的{clip_position}"

    def request(self, urls, prompt, folder):
        messages = [{"role": "user", "content": urls + [{"type": "text", "text": prompt}]}]
        h.atomic_json(folder / "request.json", {"model": h.MODEL, "endpoint": self.args.endpoint,
                                               "max_tokens": 256, "enable_thinking": False, "prompt": prompt,
                                               "images": [{"file": f"{i}.jpeg", "sha256": h.sha256(folder / f"{i}.jpeg")}
                                                          for i in range(3)]})
        for attempt in range(1, 4):
            start = time.monotonic()
            self.event("request_started", request_attempt=attempt)
            try:
                response = self.ai.chat.completions.create(model=h.MODEL, messages=messages, max_tokens=256,
                            extra_body={"chat_template_kwargs": {"enable_thinking": False}}, timeout=150)
                h.atomic_json(folder / f"response_{attempt}.json", response.model_dump(mode="json"))
                if response.model != h.MODEL:
                    raise PauseBatch("Response model does not equal requested qwen3.5-35b-a3b")
                text = response.choices[0].message.content or ""
                action, value = self.nav.parse_action(text)
                if action is None or not math.isfinite(value):
                    raise ValueError("Invalid model action format")
                self.event("request_completed", model=response.model, action=action, value=value,
                           elapsed_seconds=time.monotonic() - start)
                return action, value, text
            except PauseBatch:
                raise
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                error = {"error_type": type(exc).__name__, "http_status": status,
                         "elapsed_seconds": time.monotonic() - start, "request_attempt": attempt}
                h.atomic_json(folder / f"request_error_{attempt}.json", error)
                self.event("request_failed", **error)
                if status in (400, 401, 402, 403, 404, 422) or attempt == 3:
                    raise PauseBatch("Model request failed; preserved attempt and paused batch") from None
                time.sleep(5)

    def flight(self, job, folder):
        route = self.routes[str(job["route"])]
        h.atomic_json(folder / "route_definition.json", route)
        check, ablation = self.calibrate(route, folder)
        c = self.drone.client
        limit = 2 if job["kind"] == "smoke" else self.state["config"]["max_iterations"]
        result = {"id": job["id"], "kind": job["kind"], "route": job["route"], "repeat": job["repeat"],
                  "complete": False, "heading_check": check, "started_at": now(), "model": h.MODEL,
                  "reason": "smoke_step_limit" if job["kind"] == "smoke" else "max_iterations", "decisions": 0,
                  "actions_executed": 0, "max_action_position_change_m": 0.0, "max_action_yaw_change_deg": 0.0}
        # This AirSim RPC returns AND clears accumulated contacts. Takeoff contacts are
        # separate from visual navigation; never let them abort the first model action.
        initialization_collision = self.collision()
        collision_epoch = max(int(c.getMultirotorState().timestamp), initialization_collision["time_stamp_ns"])
        h.atomic_json(folder / "navigation_takeover.json", {"epoch_sim_ns": collision_epoch,
                      "initialization_collision": initialization_collision, "pose": self.snapshot()})
        trajectory = folder / "trajectory.csv"
        with trajectory.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerow(["X", "Y", "Z"])
        def append_position():
            position = self.snapshot()["position"]
            with trajectory.open("a", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerow(position)
            return position
        for step in range(1, limit + 1):
            self.context["step"] = step
            step_folder = folder / f"step_{step:02d}"
            with self.nav.pause_for_inference(c, True):
                urls, observation = self.capture(step_folder)
                if step == 1:
                    yaw_error = abs(h.wrap(observation["vehicle"]["yaw_deg"] - ablation["end"]["yaw_deg"]))
                    camera_error = max(abs(h.wrap(cam["yaw_deg"] - observation["vehicle"]["yaw_deg"]))
                                       for cam in observation["cameras"][:2])
                    result["first_frame_yaw_error_deg"] = yaw_error
                    result["first_frame_camera_yaw_error_deg"] = camera_error
                    h.atomic_json(folder / "first_frame_check.json", {"vehicle_yaw_error_deg": yaw_error,
                                  "front_camera_yaw_error_deg": camera_error, "passed": yaw_error <= 1 and camera_error <= 1})
                    if yaw_error > 1 or camera_error > 1:
                        raise HeadingGateError("First-frame vehicle/camera yaw failed gate before model call")
                    append_position()
                index, score, patch, position = self.nav.find_best_patch(str(step_folder / "0.png"), route["clip_text"],
                                                                       save_dir=str(step_folder / "clip_tiles"))
                patch.save(step_folder / "clip_best.png")
                h.atomic_json(step_folder / "clip.json", {"model": "ViT-B/32", "text": route["clip_text"],
                                                           "best_index": int(index), "score": float(score), "position": position})
                action, value, text = self.request(urls, self.prompt(route, position), step_folder)
            result["decisions"] += 1
            if step == 1 and abs(h.wrap(self.snapshot()["yaw_deg"] - ablation["end"]["yaw_deg"])) > 1:
                raise HeadingGateError("First-action yaw drift failed gate")
            before = self.snapshot()
            h.atomic_json(step_folder / "action.json", {"action": action, "value": value, "text": text,
                                                         "before": before})
            contact = self.collision()
            h.atomic_json(step_folder / "collision_before_action.json", contact)
            if h.new_collision(contact, collision_epoch):
                result["reason"] = "collision"
                result["navigation_collision"] = contact
                break
            if action == "stop":
                result["reason"] = "model_stop"
                break
            self.nav.execute_action(self.drone, action, value)
            result["actions_executed"] += 1
            time.sleep(self.state["config"]["post_action_wall_seconds"])
            append_position()
            after = self.snapshot()
            position_change = math.dist(before["position"], after["position"])
            yaw_change = abs(h.wrap(after["yaw_deg"] - before["yaw_deg"]))
            result["max_action_position_change_m"] = max(result["max_action_position_change_m"], position_change)
            result["max_action_yaw_change_deg"] = max(result["max_action_yaw_change_deg"], yaw_change)
            h.atomic_json(step_folder / "action_end.json", {**after, "position_change_m": position_change,
                                                          "yaw_change_deg": yaw_change, "executed": True})
            self.event("action_executed", action=action, value=value, position_change_m=position_change, yaw_change_deg=yaw_change)
            contact = self.collision()
            h.atomic_json(step_folder / "collision_after_action.json", contact)
            if h.new_collision(contact, collision_epoch):
                result["reason"] = "collision"
                result["navigation_collision"] = contact
                break
        append_position()
        result["navigation_end"] = self.snapshot()
        result["navigation_ended_at"] = now()
        result["trajectory_sha256"] = h.sha256(trajectory)
        score = h.metrics.compute_episode_metrics(validation.points(trajectory), route["start_position"],
                                                  route["goal_position"], route["success_threshold"])
        result["metrics"] = score
        if job["kind"] == "smoke":
            result["smoke_acceptance"] = h.smoke_acceptance(result)
            h.atomic_json(folder / "smoke_acceptance.json", result["smoke_acceptance"])
            if not result["smoke_acceptance"]["passed"]:
                h.atomic_json(folder / "result.json", result)
                raise PauseBatch("Smoke failed: a real executed action and measurable pose response are required")
        # Publish navigation independently of landing, so a landing RPC failure cannot erase a valid episode.
        result["complete"] = True
        h.atomic_json(folder / "result.json", result)
        try:
            c.landAsync(timeout_sec=30).join()
            h.atomic_json(folder / "landing.json", {"pose": self.snapshot(), "landed_state": int(c.getMultirotorState().landed_state)})
        except Exception as exc:
            h.atomic_json(folder / "landing.json", {"error_type": type(exc).__name__})
        self.drone.is_flying = False
        return result

    def run_job(self, job):
        for gate_attempt in range(2):
            job["attempts"] += 1
            job["status"] = "running"
            self.context = {"job": job["id"], "attempt": job["attempts"]}
            folder = self.directory / "attempts" / job["id"] / f"{job['attempts']:03d}"
            folder.mkdir(parents=True)
            self.save()
            self.event("flight_started", route=job["route"], kind=job["kind"])
            try:
                self.ensure_engine_ready()
                result = self.flight(job, folder)
            except Exception as exc:
                h.atomic_json(folder / "failure.json", {"error_type": type(exc).__name__,
                                                        "message": str(exc) if isinstance(exc, PauseBatch) else "See local traceback.log",
                                                        "at": now(), "complete": False})
                self.event("flight_interrupted", error_type=type(exc).__name__)
                if not isinstance(exc, PauseBatch):
                    (folder / "traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
                    self.stop_engine()  # Reconnect a broken local RPC/simulator on the next job.
                if isinstance(exc, SceneFrameError):
                    raise
                if isinstance(exc, HeadingGateError) and gate_attempt == 0:
                    job["status"] = "pending"
                    self.save()
                    continue
                # Per-flight protection/transport/model failures are not valid
                # outcomes and must not hold up other routes or repeats.
                job.update(status="deferred", deferrals=job.get("deferrals", 0) + 1,
                           last_error_type=type(exc).__name__,
                           last_error_message=str(exc) if isinstance(exc, PauseBatch) else "See failed attempt diagnostics",
                           last_failed_attempt=str(folder.relative_to(self.directory)), deferred_at=now())
                self.save()
                self.event("job_deferred", error_type=type(exc).__name__, next_action="continue_other_jobs")
                return False
            job.update(status="complete", result_path=str((folder / "result.json").relative_to(self.directory)))
            self.save()
            self.event("flight_completed", reason=result["reason"], metrics=result["metrics"])
            self.write_checks()
            return True

    def ensure_engine_ready(self):
        if self.engine is None or self.engine.poll() is not None or not port_open(self.args.rpc_port):
            self.stop_engine()
            self.start_engine()

    def run_scheduled_jobs(self, jobs, kind):
        self.state["scheduler_version"] = SCHEDULER_VERSION
        self.state["scheduler_phase"] = f"{kind}_primary"
        self.save()
        # Snapshot the primary queue: an error cannot reinsert the same job here.
        for job in [j for j in jobs if j["status"] == "pending"]:
            self.run_job(job)
        self.context = {}
        self.state["scheduler_phase"] = f"{kind}_recovery"
        self.state.setdefault(f"{kind}_primary_finished_at", now())
        self.save()
        while any(j["status"] != "complete" for j in jobs):
            self.state["recovery_round"] = self.state.get("recovery_round", 0) + 1
            queue = [j for j in jobs if j["status"] != "complete"]
            self.event("recovery_round_started", recovery_round=self.state["recovery_round"], jobs=[j["id"] for j in queue])
            for job in queue:
                self.run_job(job)
            self.context = {}
            self.save()
            remaining = [j["id"] for j in jobs if j["status"] != "complete"]
            if remaining:
                self.event("recovery_round_wait", jobs=remaining, cooldown_seconds=RECOVERY_COOLDOWN_SECONDS)
                # Keep going across rounds, but do not hammer a failing model or simulator.
                time.sleep(RECOVERY_COOLDOWN_SECONDS)
        self.state["scheduler_phase"] = f"{kind}_finished"
        self.save()

    def write_checks(self):
        rows = []
        for job in self.state["jobs"]:
            if job["status"] == "complete":
                result = json.loads((self.directory / job["result_path"]).read_text(encoding="utf-8"))
                rows.append({"id": job["id"], "kind": job["kind"], "repeat": job["repeat"], **result["heading_check"],
                             "first_frame_yaw_error_deg": result["first_frame_yaw_error_deg"],
                             "first_frame_camera_yaw_error_deg": result["first_frame_camera_yaw_error_deg"]})
        h.write_csv(self.directory / "heading_checks.csv", rows)

    def preflight(self):
        self.load_runtime()
        if not self.args.engine.is_file() or not self.args.project.is_file():
            raise PauseBatch("Missing Unreal engine or Downtown project")
        self.verify_manifest()
        h.atomic_json(self.directory / "preflight.json", {"time": now(), "passed": True,
                      "routes": 30, "baseline_csv_count": self.manifest["baseline_csv_count"],
                      "api_key_present": True, "protected_files": len(self.manifest["sha256"]),
                      "note": "Network, CLIP and simulator verified by smoke flights; key is never persisted"})
        self.event("offline_preflight_passed", routes=30, baseline_csv_count=300)

    def audit_initializations(self):
        """No model calls and no formal results: inspect every restored route."""
        self.state["status"] = "initialization_audit_running"
        self.save()
        rows = []
        for number in range(1, 31):
            route = self.routes[str(number)]
            row = {"route": number, "passed": False}
            route_directory = self.directory / "coordinate_preflight" / f"route_{number:02d}"
            previous = max((int(p.name) for p in route_directory.glob("*") if p.is_dir() and p.name.isdigit()), default=0)
            for attempt in range(previous + 1, previous + 3):
                folder = self.directory / "coordinate_preflight" / f"route_{number:02d}" / f"{attempt:03d}"
                folder.mkdir(parents=True, exist_ok=True)
                self.context = {"coordinate_audit_route": number, "attempt": attempt}
                try:
                    self.ensure_engine_ready()
                    check, ablation = self.calibrate(route, folder)
                    self.capture(folder / "initial_views")
                    row.update(passed=True, heading_check=check, ablation_end=ablation["end"],
                               record_path=str(folder.relative_to(self.directory)))
                    break
                except HeadingGateError as exc:
                    h.atomic_json(folder / "failure.json", {"error_type": type(exc).__name__, "message": str(exc)})
                    row.update(error_type=type(exc).__name__, message=str(exc))
            rows.append(row)
            h.atomic_json(self.directory / "coordinate_preflight/summary.json", {"routes": rows, "passed": len(rows) == 30 and all(r["passed"] for r in rows), "diagnostic_only": True})
            self.event("coordinate_audit_route_finished", **row)
        self.context = {}
        self.state["status"] = "initialization_audit_complete"
        self.save()
        if not all(row["passed"] for row in rows):
            raise PauseBatch("One or more route initialization audits failed; see coordinate_preflight/summary.json")

    def run(self):
        self.state.pop("failure_type", None)
        self.state.pop("failure_message", None)
        self.state.update(status="running", runner_pid=os.getpid())
        self.save()
        self.preflight()
        if self.args.preflight_only:
            self.state["status"] = "preflight_complete"
            self.save()
            return
        if self.args.audit_initializations or self.args.audit_then_run:
            self.audit_initializations()
            if self.args.audit_initializations:
                return
            self.state["status"] = "running"
            self.save()
        if self.args.formal_only and not all(j["status"] == "complete" for j in self.state["jobs"] if j["kind"] == "smoke"):
            raise PauseBatch("Three smoke flights must complete before formal batch")
        if not self.args.formal_only:
            self.run_scheduled_jobs([j for j in self.state["jobs"] if j["kind"] == "smoke"], "smoke")
        if not self.args.smoke_test:
            self.run_scheduled_jobs([j for j in self.state["jobs"] if j["kind"] == "formal"], "formal")
        self.context = {}
        self.verify_manifest()
        self.write_checks()
        if self.args.smoke_test:
            self.state["status"] = "smoke_complete"
            self.event("smoke_passed", routes=list(h.SMOKE_ROUTES), formal_started=False)
        else:
            from map2.scripts.analyze_heading_ablation import analyze_batch
            analyze_batch(self.directory)
            self.state["status"] = "complete"
            self.event("batch_complete", formal_flights=300)
        self.save()

    def stop_engine(self):
        if self.engine and self.engine.poll() is None:
            # Only the process created by this runner and its descendants.
            import psutil
            try:
                owner = psutil.Process(self.engine.pid)
                children = owner.children(recursive=True)
                for process in children + [owner]:
                    try:
                        process.terminate()
                    except psutil.NoSuchProcess:
                        pass
                _, alive = psutil.wait_procs(children + [owner], timeout=5)
                for process in alive:
                    process.kill()
            except psutil.NoSuchProcess:
                pass
        self.engine = None
        self.drone = None

    def close(self):
        self.stop_engine()
        if self.ai:
            self.ai.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--upgrade-scheduler", action="store_true",
                        help="Register the authorized scheduling-only source revision while preserving frozen data/results")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--formal-only", action="store_true")
    mode.add_argument("--audit-initializations", action="store_true", help="Check all 30 starts, scene anchors and heading pairs without model calls")
    mode.add_argument("--audit-then-run", action="store_true", help="Require all 30 initialization audits, then smoke flights, then formal flights")
    parser.add_argument("--engine", type=Path, default=ENGINE)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--rpc-port", type=int, default=41453)
    parser.add_argument("--endpoint", default=h.ENDPOINT)
    parser.add_argument("--scene-frame-reference", type=Path,
                        default=h.MAP / "heading_ablation/downtown_scene_frame.json")
    args = parser.parse_args(argv)
    batch = None
    try:
        scene_frame.prepare_reference()
        batch = Batch(args)
        batch.run()
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        if batch:
            batch.state.update(status="paused", failure_type=type(exc).__name__,
                               failure_message=str(exc) if isinstance(exc, PauseBatch) else "See local traceback.log")
            batch.save()
            # Traceback only for local simulator/code failures; SDK exceptions may contain request data.
            if not isinstance(exc, PauseBatch):
                (batch.directory / "traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
            batch.event("batch_paused", error_type=type(exc).__name__)
        else:
            print(f"Batch initialization failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if batch:
            batch.close()


if __name__ == "__main__":
    raise SystemExit(main())
