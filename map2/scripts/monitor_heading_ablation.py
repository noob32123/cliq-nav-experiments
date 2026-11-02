"""Read-only flight-ledger observer. Writes only BATCH/monitor; never calls AirSim."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import psutil
from map1.scripts.heading_ablation import atomic_json


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def wrap(value):
    return (value + 180) % 360 - 180


def theory(route):
    points = route["history_points"]
    segments = [[points[i + 1][k] - points[i][k] for k in (0, 1)] for i in range(5)]
    vector = [sum(w * v[k] for w, v in zip(route["weights"], segments)) for k in (0, 1)]
    delta = wrap(math.degrees(math.atan2(segments[-1][1], segments[-1][0])) -
                 math.degrees(math.atan2(vector[1], vector[0])))
    return delta, route["baseline_turn_deg"] + delta


def inspect_arm(route, data, arm, xy_tolerance=0.5, z_tolerance=0.5):
    """Post-takeoff Z is -altitude, distinct from reset spawn Z=-1."""
    delta, ablation = theory(route)
    expected = route["baseline_turn_deg"] if arm == "baseline" else ablation
    start = data["start"]
    xyz = start["position"]
    if not all(math.isfinite(v) for v in [*xyz, start["yaw_deg"], data["requested_relative_turn_deg"]]):
        raise ValueError("Non-finite initialization telemetry")
    row = {"arm": arm, "route": route["route"], "expected_x": route["start_position"][0],
           "expected_y": route["start_position"][1], "expected_flight_z": -route["altitude"],
           "reset_spawn_z": route["start_position"][2], "actual_x": xyz[0], "actual_y": xyz[1], "actual_z": xyz[2],
           "xy_error_m": math.dist(xyz[:2], route["start_position"][:2]),
           "altitude_error_m": abs(xyz[2] + route["altitude"]),
           "expected_turn_deg": expected, "requested_turn_deg": data["requested_relative_turn_deg"],
           "command_error_deg": abs(expected - data["requested_relative_turn_deg"]),
           "start_yaw_deg": start["yaw_deg"], "end_yaw_deg": None,
           "actual_relative_turn_deg_wrapped": None, "rotation_error_deg": None,
           "theoretical_delta_deg": delta, "finalized": "end" in data, "issues": []}
    if row["xy_error_m"] > xy_tolerance:
        row["issues"].append("start_xy_deviation")
    if row["altitude_error_m"] > z_tolerance:
        row["issues"].append("start_altitude_deviation")
    if row["command_error_deg"] > 1e-8:
        row["issues"].append("wrong_turn_command")
    if abs(wrap(data["target_relative_yaw_deg"] - start["yaw_deg"] - expected)) > 1e-8:
        row["issues"].append("wrong_relative_target")
    rates = [c for c in data.get("commands", []) if "yaw_rate_deg_s" in c]
    if rates and (abs(rates[0]["duration_sim_seconds"] - abs(expected) / 30) > 1e-8 or
                  abs(rates[0]["yaw_rate_deg_s"] - math.copysign(30, expected)) > 1e-8):
        row["issues"].append("wrong_initial_turn_rate_or_duration")
    if "end" in data:
        end_yaw = data["end"]["yaw_deg"]
        row["end_yaw_deg"] = end_yaw
        row["actual_relative_turn_deg_wrapped"] = wrap(end_yaw - start["yaw_deg"])
        row["rotation_error_deg"] = abs(wrap(end_yaw - start["yaw_deg"] - expected))
        if row["rotation_error_deg"] > 0.35:
            row["issues"].append("actual_turn_not_reached")
        if data.get("failed"):
            row["issues"].append("initial_turn_not_stable")
    return row


def confirmed_reset_valid(route, data):
    if not data or not data.get("passed") or not data.get("reset_confirmed_pose"):
        return False
    pose = data["reset_confirmed_pose"]
    takeoff = data["post_takeoff_pose"]
    values = [*pose["position"], *pose["orientation_xyzw"], pose["yaw_deg"], *takeoff["position"]]
    return (all(math.isfinite(x) for x in values) and
            math.dist(pose["position"], route["start_position"]) <= .1 and
            abs(wrap(pose["yaw_deg"] - route["initial_yaw_deg"])) <= .5 and
            math.dist(takeoff["position"][:2], route["start_position"][:2]) <= .5 and
            abs(takeoff["position"][2] + route["altitude"]) <= .5 and
            data["scene_frame"]["passed"] and data["scene_frame"]["max_anchor_error_m"] <= .05)


def owned_process(pid, batch):
    if not pid:
        return {"pid": pid, "alive": False}
    try:
        process = psutil.Process(pid)
        return {"pid": pid, "alive": process.is_running(),
                "matches_batch": str(batch).lower() in " ".join(process.cmdline()).lower()}
    except psutil.NoSuchProcess:
        return {"pid": pid, "alive": False}
    except psutil.AccessDenied:
        return {"pid": pid, "alive": None, "inspection_unavailable": True}


class Observer:
    def __init__(self, batch, interval=2):
        self.batch = batch.resolve()
        self.out = self.batch / "monitor"
        self.out.mkdir(exist_ok=True)
        self.interval = interval
        self.cache = {}
        self.input_revision = 0
        self.csv_revision = -1
        self.known = set()
        self.initialized = False
        self.new_alert_count = 0
        self.started = now()
        self.last_completed = None
        self.no_progress_rounds = 0
        self.last_round = None

    def cached(self, path):
        stamp = path.stat().st_mtime_ns
        if path not in self.cache or self.cache[path][0] != stamp:
            self.cache[path] = (stamp, read(path))
            self.input_revision += 1
        return self.cache[path][1]

    def scan(self):
        state = read(self.batch / "state.json")
        routes = self.cached(self.batch / "routes.json")
        jobs = {j["id"]: j for j in state["jobs"]}
        rows, problems = [], []
        frame_records = []
        frame_path = self.batch / "scene_frame_check.json"
        frame = self.cached(frame_path) if frame_path.exists() else None
        if frame and (not frame["passed"] or frame["max_anchor_error_m"] > .05):
            problems.append({"key": "scene_frame_mismatch", "issue": "scene_landmarks_disagree_with_historical_frame", "severity": "critical"})
        for job in jobs.values():
            if job["status"] != "complete" or not state.get("config", {}).get("scene_frame_reference"):
                continue
            folder = (self.batch / job["result_path"]).parent
            for arm in ("baseline", "ablation"):
                path = folder / f"{arm}_reset_takeoff.json"
                data = self.cached(path) if path.exists() else None
                valid = confirmed_reset_valid(routes[str(job["route"])], data)
                if data:
                    frame_records.append(data)
                if not valid:
                    problems.append({"key": f"{job['id']}:{arm}:scene_frame_or_reset_invalid", "job": job["id"],
                                     "issue": "accepted_result_has_invalid_scene_frame_or_missing_confirmed_reset", "severity": "critical",
                                     "record": str(path.relative_to(self.batch))})
        for path in sorted((self.batch / "attempts").glob("*/*/*_initialization.json")):
            job = jobs.get(path.parent.parent.name)
            if job is None:
                continue
            route = routes[str(job["route"])]
            arm = "baseline" if path.name.startswith("baseline") else "ablation"
            accepted = job["status"] == "complete" and (self.batch / job["result_path"]).parent == path.parent
            try:
                row = inspect_arm(route, self.cached(path), arm)
                row.update(job=job["id"], attempt=path.parent.name, accepted=accepted,
                           path=str(path.relative_to(self.batch)))
                for issue in row["issues"]:
                    problems.append({"key": f"{job['id']}:{issue}:{'accepted' if accepted else 'attempt'}",
                                     "job": job["id"], "attempt": path.parent.name, "issue": issue,
                                     "severity": "critical" if accepted or issue.startswith("wrong_") else "warning",
                                     "record": row["path"], "protected_or_unaccepted_attempt": not accepted})
                rows.append(row)
            except (ValueError, KeyError, TypeError) as exc:
                problems.append({"key": f"{job['id']}:invalid_telemetry", "job": job["id"],
                                 "issue": "invalid_initialization_telemetry", "severity": "critical",
                                 "error_type": type(exc).__name__, "record": str(path.relative_to(self.batch))})
        checks = []
        for path in sorted((self.batch / "attempts").glob("*/*/heading_check.json")):
            job = jobs.get(path.parent.parent.name)
            if job is None:
                continue
            record = self.cached(path)
            checks.append(record)
            if not record["passed"]:
                problems.append({"key": f"{job['id']}:paired_heading_rejected", "job": job["id"],
                                 "issue": "paired_heading_rejected", "severity": "warning",
                                 "angle_error_deg": record["actual_delta_error_deg"],
                                 "initial_yaw_difference_deg": record["initial_yaw_error_deg"],
                                 "record": str(path.relative_to(self.batch)), "protected_or_unaccepted_attempt": True})
        runner = owned_process(state.get("runner_pid"), self.batch)
        engine = owned_process(state.get("engine_pid"), self.batch)
        activity = max(p.stat().st_mtime for p in (self.batch / "state.json", self.batch / "events.jsonl") if p.exists())
        age = max(0, time.time() - activity)
        if state["status"] in {"running", "initialization_audit_running"} and runner["alive"] is False:
            problems.append({"key": "runner_exited", "issue": "experiment_runner_not_running", "severity": "critical"})
        if state["status"] == "running" and age > 600:
            problems.append({"key": "experiment_stalled", "issue": "no_experiment_activity_over_10_minutes", "severity": "critical"})
        round_number = state.get("recovery_round", 0)
        if round_number != self.last_round:
            self.no_progress_rounds = self.no_progress_rounds + 1 if state["formal_completed"] == self.last_completed else 0
            self.last_completed, self.last_round = state["formal_completed"], round_number
        if self.no_progress_rounds >= 3:
            problems.append({"key": "recovery_no_progress", "issue": "three_recovery_rounds_without_completed_flight", "severity": "warning"})
        # Historical rejected attempts are visible in the report, but not re-announced at startup.
        for problem in problems:
            if self.initialized and problem["key"] not in self.known:
                self.new_alert_count += 1
                with (self.out / "alerts.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"time": now(), **problem}, ensure_ascii=False) + "\n")
            self.known.add(problem["key"])
        self.initialized = True
        accepted = [row for row in rows if row["accepted"]]
        max_value = lambda name: max((row[name] for row in accepted if row[name] is not None), default=None)
        completed_checks = []
        for job in jobs.values():
            if job["kind"] == "formal" and job["status"] == "complete":
                completed_checks.append(self.cached(self.batch / job["result_path"])["heading_check"])
        summary = {"time": now(), "observer_started_at": self.started, "observer_pid": __import__("os").getpid(),
                   "observer_interval_seconds": self.interval, "experiment_status": state["status"],
                   "formal_completed": state["formal_completed"], "formal_required": 300,
                   "formal_deferred": state.get("formal_deferred", 0), "scheduler_phase": state.get("scheduler_phase"),
                   "recovery_round": round_number, "runner": runner, "engine": engine,
                   "activity_age_seconds": age, "active_jobs": [j["id"] for j in jobs.values() if j["status"] == "running"],
                   "completed_results_checked": len(completed_checks),
                   "accepted_initialization_arms_checked": len(accepted),
                   "max_accepted_xy_error_m": max_value("xy_error_m"),
                   "max_accepted_altitude_error_m": max_value("altitude_error_m"),
                   "max_accepted_turn_command_error_deg": max_value("command_error_deg"),
                   "max_accepted_turn_execution_error_deg": max_value("rotation_error_deg"),
                   "max_completed_paired_delta_error_deg": max((c["actual_delta_error_deg"] for c in completed_checks), default=None),
                   "accepted_initialization_issue_count": sum(bool(row["issues"]) for row in accepted),
                   "new_alert_count_since_observer_start": self.new_alert_count,
                   "problem_job_ids": sorted({p["job"] for p in problems if "job" in p}),
                   "current_critical_problems": [p for p in problems if p["severity"] == "critical"],
                   "thresholds": {"coordinate_xy_warning_m": 0.5, "altitude_warning_m": 0.5,
                                  "relative_turn_execution_deg": 0.35, "paired_delta_deg": 1, "paired_initial_yaw_deg": 0.5},
                   "z_interpretation": "compare takeoff initialization Z with -route.altitude; reset spawn Z=-1 is a separate stage",
                   "reset_pose_telemetry_available": bool(frame_records) and all("reset_confirmed_pose" in x for x in frame_records),
                   "scene_frame_verified": bool(frame and frame["passed"]),
                   "scene_frame_max_anchor_error_m": max([frame["max_anchor_error_m"]] + [r["scene_frame"]["max_anchor_error_m"] for r in frame_records]) if frame else None,
                   "accepted_scene_frame_and_reset_records_checked": len(frame_records),
                   "monitor_is_read_only": True, "does_not_consume_airsim_collision_flags": True}
        atomic_json(self.out / "status.json", summary)
        if rows and self.csv_revision != self.input_revision:
            with (self.out / "initialization_checks.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows({**row, "issues": ";".join(row["issues"])} for row in rows)
            self.csv_revision = self.input_revision
        lines = ["# 起点与初始化转角实时核查", "", f"更新：{summary['time']}；只读监控间隔 {self.interval} 秒。",
                 f"正式完成 {state['formal_completed']}/300，后延 {state.get('formal_deferred', 0)}；阶段 {state.get('scheduler_phase')}。", "",
                 f"已接受初始化最大 XY 偏差：{max_value('xy_error_m')} m；高度偏差：{max_value('altitude_error_m')} m。",
                 f"转角命令最大误差：{max_value('command_error_deg')}°；实际相对转角最大误差：{max_value('rotation_error_deg')}°。",
                 f"有效正式实验最大配对角差误差：{summary['max_completed_paired_delta_error_deg']}°。",
                 f"有效初始化异常条数：{summary['accepted_initialization_issue_count']}；新告警数：{self.new_alert_count}。", "",
                 "XY 按路线起点核对；起飞后的 Z 按该路线飞行高度核对，不能与复位 Z=-1 混淆。",
                 f"场景原点核验：{summary['scene_frame_verified']}；固定地标最大偏差 {summary['scene_frame_max_anchor_error_m']} m。",
                 f"已核对完成实验的场景/复位记录 {len(frame_records)} 份；已确认复位位置遥测：{summary['reset_pose_telemetry_available']}。",
                 "实际转角按初始 yaw 的相对变化核对，保留角度环绕；不强制使用末段的绝对 yaw。", "",
                 "存在保护或监控异常记录的任务：" + ", ".join(summary["problem_job_ids"]), "",
                 "## 当前关键异常", ""]
        lines.extend(json.dumps(p, ensure_ascii=False) for p in summary["current_critical_problems"])
        (self.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 0.5:
        parser.error("interval must be >=0.5s")
    observer = Observer(args.batch, args.interval)
    (observer.out / "observer.pid").write_text(str(__import__("os").getpid()), encoding="utf-8")
    while True:
        scan_started = time.monotonic()
        try:
            result = observer.scan()
            if args.once or result["experiment_status"] == "complete":
                print(json.dumps(result, ensure_ascii=False), flush=True)
                return
        except (OSError, ValueError, KeyError) as exc:
            atomic_json(observer.out / "observer_error.json", {"time": now(), "error_type": type(exc).__name__, "message": str(exc)})
            if args.once:
                raise
        time.sleep(max(0.1, args.interval - (time.monotonic() - scan_started)))


if __name__ == "__main__":
    main()
