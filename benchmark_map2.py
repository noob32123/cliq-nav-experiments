"""Run the real Map2/CLIP/Qwen loop in an owned standalone UE instance.

Uses per-run settings, preserving the editor and the user's global settings.
Only processes started here are terminated. Results are stored under map2/.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from datetime import datetime

import psutil
import setup_path
import airsim

ROOT = Path(__file__).resolve().parent
ENGINE = Path(os.environ.get("UE4_EDITOR", r"H:\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe"))
PROJECT = Path(os.environ.get("DOWNTOWN_PROJECT", r"H:\airsimdowntown\airsimdowntown\airsimdowntown.uproject"))
SETTINGS = Path.home() / "Documents/AirSim/settings.json"


def port_open():
    try:
        with socket.create_connection(("127.0.0.1", 41451), timeout=0.5):
            return True
    except OSError:
        return False


def monitor(stop, destination):
    psutil.cpu_percent()
    with destination.open("w", encoding="utf-8") as file:
        while not stop.wait(1):
            record = {"wall_time": time.time(), "cpu_percent": psutil.cpu_percent(),
                      "ram_percent": psutil.virtual_memory().percent}
            try:
                result = subprocess.run([
                    "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True,
                    timeout=4, creationflags=subprocess.CREATE_NO_WINDOW)
                gpu, used, total = map(float, result.stdout.strip().splitlines()[0].split(","))
                record.update(gpu_percent=gpu, gpu_used_mib=used, gpu_total_mib=total)
            except Exception as exc:
                record["monitor_error"] = type(exc).__name__
            file.write(json.dumps(record) + "\n")
            file.flush()


def terminate_owned(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def run(clock, iterations, destination, probe_only=False, local_only=False, flights=1, start_position=None):
    if port_open():
        raise RuntimeError("Port 41451 is occupied; refusing to interrupt an existing simulation")
    destination.mkdir(parents=True)
    config = json.loads(SETTINGS.read_text(encoding="utf-8-sig"))
    config["ClockSpeed"] = clock
    config_path = destination / "settings.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    engine = nav = None
    stop = threading.Event()
    sampler = None
    try:
        engine = subprocess.Popen([
            str(ENGINE), str(PROJECT), "/Game/maps/airsim_downtown", "-game",
            "-windowed", "-ResX=1280", "-ResY=720", "-nosound", "-unattended",
            f"-settings={config_path}", f"-abslog={destination / 'unreal.log'}"],
            cwd=ROOT, startupinfo=startup, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"ClockSpeed={clock:g}: started UE PID {engine.pid}", flush=True)
        deadline = time.monotonic() + 120
        while not port_open():
            if engine.poll() is not None:
                raise RuntimeError(f"UE exited ({engine.returncode}); see unreal.log")
            if time.monotonic() > deadline:
                raise TimeoutError("AirSim RPC did not start within 120 seconds")
            time.sleep(1)
        probe = airsim.MultirotorClient(timeout_value=15)
        actual_config = json.loads(probe.getSettingsString())
        loaded = actual_config.get("ClockSpeed", 1)
        if loaded != clock:
            raise RuntimeError(f"Loaded ClockSpeed={loaded}, requested {clock}")
        sim0 = probe.getMultirotorState().timestamp
        wall0 = time.perf_counter()
        time.sleep(3)
        measured = (probe.getMultirotorState().timestamp - sim0) / 1e9 / (time.perf_counter() - wall0)
        if probe_only:
            print(f"Local RPC probe: ClockSpeed={clock:g}, actual={measured:.3f}x", flush=True)
            return {"returncode": 0, "errors": [], "steps": 0, "probe_only": True}
        mode = "local fixed actions + CLIP (NO Qwen)" if local_only else "full CLIP + Qwen loop"
        print(f"ClockSpeed={clock:g}: measured idle clock {measured:.3f}x; starting {mode}", flush=True)
        before = set((ROOT / "map2").glob("*flight_logs_*"))
        sampler = threading.Thread(target=monitor, args=(stop, destination / "load.jsonl"), daemon=True)
        sampler.start()
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        with (destination / "navigation.log").open("w", encoding="utf-8") as log:
            command = ([sys.executable, "-u", str(Path(__file__).resolve()), "--local-worker",
                        "--iterations", str(iterations)] if local_only else
                       [sys.executable, "-u", str(ROOT / "nav_qwen_zgc.py"), "--flights", str(flights),
                        "--max-iterations", str(iterations), "--profile"])
            if start_position is not None:
                command.extend(["--start-position", *map(str, start_position)])
            nav = subprocess.Popen(command, cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            nav.wait(timeout=900)
        profiles = sorted(set((ROOT / "map2").glob("*flight_logs_*")) - before)
        records = []
        for folder in profiles:
            profile = folder / "profile.jsonl"
            if profile.exists():
                records.extend(json.loads(line) for line in profile.read_text(encoding="utf-8").splitlines())
        steps = [row for row in records if row["event"] == "step"]
        decisions = [row for row in records if row["event"] == "decision"]
        warm = steps[1:]
        summary = {"mode": mode, "flights": flights, "start_position_override": start_position,
                   "calibration_altitude_m": 80 if local_only else None,
                   "clock_speed": clock, "idle_actual_clock": measured, "returncode": nav.returncode,
                   "trajectory_directories": list(map(str, profiles)), "steps": len(steps),
                   "warm_steps": len(warm),
                   "decisions": len(decisions),
                   "pause_during_inference": next((row.get("pause_during_inference", False)
                                                    for row in records if row["event"] == "config"), False),
                   "max_inference_drift_m": max((row["inference_drift_m"] for row in decisions), default=None),
                   "max_inference_sim_seconds": max((row["inference_sim_seconds"] for row in decisions), default=None),
                   "errors": [row for row in records if row["event"].endswith("error")],
                   "collision_steps": sum(bool(row["has_collided"]) for row in steps)}
        for key in ["images_seconds", "clip_seconds", "qwen_seconds", "action_seconds", "wall_seconds"]:
            summary["warm_median_" + key] = statistics.median(row[key] for row in warm) if warm else None
        summary["loop_actual_clock"] = (sum(row["sim_seconds"] for row in warm) /
                                         sum(row["wall_seconds"] for row in warm)) if warm else None
        summary["action_actual_clock"] = (sum(row["action_sim_seconds"] for row in steps) /
                                          sum(row["action_seconds"] for row in steps)) if steps and all("action_sim_seconds" in row for row in steps) else None
        summary["final_paused"] = bool(probe.simIsPause())
        summary["final_landed_state"] = int(probe.getMultirotorState().landed_state)
        stop.set()
        sampler.join(timeout=6)
        loads = [json.loads(line) for line in (destination / "load.jsonl").read_text().splitlines()]
        for key in ["cpu_percent", "gpu_percent", "gpu_used_mib"]:
            values = [row[key] for row in loads if key in row]
            summary[key + "_mean"] = statistics.mean(values) if values else None
            summary[key + "_max"] = max(values) if values else None
        (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return summary
    finally:
        stop.set()
        if sampler:
            sampler.join(timeout=6)
        terminate_owned(nav)
        terminate_owned(engine)


def local_worker(iterations):
    """Exercise the production image/CLIP/action loop without any remote request."""
    import nav_qwen_zgc as navigation

    class LocalDecisions:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=self)
            self.index = 0

        def create(self, **kwargs):
            actions = ["forward 2", "backward 2", "up 1", "down 1", "turn_right 30", "turn_left 30"]
            action = actions[self.index % len(actions)]
            self.index += 1
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=action))])

    navigation.OpenAI = LocalDecisions
    os.environ["OPENAI_API_KEY"] = "local-benchmark-no-network"
    original_reset = navigation.reset_to_start
    original_takeoff = navigation.FixedDroneController.takeoff

    def calibration_reset(client, position, orientation, clock_speed=1.0):
        original_reset(client, (position[0], position[1], -80), (0, 0, 0, 1), clock_speed)

    def calibration_takeoff(controller, altitude, settle_seconds=1.0):
        return original_takeoff(controller, 80, settle_seconds)

    def calibration_land(controller):
        # Altitude is intentionally high for unobstructed motion calibration.
        # End with hover; the owned simulator is terminated immediately afterward.
        controller.client.hoverAsync().join()
        controller.is_flying = False

    navigation.reset_to_start = calibration_reset
    navigation.FixedDroneController.takeoff = calibration_takeoff
    navigation.FixedDroneController.land = calibration_land
    print("LOCAL FIXED ACTION BENCHMARK: Qwen is replaced; NO network/model request", flush=True)
    print("Calibration-only overrides: 80 m altitude, level orientation, finish with hover", flush=True)
    sys.argv = ["nav_qwen_zgc.py", "--flights", "1", "--max-iterations", str(iterations), "--profile"]
    navigation.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clocks", type=float, nargs="+", default=[])
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--flights", type=int, default=1, help="Number of flights for the real Qwen benchmark")
    parser.add_argument("--start-position", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--probe-only", action="store_true", help="Only start local simulation and read clock; no flight or model request")
    parser.add_argument("--local-only", action="store_true", help="Keep images and CLIP but use fixed local actions instead of Qwen")
    parser.add_argument("--local-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if any(not 0 < clock <= 100 for clock in args.clocks) or args.iterations < 1 or args.flights < 1:
        parser.error("ClockSpeed must be in (0, 100]; iterations/flights must be positive")
    if (args.local_only or args.local_worker) and args.flights != 1:
        parser.error("Local fixed-action calibration uses one flight; increase --iterations instead")
    if (args.local_only or args.local_worker) and args.start_position is not None:
        parser.error("Local calibration defines its own start position")
    if args.local_worker:
        local_worker(args.iterations)
        raise SystemExit(0)
    if not args.clocks:
        parser.error("--clocks is required")
    root = ROOT / "map2" / ("benchmark_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    print(f"Benchmark output: {root}", flush=True)
    for index, clock in enumerate(args.clocks):
        result = run(clock, args.iterations, root / f"{index:02d}_clock_{clock:g}", args.probe_only,
                     args.local_only, args.flights, args.start_position)
        if result["returncode"] or result["errors"] or (result["steps"] == 0 and not args.probe_only):
            raise SystemExit("Incomplete navigation baseline; inspect logs before increasing clock speed")
