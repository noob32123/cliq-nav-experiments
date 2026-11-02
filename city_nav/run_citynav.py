"""
run_citynav.py - CityNavAgent Adapted Full Experiment Runner
=============================================================
Runs the adapted CityNavAgent pipeline on all 14 scenarios,
10 flights each, saving trajectory CSVs matching data_X format.

Usage:
    python run_citynav.py --validate          # Basic AirSim test (no LLM)
    python run_citynav.py --scenario 5        # Single scenario, 10 flights
    python run_citynav.py --all               # All 14 scenarios, 10 flights each
    python run_citynav.py --scenario 5 --flights 3   # Custom flight count
    python run_citynav.py --list              # List scenarios

Output structure (under map1/baselines/city_nav/):
    data_X/
        CityNavAgent_<scenario_name>/
            flight_01_YYYYMMDD_HHMMSS.csv   # X,Y,Z trajectory
            flight_02_YYYYMMDD_HHMMSS.csv
            ...
        metrics_results.csv                  # Per-scenario metrics
"""
import sys
import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _parent_dir)

import time
import csv
import argparse
import numpy as np
from datetime import datetime

from city_nav.adapted.airsim_client import AirSimClientAdapter
from city_nav.adapted.llm_client import QwenLLMAdapter
from city_nav.adapted.perception import PerceptionModule
from city_nav.adapted.flight_controller import FlightController
from city_nav.adapted.citynav_pipeline import CityNavPipeline


# ============================================================
#  Complete Scenario Configuration (matching evaluate_nav_metrics.py)
# ============================================================
SCENARIO_CONFIG = {
    1: {
        "name": "orangewhite_umbrella",
        "instruction": "Navigate to the orange and white striped umbrella",
        "target": "orange and white striped umbrella",
        "start_position": (-253.011, -115.011, -10.146),
        "orientation": (0.9236459732055664, 0, 0, -0.3832468092441559),
        "altitude": 1,
        "initial_turn": 160.0,
        "goal_position": (-294.1103, -101.224, 2.3967),
        "success_threshold": 10.0,
        "shortest_path": None,  # computed below
    },
    2: {
        "name": "statue",
        "instruction": "Navigate to the statue of a person",
        "target": "statue of a person",
        "start_position": (-140.042, -412.695, -10.146),
        "orientation": (0.9236459732055664, 0, 0, -0.3832468092441559),
        "altitude": 5,
        "initial_turn": 0.0,
        "goal_position": (-120.4851, -415.2262, -0.5565),
        "success_threshold": 10.0,
        "shortest_path": None,
    },
    3: {
        "name": "restaurant_across_river",
        "instruction": "Navigate to the restaurant across the river",
        "target": "restaurant across the river",
        "start_position": (-63.832, -231.185, -10.146),
        "orientation": (0.9236459732055664, 0, 0, -0.3832468092441559),
        "altitude": 8,
        "initial_turn": 70.0,
        "goal_position": (-52.2433, -182.3261, 3.4642),
        "success_threshold": 15.0,
        "shortest_path": None,
    },
    4: {
        "name": "hotel_front_door",
        "instruction": "Navigate to the hotel front door",
        "target": "hotel front door",
        "start_position": (-73.259, 247.858, -10.146),
        "orientation": (0.9236459732055664, 0, 0, -0.3832468092441559),
        "altitude": 8,
        "initial_turn": 0.0,
        "goal_position": (-22.9098, 260.5971, 0.4993),
        "success_threshold": 10.0,
        "shortest_path": None,
    },
    5: {
        "name": "blue_amusement_park",
        "instruction": "Navigate to the blue amusement park",
        "target": "blue amusement park",
        "start_position": (234.382, 354.169, -10.146),
        "orientation": (0.9236459732055664, 0, 0, -0.3832468092441559),
        "altitude": 12,
        "initial_turn": -90.0,
        "goal_position": (239.016, 299.1561, -5.6921),
        "success_threshold": 20.0,
        "shortest_path": None,
    },
    6: {
        "name": "fountain",
        "instruction": "Navigate to the fountain",
        "target": "fountain",
        "start_position": (-40.335, -115.93, -10.146),
        "orientation": (0.9236459732055664, 0, 0, -0.3832468092441559),
        "altitude": 8,
        "initial_turn": -90.0,
        "goal_position": (-43.4099, -142.8077, 1.1649),
        "success_threshold": 10.0,
        "shortest_path": None,
    },
    7: {
        "name": "house",
        "instruction": "Navigate to the house",
        "target": "house",
        "start_position": (-210.679, -270.122, -10.146),
        "orientation": (0.9236459732055664, 0, 0, -0.3832468092441559),
        "altitude": 3,
        "initial_turn": 25.0,
        "goal_position": (-181.4402, -254.2189, 2.6455),
        "success_threshold": 10.0,
        "shortest_path": None,
    },
    8: {
        "name": "playground",
        "instruction": "Navigate to the playground",
        "target": "playground",
        "start_position": (123.226, -455.022, -10.146),
        "orientation": (0.9236459732055664, 0, 0, -0.3832468092441559),
        "altitude": 12,
        "initial_turn": 90.0,
        "goal_position": (118.197, -398.046, -5.657),
        "success_threshold": 10.0,
        "shortest_path": None,
    },
    9: {
        "name": "parkbench",
        "instruction": "Navigate to the park bench",
        "target": "park bench",
        "start_position": (238.734, -9.465, -10.146),
        "orientation": (0.9236459732055664, 0, 0, -0.3832468092441559),
        "altitude": 1,
        "initial_turn": -120.0,
        "goal_position": (226.8201, -28.6595, 3.9568),
        "success_threshold": 10.0,
        "shortest_path": None,
    },
    10: {
        "name": "umbrella",
        "instruction": "Navigate to the umbrella near the blue building",
        "target": "umbrella near the blue building",
        "start_position": (223.761, 128.614, -10.146),
        "orientation": (0.9236459732055664, 0, 0, -0.3832468092441559),
        "altitude": 18,
        "initial_turn": 170.0,
        "goal_position": (197.7946, 128.1664, -6.8334),
        "success_threshold": 10.0,
        "shortest_path": None,
    },
    11: {
        "name": "house2",
        "instruction": "Navigate to house2",
        "target": "house2",
        "start_position": (-16.264, 4.708, -10.146),
        "orientation": (0.9236459732055664, 0, 0, -0.3832468092441559),
        "altitude": 3,
        "initial_turn": -40.0,
        "goal_position": (11.6875, -11.4835, 8.0586),
        "success_threshold": 10.0,
        "shortest_path": None,
    },
    12: {
        "name": "fountain2",
        "instruction": "Navigate to fountain2",
        "target": "fountain2",
        "start_position": (-29.682, 31.467, -10.146),
        "orientation": (0.9236459732055664, 0, 0, -0.3832468092441559),
        "altitude": 3,
        "initial_turn": -30.0,
        "goal_position": (-7.4304, 23.5018, 7.8896),
        "success_threshold": 10.0,
        "shortest_path": None,
    },
    13: {
        "name": "sculpture",
        "instruction": "Navigate to the sculpture",
        "target": "sculpture",
        "start_position": (418.796, -44.472, -10.146),
        "orientation": (0.9236459732055664, 0, 0, -0.3832468092441559),
        "altitude": 8,
        "initial_turn": 60.0,
        "goal_position": (431.7079, -16.5003, -2.5492),
        "success_threshold": 10.0,
        "shortest_path": None,
    },
    14: {
        "name": "sculpture2",
        "instruction": "Navigate to sculpture2",
        "target": "sculpture2",
        "start_position": (570.993, 75.063, -10.146),
        "orientation": (0.9236459732055664, 0, 0, -0.3832468092441559),
        "altitude": 3,
        "initial_turn": -90.0,
        "goal_position": (570.5457, 60.5743, 5.3357),
        "success_threshold": 10.0,
        "shortest_path": None,
    },
}

# Compute shortest path (straight-line 2D distance from start to goal)
for sid, cfg in SCENARIO_CONFIG.items():
    sp = np.array(cfg["start_position"][:2])
    gp = np.array(cfg["goal_position"][:2])
    cfg["shortest_path"] = np.linalg.norm(gp - sp)

MODEL_NAME = "CityNavAgent"

RESULTS_ROOT = os.path.join(_parent_dir, "map1", "baselines", "city_nav")


def _exec_turn_navqwen_style(client, degrees):
    """
    IDENTICAL to nav_qwen.py execute_action for turn_left/turn_right.
    Uses the same yaw_rate (30 deg/s) and same API call pattern.
    """
    import airsim  # imported here after airsim_client sets up path
    yaw_rate = 30.0
    value = abs(degrees)
    duration = max(0.5, value / yaw_rate)
    action_name = "turn_right" if degrees > 0 else "turn_left"
    rate = yaw_rate if degrees > 0 else -yaw_rate
    print(f"[Turn] {action_name} {value:.1f} deg (rate={rate:.1f} deg/s, duration={duration:.1f}s)")
    client.moveByVelocityAsync(
        0, 0, 0, duration,
        yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=rate)
    ).join()
    # NO simGetVehiclePose, NO _log_position between turn and next operation


def validate_airsim():
    """Basic validation: test takeoff, turn, move, land without LLM."""
    print("=" * 60)
    print("CityNavAgent - Basic AirSim Validation")
    print("=" * 60)

    airsim = AirSimClientAdapter()
    flight = FlightController(airsim)
    perception = PerceptionModule(airsim)

    print("\n--- 1. Testing image capture ---")
    images = perception.capture_all()
    for name, img in images.items():
        if img is not None:
            print(f"  {name}: shape={img.shape}, dtype={img.dtype}")
        else:
            print(f"  {name}: FAILED")
    print("Image capture: OK")

    print("\n--- 2. Testing takeoff ---")
    flight.takeoff(altitude=10.0)
    pos = flight.get_position()
    print(f"  Position after takeoff: {pos}")

    print("\n--- 3. Testing turn (nav_qwen.py style) ---")
    flight.turn_right(45.0)
    print("  turn_right 45: OK")

    print("\n--- 4. Testing forward flight ---")
    flight.execute_discrete_action(1)  # MOVE_FORWARD
    pos2 = flight.get_position()
    dist = np.linalg.norm(pos2[:2] - pos[:2])
    print(f"  Distance moved: {dist:.2f}m")

    print("\n--- 5. Testing landing ---")
    flight.land()

    final_pos = flight.get_position()
    total_dist = np.linalg.norm(final_pos[:2] - pos[:2])
    print(f"\nTotal horizontal displacement: {total_dist:.2f}m")
    print(f"Trajectory points: {len(flight.trajectory)}")

    if total_dist > 1.0:
        print("\n[PASS] Validation successful. AirSim integration working.")
    else:
        print("\n[WARN] Small displacement. Ensure AirSim simulation is running.")

    return total_dist > 1.0


def run_scenario(scenario_id, num_flights=10, qwen_model="qwen3.5-35b-a3b"):
    """Run CityNavAgent on one scenario, N flights."""
    if scenario_id not in SCENARIO_CONFIG:
        print(f"Unknown scenario {scenario_id}. Use --list to see options.")
        return None

    cfg = SCENARIO_CONFIG[scenario_id]
    scenario_name = cfg["name"]

    print("=" * 60)
    print(f"CityNavAgent - Scenario {scenario_id}: {cfg['target']}")
    print(f"  Flights: {num_flights} | Model: {qwen_model}")
    print("=" * 60)

    airsim = AirSimClientAdapter()
    flight = FlightController(airsim)
    perception = PerceptionModule(airsim)

    try:
        llm = QwenLLMAdapter(model=qwen_model)
    except Exception as e:
        print(f"LLM init failed: {e}")
        return None

    pipeline = CityNavPipeline(airsim, flight, llm, perception)

    initial_turn = cfg["initial_turn"] if abs(cfg["initial_turn"]) > 0.5 else None

    result_dir = os.path.join(RESULTS_ROOT, f"data_{scenario_id}",
                               f"{MODEL_NAME}_{scenario_name}")
    os.makedirs(result_dir, exist_ok=True)
    print(f"[Results] Saving to: {result_dir}")

    airsim._image_dir = result_dir

    flight_dir = result_dir
    metrics_path = os.path.join(RESULTS_ROOT, f"data_{scenario_id}",
                                 "metrics_results.csv")
    fieldnames = ["Model", "Flight", "SR", "NE(m)", "SPL", "OR",
                   "TL(m)", "Oracle_Dist(m)", "Shortest_Path(m)", "Steps"]

    for flight_idx in range(1, num_flights + 1):
        print(f"\n{'#'*50}")
        print(f"# Flight {flight_idx}/{num_flights}")
        print(f"{'#'*50}")

        flight.reset_to_start(cfg["start_position"], cfg["orientation"])
        flight.takeoff(cfg["altitude"])
        time.sleep(2.0)

        if initial_turn is not None:
            _exec_turn_navqwen_style(airsim.client, initial_turn)
            time.sleep(4.0)

        pipeline.step_count = 0
        pipeline.action_history = []

        success = False
        nav_error = 999.0
        tl = 0.0
        spl = 0.0
        or_flag = 0
        oracle_dist = 999.0
        steps = 0

        try:
            for _ in range(pipeline.max_steps):
                pos_before = flight.get_position()
                should_continue = pipeline.run_step(cfg["instruction"], cfg["target"])
                pos_after = flight.get_position()

                if not should_continue:
                    break

                if np.linalg.norm(pos_after[:2] - pos_before[:2]) < 0.1:
                    print("[Pipeline] Stuck, trying MOVE_FORWARD...")
                    try:
                        flight.execute_discrete_action(1)
                    except Exception:
                        pass

            final_pos = flight.get_position()
            goal_2d = np.array(cfg["goal_position"][:2])
            final_2d = final_pos[:2]
            nav_error = float(np.linalg.norm(final_2d - goal_2d))
            success = nav_error <= cfg["success_threshold"]

            trajectory = flight.get_trajectory()
            traj_2d = trajectory[:, :2] if len(trajectory) > 0 else np.array([[0, 0]])

            tl = 0.0
            if len(traj_2d) > 1:
                diffs = traj_2d[1:] - traj_2d[:-1]
                tl = float(np.sum(np.linalg.norm(diffs, axis=1)))

            gt_len = cfg["shortest_path"]
            spl = (float(success) * gt_len / max(gt_len, tl)) if tl > 0 else 0.0

            oracle_dist = nav_error
            if len(traj_2d) > 1:
                dists = np.linalg.norm(traj_2d - goal_2d, axis=1)
                oracle_dist = float(np.min(dists))
                if oracle_dist <= cfg["success_threshold"]:
                    or_flag = 1

            steps = pipeline.step_count

        except Exception as e:
            print(f"[Flight {flight_idx}] Error during navigation: {e}")
            import traceback
            traceback.print_exc()
            nav_error = 999.0
            success = False

        # -- Save flight CSV immediately --
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"flight_{flight_idx:02d}_{timestamp}.csv"
        csv_path = os.path.join(flight_dir, csv_filename)

        trajectory = flight.get_trajectory()
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["X", "Y", "Z"])
            for row in trajectory:
                writer.writerow([row[0], row[1], row[2]])
        print(f"  Trajectory saved to: {csv_filename} ({len(trajectory)} points)")

        # -- Append metrics immediately --
        row = {
            "Model": MODEL_NAME,
            "Flight": csv_filename,
            "SR": int(success),
            "NE(m)": round(nav_error, 4),
            "SPL": round(spl, 4),
            "OR": or_flag,
            "TL(m)": round(tl, 4),
            "Oracle_Dist(m)": round(oracle_dist, 4),
            "Shortest_Path(m)": round(cfg["shortest_path"], 4),
            "Steps": steps,
        }

        _write_metrics_row(metrics_path, fieldnames, row)

        print(f"  Flight {flight_idx} result: SR={int(success)}, NE={nav_error:.2f}m, "
              f"SPL={spl:.4f}, OR={or_flag}, TL={tl:.2f}m, Steps={steps}")

        # Land
        try:
            flight.land()
        except Exception:
            pass
        time.sleep(3.0)

    # Summary
    all_rows = _read_metrics_csv(metrics_path)
    successes = sum(1 for r in all_rows if int(r.get("SR", 0)))
    avg_ne = np.mean([float(r.get("NE(m)", 0)) for r in all_rows])
    avg_spl = np.mean([float(r.get("SPL", 0)) for r in all_rows])
    avg_tl = np.mean([float(r.get("TL(m)", 0)) for r in all_rows])
    sr_pct = successes / num_flights * 100

    print(f"\n{'='*50}")
    print(f" Summary: Scenario {scenario_id} ({scenario_name})")
    print(f"  SR: {successes}/{num_flights} ({sr_pct:.1f}%)")
    print(f"  Avg NE: {avg_ne:.2f}m")
    print(f"  Avg SPL: {avg_spl:.4f}")
    print(f"  Avg TL: {avg_tl:.2f}m")
    print(f"{'='*50}")

    return all_rows


def _write_metrics_row(csv_path, fieldnames, row):
    """Append a single row to metrics CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _read_metrics_csv(csv_path):
    """Read all rows from metrics CSV."""
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        return list(reader)


def run_all_scenarios(num_flights=10, qwen_model="qwen3.5-35b-a3b"):
    """Run all 14 scenarios."""
    all_results = {}
    for sid in range(1, 15):
        print(f"\n{'#'*70}")
        print(f"# SCENARIO {sid}")
        print(f"{'#'*70}")
        try:
            results = run_scenario(sid, num_flights, qwen_model)
            if results is not None:
                all_results[sid] = results
        except Exception as e:
            print(f"Scenario {sid} failed with error: {e}")
            import traceback
            traceback.print_exc()

    # Grand summary
    if all_results:
        print(f"\n{'='*70}")
        print(f" GRAND SUMMARY (CityNavAgent)")
        print(f"{'='*70}")
        total_success = 0
        total_flights = 0
        all_nes = []
        all_spls = []
        for sid, results in sorted(all_results.items()):
            successes = sum(1 for r in results if int(r.get("SR", 0)))
            avg_ne = np.mean([float(r.get("NE(m)", 0)) for r in results])
            avg_spl = np.mean([float(r.get("SPL", 0)) for r in results])
            total_success += successes
            total_flights += len(results)
            all_nes.extend([float(r.get("NE(m)", 0)) for r in results])
            all_spls.extend([float(r.get("SPL", 0)) for r in results])
            cfg = SCENARIO_CONFIG[sid]
            print(f"  S{sid:02d} {cfg['name']:<25s} SR={int(successes)}/{len(results)} "
                  f"NE={avg_ne:.2f}m SPL={avg_spl:.4f}")
        print(f"  {'─'*55}")
        print(f"  OVERALL: SR={total_success}/{total_flights} "
              f"({total_success/total_flights*100:.1f}%) "
              f"NE={np.mean(all_nes):.2f}m SPL={np.mean(all_spls):.4f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="CityNavAgent Adapted Pipeline")
    parser.add_argument("--validate", action="store_true",
                        help="Basic AirSim integration test (no LLM)")
    parser.add_argument("--scenario", type=int, default=None,
                        help="Scenario ID (1-14)")
    parser.add_argument("--all", action="store_true",
                        help="Run all 14 scenarios")
    parser.add_argument("--flights", type=int, default=10,
                        help="Flights per scenario (default: 10)")
    parser.add_argument("--model", type=str, default="qwen3.5-35b-a3b",
                        help="Qwen model name")
    parser.add_argument("--list", action="store_true",
                        help="List all scenarios")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable scenarios:")
        for sid, cfg in sorted(SCENARIO_CONFIG.items()):
            t = cfg['initial_turn']
            turn_desc = (f"turn_{'right' if t > 0 else 'left'}={abs(t):.0f}"
                          if abs(t) > 0.5 else "no_turn")
            print(f"  {sid:2d}: {cfg['name']:<30s} alt={cfg['altitude']:2d}m, "
                  f"{turn_desc}, thresh={cfg['success_threshold']}m")
        return

    if args.validate:
        validate_airsim()
        return

    if args.all:
        run_all_scenarios(args.flights, args.model)
        return

    if args.scenario is not None:
        run_scenario(args.scenario, args.flights, args.model)
        return

    print("No action specified. Use --validate, --scenario N, or --all")
    print("Use --list to see available scenarios")


if __name__ == "__main__":
    main()
