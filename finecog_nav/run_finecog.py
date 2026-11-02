"""
run_finecog.py - FineCog-Nav Full Experiment Runner
===================================================
Runs the adapted FineCog-Nav pipeline on all 14 scenarios, 10 flights each.

Usage:
    python run_finecog.py --scenario 5 --flights 1    # Single test
    python run_finecog.py --all                        # All 14 scenarios
    python run_finecog.py --list                       # List scenarios
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

from finecog_nav.adapted.airsim_client import AirSimClientAdapter
from finecog_nav.adapted.llm_client import FineCogLLM
from finecog_nav.adapted.perception_module import PerceptionModule
from finecog_nav.adapted.flight_controller import FlightController
from finecog_nav.adapted.collision_checker import CollisionChecker
from finecog_nav.adapted.finecog_modules import FineCogModules
from finecog_nav.adapted.finecog_pipeline import FineCogPipeline

# ============================================================
#  Scenario Config (same as evaluate_nav_metrics.py)
# ============================================================
SCENARIO_CONFIG = {
    1: {"name": "orangewhite_umbrella", "instruction": "Navigate to the orange and white striped umbrella", "target": "orange and white striped umbrella", "start_position": (-253.011, -115.011, -10.146), "orientation": (0.9236459732055664, 0, 0, -0.3832468092441559), "altitude": 1, "initial_turn": 160.0, "goal_position": (-294.1103, -101.224, 2.3967), "success_threshold": 10.0},
    2: {"name": "statue", "instruction": "Navigate to the statue of a person", "target": "statue of a person", "start_position": (-140.042, -412.695, -10.146), "orientation": (0.9236459732055664, 0, 0, -0.3832468092441559), "altitude": 5, "initial_turn": 0.0, "goal_position": (-120.4851, -415.2262, -0.5565), "success_threshold": 10.0},
    3: {"name": "restaurant_across_river", "instruction": "Navigate to the restaurant across the river", "target": "restaurant across the river", "start_position": (-63.832, -231.185, -10.146), "orientation": (0.9236459732055664, 0, 0, -0.3832468092441559), "altitude": 8, "initial_turn": 70.0, "goal_position": (-52.2433, -182.3261, 3.4642), "success_threshold": 15.0},
    4: {"name": "hotel_front_door", "instruction": "Navigate to the hotel front door", "target": "hotel front door", "start_position": (-73.259, 247.858, -10.146), "orientation": (0.9236459732055664, 0, 0, -0.3832468092441559), "altitude": 8, "initial_turn": 0.0, "goal_position": (-22.9098, 260.5971, 0.4993), "success_threshold": 10.0},
    5: {"name": "blue_amusement_park", "instruction": "Navigate to the blue amusement park", "target": "blue amusement park", "start_position": (234.382, 354.169, -10.146), "orientation": (0.9236459732055664, 0, 0, -0.3832468092441559), "altitude": 12, "initial_turn": -90.0, "goal_position": (239.016, 299.1561, -5.6921), "success_threshold": 20.0},
    6: {"name": "fountain", "instruction": "Navigate to the fountain", "target": "fountain", "start_position": (-40.335, -115.93, -10.146), "orientation": (0.9236459732055664, 0, 0, -0.3832468092441559), "altitude": 8, "initial_turn": -90.0, "goal_position": (-43.4099, -142.8077, 1.1649), "success_threshold": 10.0},
    7: {"name": "house", "instruction": "Navigate to the house", "target": "house", "start_position": (-210.679, -270.122, -10.146), "orientation": (0.9236459732055664, 0, 0, -0.3832468092441559), "altitude": 3, "initial_turn": 25.0, "goal_position": (-181.4402, -254.2189, 2.6455), "success_threshold": 10.0},
    8: {"name": "playground", "instruction": "Navigate to the playground", "target": "playground", "start_position": (123.226, -455.022, -10.146), "orientation": (0.9236459732055664, 0, 0, -0.3832468092441559), "altitude": 12, "initial_turn": 90.0, "goal_position": (118.197, -398.046, -5.657), "success_threshold": 10.0},
    9: {"name": "parkbench", "instruction": "Navigate to the park bench", "target": "park bench", "start_position": (238.734, -9.465, -10.146), "orientation": (0.9236459732055664, 0, 0, -0.3832468092441559), "altitude": 1, "initial_turn": -120.0, "goal_position": (226.8201, -28.6595, 3.9568), "success_threshold": 10.0},
    10: {"name": "umbrella", "instruction": "Navigate to the umbrella near the blue building", "target": "umbrella near the blue building", "start_position": (223.761, 128.614, -10.146), "orientation": (0.9236459732055664, 0, 0, -0.3832468092441559), "altitude": 18, "initial_turn": 170.0, "goal_position": (197.7946, 128.1664, -6.8334), "success_threshold": 10.0},
    11: {"name": "house2", "instruction": "Navigate to house2", "target": "house2", "start_position": (-16.264, 4.708, -10.146), "orientation": (0.9236459732055664, 0, 0, -0.3832468092441559), "altitude": 3, "initial_turn": -40.0, "goal_position": (11.6875, -11.4835, 8.0586), "success_threshold": 10.0},
    12: {"name": "fountain2", "instruction": "Navigate to fountain2", "target": "fountain2", "start_position": (-29.682, 31.467, -10.146), "orientation": (0.9236459732055664, 0, 0, -0.3832468092441559), "altitude": 3, "initial_turn": -30.0, "goal_position": (-7.4304, 23.5018, 7.8896), "success_threshold": 10.0},
    13: {"name": "sculpture", "instruction": "Navigate to the sculpture", "target": "sculpture", "start_position": (418.796, -44.472, -10.146), "orientation": (0.9236459732055664, 0, 0, -0.3832468092441559), "altitude": 8, "initial_turn": 60.0, "goal_position": (431.7079, -16.5003, -2.5492), "success_threshold": 10.0},
    14: {"name": "sculpture2", "instruction": "Navigate to sculpture2", "target": "sculpture2", "start_position": (570.993, 75.063, -10.146), "orientation": (0.9236459732055664, 0, 0, -0.3832468092441559), "altitude": 3, "initial_turn": -90.0, "goal_position": (570.5457, 60.5743, 5.3357), "success_threshold": 10.0},
}

for sid, cfg in SCENARIO_CONFIG.items():
    sp = np.array(cfg["start_position"][:2])
    gp = np.array(cfg["goal_position"][:2])
    cfg["shortest_path"] = np.linalg.norm(gp - sp)

MODEL_NAME = "FineCogNav"
RESULTS_ROOT = os.path.join(_parent_dir, "map1", "baselines", "finecog_nav")


def run_scenario(scenario_id, num_flights=10):
    cfg = SCENARIO_CONFIG[scenario_id]
    sn = cfg["name"]
    print(f"\nFineCog-Nav Scenario {scenario_id}: {cfg['target']}")
    print(f"  Flights: {num_flights}")

    airsim = AirSimClientAdapter()
    flight = FlightController(airsim)
    perception = PerceptionModule(airsim)
    llm = FineCogLLM()
    collision = CollisionChecker()

    initial_turn = cfg["initial_turn"] if abs(cfg["initial_turn"]) > 0.5 else None
    result_dir = os.path.join(RESULTS_ROOT, f"data_{scenario_id}", f"{MODEL_NAME}_{sn}")
    os.makedirs(result_dir, exist_ok=True)
    airsim._image_dir = result_dir
    metrics_path = os.path.join(RESULTS_ROOT, f"data_{scenario_id}", "metrics_results.csv")
    fieldnames = ["Model", "Flight", "SR", "NE(m)", "SPL", "OR", "TL(m)", "Oracle_Dist(m)", "Shortest_Path(m)", "Steps"]

    for fi in range(1, num_flights + 1):
        print(f"\n### Flight {fi}/{num_flights} ###")
        flight.reset_to_start(cfg["start_position"], cfg["orientation"])
        flight.takeoff(cfg["altitude"]); time.sleep(2.0)
        if initial_turn:
            import airsim as _as
            yaw_rate=30.0; dur=max(0.5,abs(initial_turn)/yaw_rate); rate=yaw_rate if initial_turn>0 else -yaw_rate
            flight.client.moveByVelocityAsync(0,0,0,dur,yaw_mode=_as.YawMode(is_rate=True,yaw_or_rate=rate)).join()
            time.sleep(4.0)

        modules = FineCogModules(llm, flight, collision, airsim)
        pipeline = FineCogPipeline(modules, flight, airsim, perception)

        try:
            result = pipeline.run_navigation_full(cfg["instruction"],
                cfg["goal_position"], cfg["success_threshold"])
            nav_error = result["nav_error"]; success = result["success"]
            trajectory = result["trajectory"]
        except Exception as e:
            print(f"Error: {e}"); import traceback; traceback.print_exc()
            nav_error = 999.0; success = False
            trajectory = flight.get_trajectory()

        traj_2d = trajectory[:,:2] if len(trajectory)>0 else np.array([[0,0]])
        tl = float(sum(np.linalg.norm(traj_2d[1:]-traj_2d[:-1],axis=1))) if len(traj_2d)>1 else 0.0
        gt_len = cfg["shortest_path"]
        spl = (float(success)*gt_len/max(gt_len,tl)) if tl>0 else 0.0
        oracle_dist = nav_error
        or_flag = 0
        if len(traj_2d)>1:
            dists = np.linalg.norm(traj_2d-np.array(cfg["goal_position"][:2]),axis=1)
            oracle_dist=float(np.min(dists))
            if oracle_dist<=cfg["success_threshold"]: or_flag=1
        steps = pipeline.step_count

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_fn = f"flight_{fi:02d}_{ts}.csv"
        csv_fp = os.path.join(result_dir, csv_fn)
        with open(csv_fp,'w',newline='') as f:
            w=csv.writer(f); w.writerow(["X","Y","Z"])
            for r in trajectory: w.writerow([r[0],r[1],r[2]])

        row = {"Model":MODEL_NAME,"Flight":csv_fn,"SR":int(success),"NE(m)":round(nav_error,4),
               "SPL":round(spl,4),"OR":or_flag,"TL(m)":round(tl,4),
               "Oracle_Dist(m)":round(oracle_dist,4),"Shortest_Path(m)":round(gt_len,4),"Steps":steps}
        _write_metrics_row(metrics_path, fieldnames, row)
        print(f"  SR={int(success)} NE={nav_error:.2f}m SPL={spl:.4f} TL={tl:.2f}m Steps={steps}")

        try: flight.land()
        except: pass
        time.sleep(3.0)

    rows = _read_metrics_csv(metrics_path)
    suc = sum(1 for r in rows if int(r.get("SR",0)))
    print(f"\n  Scenario {scenario_id}: SR={suc}/{num_flights} ({suc/num_flights*100:.1f}%)")
    return rows


def _write_metrics_row(csv_path, fieldnames, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fe = os.path.exists(csv_path)
    with open(csv_path,'a',newline='') as f:
        w = csv.DictWriter(f,fieldnames=fieldnames)
        if not fe: w.writeheader()
        w.writerow(row)


def _read_metrics_csv(csv_path):
    if not os.path.exists(csv_path): return []
    with open(csv_path,'r',newline='') as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario",type=int); p.add_argument("--all",action="store_true")
    p.add_argument("--flights",type=int,default=10); p.add_argument("--list",action="store_true")
    a = p.parse_args()
    if a.list:
        for sid in sorted(SCENARIO_CONFIG):
            c=SCENARIO_CONFIG[sid]; t=c['initial_turn']
            td=f"turn_{'right'if t>0 else'left'}={abs(t):.0f}" if abs(t)>0.5 else "no_turn"
            print(f"  {sid:2d}: {c['name']:<30s} alt={c['altitude']:2d}m {td}")
        return
    if a.all:
        for sid in range(1,15): run_scenario(sid, a.flights)
        return
    if a.scenario: run_scenario(a.scenario, a.flights)


if __name__ == "__main__":
    main()
