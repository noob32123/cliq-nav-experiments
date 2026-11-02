# -*- coding: utf-8 -*-
"""
CityNavAgent 指标评估脚本
=========================
对 map1/baselines/city_nav/data_1 ~ data_14 中 CityNavAgent 的飞行轨迹进行计算，
指标定义与计算方式与 evaluate_nav_metrics.py 完全一致。

评估指标：
  - SR   (Success Rate)           : 成功率，终点距目标 ≤ 阈值
  - NE   (Navigation Error)       : 导航误差，终点到目标的 2D 欧氏距离
  - OR   (Oracle Success Rate)    : 轨迹中任意点距目标 ≤ 阈值
  - SPL  (Success weighted by PL) : S_i * L_i / max(P_i, L_i)
  - TL   (Trajectory Length)      : 实际飞行路径长度 (2D)

输出：
  - per_scenario_metrics.csv      : 每场景详细指标
  - overall_metrics_summary.csv   : 全局聚合指标
"""

import os
import argparse
from pathlib import Path
import csv
import math

# ============================================================
#  配置 —— 与 evaluate_nav_metrics.py 完全相同
# ============================================================
USE_3D_DISTANCE = False

SCENARIO_CONFIG = {
    "data_1": {
        "name": "orangewhite_umbrella",
        "start_position": (-253.011, -115.011, -10.146),
        "goal_position": (-294.1103, -101.224, 2.3967),
        "success_threshold": 10.0,
    },
    "data_2": {
        "name": "statue",
        "start_position": (-140.042, -412.695, -10.146),
        "goal_position": (-120.4851, -415.2262, -0.5565),
        "success_threshold": 10.0,
    },
    "data_3": {
        "name": "restaurant_across_river",
        "start_position": (-63.832, -231.185, -10.146),
        "goal_position": (-52.2433, -182.3261, 3.4642),
        "success_threshold": 15.0,
    },
    "data_4": {
        "name": "hotel_front_door",
        "start_position": (-73.259, 247.858, -10.146),
        "goal_position": (-22.9098, 260.5971, 0.4993),
        "success_threshold": 10.0,
    },
    "data_5": {
        "name": "blue_amusement_park",
        "start_position": (234.382, 354.169, -10.146),
        "goal_position": (239.016, 299.1561, -5.6921),
        "success_threshold": 20.0,
    },
    "data_6": {
        "name": "fountain",
        "start_position": (-40.335, -115.93, -10.146),
        "goal_position": (-43.4099, -142.8077, 1.1649),
        "success_threshold": 10.0,
    },
    "data_7": {
        "name": "house",
        "start_position": (-210.679, -270.122, -10.146),
        "goal_position": (-181.4402, -254.2189, 2.6455),
        "success_threshold": 10.0,
    },
    "data_8": {
        "name": "playground",
        "start_position": (123.226, -455.022, -10.146),
        "goal_position": (118.197, -398.046, -5.657),
        "success_threshold": 10.0,
    },
    "data_9": {
        "name": "parkbench",
        "start_position": (238.734, -9.465, -10.146),
        "goal_position": (226.8201, -28.6595, 3.9568),
        "success_threshold": 10.0,
    },
    "data_10": {
        "name": "umbrella",
        "start_position": (223.761, 128.614, -10.146),
        "goal_position": (197.7946, 128.1664, -6.8334),
        "success_threshold": 10.0,
    },
    "data_11": {
        "name": "house2",
        "start_position": (-16.264, 4.708, -10.146),
        "goal_position": (11.6875, -11.4835, 8.0586),
        "success_threshold": 10.0,
    },
    "data_12": {
        "name": "fountain2",
        "start_position": (-29.682, 31.467, -10.146),
        "goal_position": (-7.4304, 23.5018, 7.8896),
        "success_threshold": 10.0,
    },
    "data_13": {
        "name": "sculpture",
        "start_position": (418.796, -44.472, -10.146),
        "goal_position": (431.7079, -16.5003, -2.5492),
        "success_threshold": 10.0,
    },
    "data_14": {
        "name": "sculpture2",
        "start_position": (570.993, 75.063, -10.146),
        "goal_position": (570.5457, 60.5743, 5.3357),
        "success_threshold": 10.0,
    },
}

MODEL_NAME = "CityNavAgent"

# ============================================================
#  工具函数 —— 与 evaluate_nav_metrics.py 完全相同
# ============================================================

def euclidean_2d(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def load_trajectory(csv_path):
    trajectory = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row['X'])
                y = float(row['Y'])
                z = float(row['Z'])
                trajectory.append((x, y, z))
            except (ValueError, KeyError):
                continue
    return trajectory


def compute_path_length(trajectory):
    if len(trajectory) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(trajectory)):
        total += euclidean_2d(trajectory[i - 1], trajectory[i])
    return total


def compute_navigation_error(trajectory, goal):
    if not trajectory:
        return float('inf')
    return euclidean_2d(trajectory[-1], goal)


def compute_oracle_distance(trajectory, goal):
    if not trajectory:
        return float('inf')
    return min(euclidean_2d(p, goal) for p in trajectory)


def compute_episode_metrics(trajectory, start, goal, threshold):
    ne = compute_navigation_error(trajectory, goal)
    tl = compute_path_length(trajectory)
    oracle_dist = compute_oracle_distance(trajectory, goal)

    success = 1 if ne <= threshold else 0
    oracle_success = 1 if oracle_dist <= threshold else 0

    shortest_path = euclidean_2d(start, goal)

    if success:
        spl = shortest_path / max(tl, shortest_path) if max(tl, shortest_path) > 0 else 0.0
    else:
        spl = 0.0

    return {
        "NE": ne,
        "TL": tl,
        "SR": success,
        "OR": oracle_success,
        "SPL": spl,
        "oracle_dist": oracle_dist,
        "shortest_path": shortest_path,
        "num_steps": len(trajectory) - 1 if len(trajectory) > 1 else 0,
    }


# ============================================================
#  主处理逻辑
# ============================================================

def process_all(root_dir, output_dir=None):
    summary_dir = Path(output_dir) if output_dir is not None else Path(root_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    scenario_rows = []
    all_episodes = []

    for data_key in sorted(SCENARIO_CONFIG.keys(), key=lambda k: int(k.split('_')[1])):
        cfg = SCENARIO_CONFIG[data_key]
        scenario_name = cfg["name"]
        start = cfg["start_position"]
        goal = cfg["goal_position"]
        threshold = cfg["success_threshold"]

        data_dir = os.path.join(root_dir, data_key)
        if not os.path.isdir(data_dir):
            continue

        model_dir = os.path.join(data_dir, f"{MODEL_NAME}_{scenario_name}")
        if not os.path.isdir(model_dir):
            continue

        csv_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.csv')])
        if not csv_files:
            continue

        episodes = []
        for csv_file in csv_files:
            csv_path = os.path.join(model_dir, csv_file)
            traj = load_trajectory(csv_path)
            if not traj:
                continue
            metrics = compute_episode_metrics(traj, start, goal, threshold)
            metrics["flight"] = csv_file
            episodes.append(metrics)
            all_episodes.append(metrics)

        if not episodes:
            continue

        n = len(episodes)
        sr = sum(e["SR"] for e in episodes) / n * 100
        ne = sum(e["NE"] for e in episodes) / n
        spl = sum(e["SPL"] for e in episodes) / n * 100
        or_rate = sum(e["OR"] for e in episodes) / n * 100
        tl = sum(e["TL"] for e in episodes) / n

        print(f"  {data_key} ({cfg['name']:<25s})  "
              f"n={n}  SR={sr:5.1f}%  NE={ne:6.2f}m  "
              f"SPL={spl:5.1f}%  OR={or_rate:5.1f}%  TL={tl:7.2f}m")

        scenario_rows.append({
            "Scenario": data_key,
            "Target": cfg["name"],
            "Model": MODEL_NAME,
            "Num_Flights": n,
            "SR(%)": f"{sr:.1f}",
            "NE(m)": f"{ne:.4f}",
            "SPL(%)": f"{spl:.1f}",
            "OR(%)": f"{or_rate:.1f}",
            "TL(m)": f"{tl:.4f}",
        })

    # ── per_scenario_metrics.csv ──
    per_csv = os.path.join(summary_dir, "per_scenario_metrics.csv")
    with open(per_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["Scenario", "Target", "Model", "Num_Flights",
                      "SR(%)", "NE(m)", "SPL(%)", "OR(%)", "TL(m)"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scenario_rows)
    print(f"\nper_scenario_metrics -> {per_csv}")

    # ── overall_metrics_summary.csv ──
    if not all_episodes:
        print("No episodes found.")
        return

    total = len(all_episodes)
    overall_sr = sum(e["SR"] for e in all_episodes) / total * 100
    overall_ne = sum(e["NE"] for e in all_episodes) / total
    overall_spl = sum(e["SPL"] for e in all_episodes) / total * 100
    overall_or = sum(e["OR"] for e in all_episodes) / total * 100
    overall_tl = sum(e["TL"] for e in all_episodes) / total

    overall_csv = os.path.join(summary_dir, "overall_metrics_summary.csv")
    with open(overall_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Model", "Num_Episodes", "SR(%)", "NE(m)", "SPL(%)", "OR(%)", "TL(m)"
        ])
        writer.writeheader()
        writer.writerow({
            "Model": MODEL_NAME,
            "Num_Episodes": total,
            "SR(%)": f"{overall_sr:.1f}",
            "NE(m)": f"{overall_ne:.4f}",
            "SPL(%)": f"{overall_spl:.1f}",
            "OR(%)": f"{overall_or:.1f}",
            "TL(m)": f"{overall_tl:.4f}",
        })
    print(f"overall_metrics_summary -> {overall_csv}")

    print(f"\n{'='*65}")
    print(f"  CityNavAgent (all {total} flights across {len(scenario_rows)} scenarios)")
    print(f"  SR={overall_sr:.1f}%  NE={overall_ne:.2f}m  SPL={overall_spl:.1f}%  "
          f"OR={overall_or:.1f}%  TL={overall_tl:.2f}m")
    print(f"{'='*65}\n")

    # ── 校验 config 正确性 ──
    print("Config validation:")
    print(f"  {'Scene':<10} {'start':<22} {'goal':<22} {'thresh':>6}  {'shortest':>8}")
    print(f"  {'─'*10} {'─'*22} {'─'*22} {'─'*6}  {'─'*8}")
    for data_key in sorted(SCENARIO_CONFIG.keys(), key=lambda k: int(k.split('_')[1])):
        cfg = SCENARIO_CONFIG[data_key]
        s2d = (cfg["start_position"][0], cfg["start_position"][1])
        g2d = (cfg["goal_position"][0], cfg["goal_position"][1])
        sp = euclidean_2d(s2d, g2d)
        print(f"  {data_key:<10} ({s2d[0]:>7.1f},{s2d[1]:>7.1f})  "
              f"({g2d[0]:>7.1f},{g2d[1]:>7.1f})  "
              f"{cfg['success_threshold']:>5.0f}m  {sp:>7.2f}m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline CityNavAgent map1 metrics")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "baselines/city_nav")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.data_root
    print(f"Data root: {root}\n")
    process_all(root, args.output_dir)
