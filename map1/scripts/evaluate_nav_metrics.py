# -*- coding: utf-8 -*-
"""
UAV视觉导航指标评估脚本
=======================
对 data_1 ~ data_14 中所有模型文件夹的飞行轨迹数据进行处理，
计算常见无人机视觉导航指标：
  - SR   (Success Rate)           : 成功率，终点距目标 ≤ 阈值即为成功
  - NE   (Navigation Error)       : 导航误差，终点到目标的欧式距离
  - OR   (Oracle Success Rate)    : 轨迹中任意点距目标 ≤ 阈值即为成功
  - SPL  (Success weighted by PL) : S_i * L_i / max(P_i, L_i)
  - TL   (Trajectory Length)      : 实际飞行路径长度

坐标系: NED (North-East-Down)
  - X = North, Y = East, Z = Down (Z 负值表示高度)
  - 距离计算使用 2D (X-Y平面) 或 3D，此脚本默认使用 2D

输出:
  - 每个 data_X 文件夹下生成 metrics_results.csv
  - map1/metrics 下生成 overall_metrics_summary.csv（按模型聚合所有场景）
  - map1/metrics 下生成 per_scenario_metrics.csv（每个场景每个模型的详细指标）
"""

import os
import argparse
from pathlib import Path
import csv
import math
import json
from collections import defaultdict

# ============================================================
#  配置区域 —— 请根据你的 AirSim 仿真环境填入目标位置坐标
# ============================================================

# 成功判定阈值（米）
SUCCESS_THRESHOLD = 10.0  # 终点距目标 ≤ 此值视为导航成功

# 是否使用 3D 距离（包含高度 Z）。False 则使用 X-Y 平面 2D 距离
USE_3D_DISTANCE = False

# 每个 data_X 场景的起点位置和目标位置 (NED 坐标)
# start_position: 无人机出发位置 (X_north, Y_east, Z_down)
# goal_position:  导航目标位置 (X_north, Y_east, Z_down)
# ★★★ 请将 goal_position 替换为你 AirSim 环境中各地标的真实坐标 ★★★
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

# 7 种模型（按文件夹命名中的前缀识别）
MODEL_PREFIXES = [
    "qwen-vl-max",
    "qwen-vl-plus",
    "qwen3-vl-flash",
    "qwen3.5-35b-a3b_CLIP",
    "qwen3.5-35b-a3b",
    "qwen3.5-flash_CLIP",
    "qwen3.5-flash",
]

# ============================================================
#  工具函数
# ============================================================

def euclidean_2d(p1, p2):
    """2D 欧式距离 (X-Y 平面)"""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def euclidean_3d(p1, p2):
    """3D 欧式距离"""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


def distance_func(p1, p2):
    """根据配置选择 2D 或 3D 距离"""
    if USE_3D_DISTANCE:
        return euclidean_3d(p1, p2)
    else:
        return euclidean_2d(p1, p2)


def load_trajectory(csv_path):
    """
    读取单个飞行轨迹 CSV 文件
    返回: [(x, y, z), ...] 坐标点列表
    """
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
    """计算轨迹的实际路径长度"""
    if len(trajectory) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(trajectory)):
        total += distance_func(trajectory[i - 1], trajectory[i])
    return total


def compute_navigation_error(trajectory, goal):
    """计算导航误差: 终点到目标的距离"""
    if not trajectory:
        return float('inf')
    final_pos = trajectory[-1]
    return distance_func(final_pos, goal)


def compute_oracle_distance(trajectory, goal):
    """计算轨迹中距目标最近的距离（Oracle 距离）"""
    if not trajectory:
        return float('inf')
    return min(distance_func(p, goal) for p in trajectory)


def compute_episode_metrics(trajectory, start, goal, threshold):
    """
    计算单次飞行的所有指标
    返回字典: {NE, TL, SR, OR, SPL}
    """
    ne = compute_navigation_error(trajectory, goal)
    tl = compute_path_length(trajectory)
    oracle_dist = compute_oracle_distance(trajectory, goal)

    # 成功判定
    success = 1 if ne <= threshold else 0
    oracle_success = 1 if oracle_dist <= threshold else 0

    # 最短路径长度 (起点到目标的直线距离)
    shortest_path = distance_func(start, goal)

    # SPL: S_i * L_i / max(P_i, L_i)
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


def identify_model(folder_name, scenario_name):
    """
    从子文件夹名中识别模型名称
    文件夹名格式: <model_prefix>_<scenario_name>
    注意: 需要按前缀长度降序匹配（避免 "qwen3.5-flash" 误匹配 "qwen3.5-flash_CLIP"）
    """
    # 按长度降序排列前缀，确保 CLIP 版本优先匹配
    sorted_prefixes = sorted(MODEL_PREFIXES, key=len, reverse=True)
    for prefix in sorted_prefixes:
        expected = f"{prefix}_{scenario_name}"
        if folder_name == expected:
            return prefix
    return None


# ============================================================
#  主处理逻辑
# ============================================================

def process_all_data(root_dir, output_dir=None):
    """处理所有 data_1 ~ data_14 文件夹"""

    # 全局结果收集: {model_name: [episode_metrics, ...]}
    summary_dir = Path(output_dir) if output_dir is not None else Path(root_dir) / "metrics"
    summary_dir.mkdir(parents=True, exist_ok=True)
    global_results = defaultdict(list)
    # 场景级结果: {(data_key, model_name): [episode_metrics, ...]}
    scenario_results = defaultdict(list)

    for data_key in sorted(SCENARIO_CONFIG.keys(), key=lambda k: int(k.split('_')[1])):
        config = SCENARIO_CONFIG[data_key]
        scenario_name = config["name"]
        start = config["start_position"]
        goal = config["goal_position"]
        threshold = config.get("success_threshold", SUCCESS_THRESHOLD)

        data_dir = os.path.join(root_dir, data_key)
        if not os.path.isdir(data_dir):
            print(f"[警告] 文件夹不存在，跳过: {data_dir}")
            continue

        print(f"\n{'=' * 70}")
        print(f"处理 {data_key}: {scenario_name}")
        print(f"  起点: ({start[0]:.3f}, {start[1]:.3f}, {start[2]:.3f})")
        print(f"  目标: ({goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f})")
        print(f"  成功阈值: {threshold:.2f} m")
        print(f"  起点→目标直线距离: {distance_func(start, goal):.2f} m")
        print(f"{'=' * 70}")

        # 本场景的每个模型的详细结果
        scenario_detail_rows = []

        for subfolder in sorted(os.listdir(data_dir)):
            subfolder_path = os.path.join(data_dir, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            model_name = identify_model(subfolder, scenario_name)
            if model_name is None:
                print(f"  [警告] 无法识别模型: {subfolder}，跳过")
                continue

            # 收集该模型在本场景下所有 flight 的指标
            csv_files = sorted([f for f in os.listdir(subfolder_path) if f.endswith('.csv')])
            if not csv_files:
                print(f"  [警告] {subfolder} 下无CSV文件")
                continue

            model_episodes = []
            for csv_file in csv_files:
                csv_path = os.path.join(subfolder_path, csv_file)
                traj = load_trajectory(csv_path)
                if not traj:
                    print(f"    [警告] 空轨迹: {csv_file}")
                    continue

                metrics = compute_episode_metrics(traj, start, goal, threshold)
                metrics["flight"] = csv_file
                metrics["model"] = model_name
                metrics["scenario"] = data_key
                model_episodes.append(metrics)

                # 汇入全局
                global_results[model_name].append(metrics)
                scenario_results[(data_key, model_name)].append(metrics)

            # 打印本模型在本场景的汇总
            if model_episodes:
                n = len(model_episodes)
                avg_ne = sum(e["NE"] for e in model_episodes) / n
                avg_tl = sum(e["TL"] for e in model_episodes) / n
                sr = sum(e["SR"] for e in model_episodes) / n * 100
                or_rate = sum(e["OR"] for e in model_episodes) / n * 100
                avg_spl = sum(e["SPL"] for e in model_episodes) / n * 100

                print(f"  [{model_name}] {n}次飞行 | SR={sr:.1f}% | NE={avg_ne:.2f}m | "
                      f"SPL={avg_spl:.1f}% | OR={or_rate:.1f}% | TL={avg_tl:.2f}m")

                # 记录每次飞行详情
                for ep in model_episodes:
                    scenario_detail_rows.append({
                        "Model": ep["model"],
                        "Flight": ep["flight"],
                        "SR": ep["SR"],
                        "NE(m)": f"{ep['NE']:.4f}",
                        "SPL": f"{ep['SPL']:.4f}",
                        "OR": ep["OR"],
                        "TL(m)": f"{ep['TL']:.4f}",
                        "Oracle_Dist(m)": f"{ep['oracle_dist']:.4f}",
                        "Shortest_Path(m)": f"{ep['shortest_path']:.4f}",
                        "Steps": ep["num_steps"],
                    })

        # 写入本场景的详细结果 CSV
        if scenario_detail_rows:
            detail_dir = Path(output_dir) / data_key if output_dir is not None else Path(data_dir)
            detail_dir.mkdir(parents=True, exist_ok=True)
            output_csv = detail_dir / "metrics_results.csv"
            fieldnames = ["Model", "Flight", "SR", "NE(m)", "SPL", "OR", "TL(m)",
                          "Oracle_Dist(m)", "Shortest_Path(m)", "Steps"]
            with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(scenario_detail_rows)
            print(f"  → 详细结果已保存至: {output_csv}")

    # ========================
    #  生成全局汇总报告
    # ========================

    # 1. 每个场景 × 每个模型的聚合指标
    per_scenario_rows = []
    for data_key in sorted(SCENARIO_CONFIG.keys(), key=lambda k: int(k.split('_')[1])):
        config = SCENARIO_CONFIG[data_key]
        for model_name in MODEL_PREFIXES:
            key = (data_key, model_name)
            episodes = scenario_results.get(key, [])
            if not episodes:
                continue
            n = len(episodes)
            per_scenario_rows.append({
                "Scenario": data_key,
                "Target": config["name"],
                "Model": model_name,
                "Num_Flights": n,
                "SR(%)": f"{sum(e['SR'] for e in episodes) / n * 100:.1f}",
                "NE(m)": f"{sum(e['NE'] for e in episodes) / n:.4f}",
                "SPL(%)": f"{sum(e['SPL'] for e in episodes) / n * 100:.1f}",
                "OR(%)": f"{sum(e['OR'] for e in episodes) / n * 100:.1f}",
                "TL(m)": f"{sum(e['TL'] for e in episodes) / n:.4f}",
            })

    per_scenario_csv = os.path.join(summary_dir, "per_scenario_metrics.csv")
    if per_scenario_rows:
        fieldnames = ["Scenario", "Target", "Model", "Num_Flights",
                      "SR(%)", "NE(m)", "SPL(%)", "OR(%)", "TL(m)"]
        with open(per_scenario_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_scenario_rows)
        print(f"\n场景级汇总已保存至: {per_scenario_csv}")

    # 2. 按模型聚合所有场景的总指标
    overall_rows = []
    print(f"\n{'=' * 70}")
    print(f"  全局汇总 (所有 14 个场景)")
    print(f"{'=' * 70}")
    print(f"{'模型':<30} {'SR(%)':<8} {'NE(m)':<10} {'SPL(%)':<8} {'OR(%)':<8} {'TL(m)':<10} {'N':>5}")
    print("-" * 80)

    for model_name in MODEL_PREFIXES:
        episodes = global_results.get(model_name, [])
        if not episodes:
            continue
        n = len(episodes)
        sr = sum(e["SR"] for e in episodes) / n * 100
        ne = sum(e["NE"] for e in episodes) / n
        spl = sum(e["SPL"] for e in episodes) / n * 100
        or_rate = sum(e["OR"] for e in episodes) / n * 100
        tl = sum(e["TL"] for e in episodes) / n

        print(f"{model_name:<30} {sr:<8.1f} {ne:<10.2f} {spl:<8.1f} {or_rate:<8.1f} {tl:<10.2f} {n:>5}")

        overall_rows.append({
            "Model": model_name,
            "Num_Episodes": n,
            "SR(%)": f"{sr:.1f}",
            "NE(m)": f"{ne:.4f}",
            "SPL(%)": f"{spl:.1f}",
            "OR(%)": f"{or_rate:.1f}",
            "TL(m)": f"{tl:.4f}",
        })

    overall_csv = os.path.join(summary_dir, "overall_metrics_summary.csv")
    if overall_rows:
        fieldnames = ["Model", "Num_Episodes", "SR(%)", "NE(m)", "SPL(%)", "OR(%)", "TL(m)"]
        with open(overall_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(overall_rows)
        print(f"\n全局汇总已保存至: {overall_csv}")

    print("\n[DONE] 所有指标计算完成！")


# ============================================================
#  入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline map1 navigation metrics")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, help="Write all generated metrics to a separate directory")
    args = parser.parse_args()
    root = args.data_root
    print(f"数据根目录: {root}")
    print(f"成功阈值: {SUCCESS_THRESHOLD} m")
    print(f"距离模式: {'3D' if USE_3D_DISTANCE else '2D (X-Y 平面)'}")
    process_all_data(root, args.output_dir)
