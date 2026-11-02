# -*- coding: utf-8 -*-
"""
验证历史航迹脚本。
本脚本将读取生成的 history_tracks 目录下的CSV文件，并读取原始的轨迹文件提取真实航向。
计算历史航迹点之间的5个向量，进行加权和，并将其与真实航向向量进行方向（夹角余弦值）对比，以验证其严格一致。
"""

import os
import argparse
import csv
import math
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 从之前的配置文件中提取起点
SCENARIO_CONFIG = {
    "data_1":  {"name": "orangewhite_umbrella"},
    "data_2":  {"name": "statue"},
    "data_3":  {"name": "restaurant_across_river"},
    "data_4":  {"name": "hotel_front_door"},
    "data_5":  {"name": "blue_amusement_park"},
    "data_6":  {"name": "fountain"},
    "data_7":  {"name": "house"},
    "data_8":  {"name": "playground"},
    "data_9":  {"name": "parkbench"},
    "data_10": {"name": "umbrella"},
    "data_11": {"name": "house2"},
    "data_12": {"name": "fountain2"},
    "data_13": {"name": "sculpture"},
    "data_14": {"name": "sculpture2"},
}

def get_initial_heading_points(csv_path, xy_threshold=0.5):
    points = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                points.append(np.array([float(row['X']), float(row['Y']), float(row['Z'])]))
            except (ValueError, KeyError):
                continue
                
    if len(points) < 2:
        return None, None
        
    for i in range(len(points) - 1):
        p_curr = points[i]
        p_next = points[i+1]
        
        # 计算平面 XY 距离变化
        xy_dist = math.hypot(p_next[0] - p_curr[0], p_next[1] - p_curr[1])
        
        # 如果 XY 平面变化显著，说明开始朝着目标方向移动了
        if xy_dist >= xy_threshold:
            return p_curr, p_next
            
    # 如果找不到显著的 XY 变化，Fallback 回前两个点
    return points[0], points[1]

def load_history_points(csv_path):
    points = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                points.append(np.array([float(row['X']), float(row['Y']), float(row['Z'])]))
            except (ValueError, KeyError):
                continue
    return points

def main(history_dir=None):
    history_dir = history_dir or os.path.join(ROOT, "history_tracks")
    if not os.path.isdir(history_dir):
        print(f"找不到目录: {history_dir}，请先生成历史航迹。")
        return

    # 权重设定 (离现在起始点越近权重越高)
    weights = [0.1, 0.15, 0.2, 0.25, 0.3]
    print(f"{'='*60}")
    print(f"开始验证 14 条航路的加权航向")
    print(f"权重数组: {weights}")
    print(f"{'='*60}\n")
    
    all_passed = True

    for data_key in sorted(SCENARIO_CONFIG.keys(), key=lambda k: int(k.split('_')[1])):
        scenario_name = SCENARIO_CONFIG[data_key]["name"]
        
        # 1. 获取真实航向向量
        data_dir = os.path.join(ROOT, data_key)
        clip_dir = os.path.join(data_dir, f"qwen3.5-35b-a3b_CLIP_{scenario_name}")
        if not os.path.isdir(clip_dir):
            continue
            
        csv_files = sorted([f for f in os.listdir(clip_dir) if f.endswith('.csv')])
        if not csv_files:
            continue
            
        csv_files = sorted([f for f in os.listdir(clip_dir) if f.endswith('.csv')])
        if not csv_files:
            continue
            
        p0, p1 = None, None
        for csv_file in csv_files:
            csv_path = os.path.join(clip_dir, csv_file)
            curr_p0, curr_p1 = get_initial_heading_points(csv_path, xy_threshold=0.5)
            # 检查是否有显著的平面位移
            if curr_p0 is not None and curr_p1 is not None:
                if math.hypot(curr_p1[0] - curr_p0[0], curr_p1[1] - curr_p0[1]) >= 0.5:
                    p0, p1 = curr_p0, curr_p1
                    break
                    
        # 如果所有文件都没有显著平面位移，就退回到使用第一个文件的原Fallback结果
        if p0 is None or p1 is None:
            p0, p1 = get_initial_heading_points(os.path.join(clip_dir, csv_files[0]), xy_threshold=0.5)
            
        if p0 is None or p1 is None:
            continue
        v_actual = p1 - p0
        norm_v_actual = np.linalg.norm(v_actual)
        
        if norm_v_actual < 1e-5:
            continue
            
        # 2. 获取生成的历史航迹点
        hist_csv = os.path.join(history_dir, f"{data_key}_history.csv")
        if not os.path.isfile(hist_csv):
            print(f"{data_key}: 找不到历史航迹文件 {hist_csv}")
            all_passed = False
            continue
            
        hist_points = load_history_points(hist_csv)
        if len(hist_points) != 6:
            print(f"{data_key}: 历史点数量不是 6 个")
            all_passed = False
            continue
            
        # 3. 计算5个向量
        # 向量是后一个点减前一个点，即 V_i = H_i - H_{i-1}
        vectors = []
        for i in range(1, 6):
            vectors.append(hist_points[i] - hist_points[i-1])
            
        # 4. 加权和
        weighted_sum = np.zeros(3)
        for i in range(5):
            weighted_sum += weights[i] * vectors[i]
            
        norm_w_sum = np.linalg.norm(weighted_sum)
        
        # 5. 计算夹角余弦值
        # cos(theta) = (A dot B) / (|A| * |B|)
        cos_theta = np.dot(weighted_sum, v_actual) / (norm_w_sum * norm_v_actual)
        
        # 验证是否为 1.0 (平行且同向)
        # 考虑到浮点数精度误差，用 np.isclose
        is_parallel = np.isclose(cos_theta, 1.0, atol=1e-5)
        
        print(f"[{data_key}] - {scenario_name}")
        print(f"  真实航向向量 v_actual : [{v_actual[0]:.4f}, {v_actual[1]:.4f}, {v_actual[2]:.4f}]")
        print(f"  历史加权和向量 w_sum  : [{weighted_sum[0]:.4f}, {weighted_sum[1]:.4f}, {weighted_sum[2]:.4f}]")
        print(f"  夹角余弦值 cos(theta) : {cos_theta:.6f} ", end="")
        
        if is_parallel:
            print("=> [PASS] 验证通过 (方向一致)")
        else:
            print("=> [FAIL] 验证失败 (方向不一致)")
            all_passed = False
        print("-" * 60)
        
    print("\n总体验证结果: ", end="")
    if all_passed:
        print("[PASS] 所有场景历史航迹加权方向均完全符合实际航向！")
    else:
        print("[FAIL] 存在不符合实际航向的场景！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-dir", default=os.path.join(ROOT, "history_tracks"))
    args = parser.parse_args()
    main(args.history_dir)
