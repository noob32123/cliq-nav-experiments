# -*- coding: utf-8 -*-
"""
计算 GNSS 拒止前后航向夹角脚本。
提取 GNSS 历史航迹加权预测的飞行向量 (XY平面)
与 大模型视觉导航 (VLM) 第一次明显 XY 平移的飞行向量 (XY平面) 之间的夹角。
"""

import os
import argparse
import csv
import math
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    """提取第一次有明显平面 XY 位移的两个点"""
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
        
        xy_dist = math.hypot(p_next[0] - p_curr[0], p_next[1] - p_curr[1])
        if xy_dist >= xy_threshold:
            return p_curr, p_next
            
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

def vector_angle_2d_degrees(v1, v2):
    """计算两个向量在 XY 平面上的夹角（度数），范围 [0, 180]"""
    # 提取 XY 分量
    v1_2d = np.array([v1[0], v1[1]])
    v2_2d = np.array([v2[0], v2[1]])
    
    norm1 = np.linalg.norm(v1_2d)
    norm2 = np.linalg.norm(v2_2d)
    
    if norm1 < 1e-5 or norm2 < 1e-5:
        return 0.0
        
    cos_theta = np.dot(v1_2d, v2_2d) / (norm1 * norm2)
    # 防治浮点数误差导致 arccos 报错
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    return math.degrees(angle_rad)

def main(history_dir=None):
    history_dir = history_dir or os.path.join(ROOT, "history_tracks")
    if not os.path.isdir(history_dir):
        print(f"找不到历史航迹目录: {history_dir}")
        return

    weights = [0.1, 0.15, 0.2, 0.25, 0.3]
    
    print("=" * 70)
    print(" 计算 GNSS 历史预测航向 与 VLM 视觉导航真实航向 的夹角 (XY 平面)")
    print("=" * 70)

    for data_key in sorted(SCENARIO_CONFIG.keys(), key=lambda k: int(k.split('_')[1])):
        scenario_name = SCENARIO_CONFIG[data_key]["name"]
        data_dir = os.path.join(ROOT, data_key)
        
        # 1. 获取大模型视觉导航 (VLM) 下第一次有明显 XY 平移的飞行向量
        clip_dir = os.path.join(data_dir, f"qwen3.5-35b-a3b_CLIP_{scenario_name}")
        if not os.path.isdir(clip_dir):
            continue
            
        csv_files = sorted([f for f in os.listdir(clip_dir) if f.endswith('.csv')])
        if not csv_files:
            continue
            
        p0, p1 = None, None
        for csv_file in csv_files:
            csv_path = os.path.join(clip_dir, csv_file)
            curr_p0, curr_p1 = get_initial_heading_points(csv_path, xy_threshold=0.5)
            if curr_p0 is not None and curr_p1 is not None:
                if math.hypot(curr_p1[0] - curr_p0[0], curr_p1[1] - curr_p0[1]) >= 0.5:
                    p0, p1 = curr_p0, curr_p1
                    break
                    
        if p0 is None or p1 is None:
            p0, p1 = get_initial_heading_points(os.path.join(clip_dir, csv_files[0]), xy_threshold=0.5)
            
        if p0 is None or p1 is None:
            continue
            
        v_vlm = p1 - p0
        
        # 2. 获取历史航迹点
        hist_csv = os.path.join(history_dir, f"{data_key}_history.csv")
        if not os.path.isfile(hist_csv):
            continue
            
        hist_points = load_history_points(hist_csv)
        if len(hist_points) != 6:
            continue
            
        # 历史航迹最后两个点 H4 和 H5 (H5最靠近P0)
        h5 = hist_points[-1]
        h4 = hist_points[-2]
        
        # H4 与 H5 之间的向量 (H4 -> H5)
        v_h4_h5 = h5 - h4
        
        # 3. 计算夹角 (XY平面)
        angle_deg = vector_angle_2d_degrees(v_h4_h5, v_vlm)
        
        # 4. 判断转向方向 (北东地 NED 坐标系下)
        # 北(X) 东(Y) 地(Z)
        # X1*Y2 - Y1*X2 > 0 时代表向右转 (顺时针)， < 0 代表向左转 (逆时针)
        cross_z = v_h4_h5[0] * v_vlm[1] - v_h4_h5[1] * v_vlm[0]
        if cross_z > 1e-4:
            turn_dir = "[RIGHT] 右转"
        elif cross_z < -1e-4:
            turn_dir = "[LEFT] 左转"
        else:
            turn_dir = "[STRAIGHT] 直行"
        
        print(f"[{data_key}] {scenario_name}:")
        print(f"  历史末端飞行向量 (H4->H5) (XY): [{v_h4_h5[0]:.4f}, {v_h4_h5[1]:.4f}]")
        print(f"  VLM第一次平移向量(P0->P1) (XY): [{v_vlm[0]:.4f}, {v_vlm[1]:.4f}]")
        print(f"  ==> 转向情况: {turn_dir} {angle_deg:.4f} 度\n")
        
        # 5. 将结果附在 CSV 文件后 (新增列)
        rows = []
        with open(hist_csv, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            header = next(reader)
            for r in reader:
                rows.append(r)
                
        if "Turn_Direction" not in header:
            header.extend(["Turn_Direction", "Turn_Angle_Deg"])
            for r in rows:
                r.extend([turn_dir, f"{angle_deg:.4f}"])
        else:
            dir_idx = header.index("Turn_Direction")
            ang_idx = header.index("Turn_Angle_Deg")
            for r in rows:
                r[dir_idx] = turn_dir
                r[ang_idx] = f"{angle_deg:.4f}"
                
        with open(hist_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-dir", default=os.path.join(ROOT, "history_tracks"))
    args = parser.parse_args()
    main(args.history_dir)
