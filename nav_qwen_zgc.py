import sys
import torch
import clip
from PIL import Image
import setup_path
import os
import time
from datetime import datetime
import re
import base64
import numpy as np
import cv2
from openai import OpenAI
import airsim
import csv
import argparse
import json
import math
from pathlib import Path
from contextlib import contextmanager
from fly_on_keyboard import FixedDroneController  # 复用键盘控制脚本中的类


_model_cache = {}

# BEGIN DOWNTOWN ROUTES
# NED 米；altitude 对应飞行 Z=-altitude，并非离地高度。
# 目标坐标仅作路线审阅/核验，不进入 CLIP 或 Qwen 提示。
ROUTES = {
    # 01 街角蓝白条纹棚侧向接近
    "01": {
        "id": "01",
        "name": "街角蓝白条纹棚侧向接近",
        "start_position": [-5, 15, -1],
        "initial_yaw_deg": -45,
        "turn_direction": "turn_right",
        "turn_degrees": 43.6015,
        "heading_deg": -1.3985,
        "altitude": 3,
        "target": "前方街角米黄色建筑底层蓝白条纹遮阳棚旁的入口附近",
        "clip_text": "blue and white striped restaurant awnings",
        "actor": "signs_sign_sandwich_5",
        "target_position": [16.71731948852539, 14.469796180725098, -3.9880566596984863],
        "goal_position": [12, 15, 0.6],
    },
    # 02 BRUN & PEREZ 黑色门面
    "02": {
        "id": "02",
        "name": "BRUN & PEREZ 黑色门面",
        "start_position": [-84.991, 9, -1],
        "initial_yaw_deg": -45,
        "turn_direction": "turn_left",
        "turn_degrees": 45.0,
        "heading_deg": -90.0,
        "altitude": 3,
        "target": "前方 BRUN & PEREZ 黑色长招牌下的内凹入口附近，两侧有橘红色促销海报",
        "clip_text": "black storefront entrance with orange sale posters",
        "actor": "signs_sign_brunperez_11",
        "target_position": [-84.99101257324219, -16.731689453125, -3.743428945541382],
        "goal_position": [-84.991, -12.732, 0.6],
    },
    # 03 条纹棚餐厅与绿色立柱
    "03": {
        "id": "03",
        "name": "条纹棚餐厅与绿色立柱",
        "start_position": [-57, 8, -1],
        "initial_yaw_deg": -45,
        "turn_direction": "turn_left",
        "turn_degrees": 57.1919,
        "heading_deg": -102.1919,
        "altitude": 3,
        "target": "前方蓝白条纹遮阳棚、户外深色桌椅和绿色立柱旁的餐厅入口附近",
        "clip_text": "restaurant entrance with striped awnings and outdoor tables",
        "actor": "signs_sign_bistro_14",
        "target_position": [-62.30513381958008, -16.553997039794922, -3.946885824203491],
        "goal_position": [-60, -11, 0.6],
    },
    # 04 Weefit 白色门面
    "04": {
        "id": "04",
        "name": "Weefit 白色门面",
        "start_position": [-97.044, 9, -1],
        "initial_yaw_deg": -45,
        "turn_direction": "turn_left",
        "turn_degrees": 45.0009,
        "heading_deg": -90.0009,
        "altitude": 3,
        "target": "前方白色门框、红色 Weefit 招牌的商店入口附近",
        "clip_text": "white double doors at the entrance of a shop",
        "actor": "signs_sign_weefit_24",
        "target_position": [-97.0444107055664, -16.71669578552246, -3.952580451965332],
        "goal_position": [-97.044, -12.717, 0.6],
    },
    # 05 蓝灰框内凹入口
    "05": {
        "id": "05",
        "name": "蓝灰框内凹入口",
        "start_position": [-164.18, -9, -1],
        "initial_yaw_deg": -45,
        "turn_direction": "turn_right",
        "turn_degrees": 134.9997,
        "heading_deg": 89.9997,
        "altitude": 3,
        "target": "前方蓝灰色框架、深色玻璃的内凹商店入口附近",
        "clip_text": "recessed storefront entrance with dark blue frames and glass windows",
        "actor": "signs_sign_electronics_47",
        "target_position": [-164.1798858642578, 15.900839805603027, -3.049642324447632],
        "goal_position": [-164.18, 11.901, 0.6],
    },
    # 06 EUROPA 红框门面
    "06": {
        "id": "06",
        "name": "EUROPA 红框门面",
        "start_position": [-78.646, 9, -1],
        "initial_yaw_deg": -45,
        "turn_direction": "turn_left",
        "turn_degrees": 44.9996,
        "heading_deg": -89.9996,
        "altitude": 3,
        "target": "前方 EUROPA 黑色招牌下、红色门框的商店入口附近",
        "clip_text": "red framed storefront entrance with a black EUROPA sign",
        "actor": "signs_sign_europa_65",
        "target_position": [-78.64580535888672, -16.772363662719727, -3.578608989715576],
        "goal_position": [-78.646, -12.772, 0.6],
    },
    # 07 Jacob's Deli 正面接近
    "07": {
        "id": "07",
        "name": "Jacob's Deli 正面接近",
        "start_position": [22.717, -9, -1],
        "initial_yaw_deg": -45,
        "turn_direction": "turn_right",
        "turn_degrees": 134.9995,
        "heading_deg": 89.9995,
        "altitude": 3,
        "target": "前方蓝白条纹遮阳棚之间、Jacob's Downtown Deli 招牌下的入口附近",
        "clip_text": "deli entrance between blue and white striped awnings",
        "actor": "signs_sign_locafe_68",
        "target_position": [22.717199325561523, 14.48806095123291, -4.0690717697143555],
        "goal_position": [22.717, 10.488, 0.6],
    },
    # 08 街尾浅蓝遮阳棚
    "08": {
        "id": "08",
        "name": "街尾浅蓝遮阳棚",
        "start_position": [-221.655, 9, -1],
        "initial_yaw_deg": -45,
        "turn_direction": "turn_left",
        "turn_degrees": 44.9991,
        "heading_deg": -89.9991,
        "altitude": 3,
        "target": "前方街尾浅色建筑底层浅蓝色遮阳棚下的门口附近",
        "clip_text": "light blue shop awning",
        "actor": "signs_sign_suni_75",
        "target_position": [-221.6545867919922, -17.051128387451172, -4.133761405944824],
        "goal_position": [-221.655, -13.051, 0.6],
    },
    # 09 西段黄色 SHOP 标志
    "09": {
        "id": "09",
        "name": "西段黄色 SHOP 标志",
        "start_position": [-193, 11, -1],
        "initial_yaw_deg": -45,
        "turn_direction": "turn_left",
        "turn_degrees": 144.2748,
        "heading_deg": 170.7252,
        "altitude": 3,
        "target": "前方浅色建筑底层、黄色 SHOP 图案招牌旁的商店入口附近",
        "clip_text": "shop entrance with a yellow hanging sign",
        "actor": "signs_sign_shop_78",
        "target_position": [-208.4520263671875, 13.523380279541016, -3.8973917961120605],
        "goal_position": [-204.5, 11, 0.6],
    },
    # 10 Sleevo 黑白门面
    "10": {
        "id": "10",
        "name": "Sleevo 黑白门面",
        "start_position": [-160, -13, -1],
        "initial_yaw_deg": -45,
        "turn_direction": "turn_left",
        "turn_degrees": 121.7917,
        "heading_deg": -166.7917,
        "altitude": 1.8,
        "target": "前方白色 Sleevo 招牌、深色玻璃门面的商店入口附近",
        "clip_text": "black glass storefront with a white Sleevo sign",
        "actor": "signs_sign_sleevo_84",
        "target_position": [-176.56314086914062, -16.88739013671875, -3.237053155899048],
        "goal_position": [-172, -13, 0.6],
    },
    # 11 转角绿色遮阳棚
    "11": {
        "id": "11",
        "name": "转角绿色遮阳棚",
        "start_position": [-185, -3, -1],
        "initial_yaw_deg": -45,
        "turn_direction": "turn_right",
        "turn_degrees": 131.4209,
        "heading_deg": 86.4209,
        "altitude": 3,
        "target": "前方建筑转角绿色遮阳棚下的店铺入口附近",
        "clip_text": "green awning above a corner shop entrance",
        "actor": "signs_sign_deli_222",
        "target_position": [-183.8603515625, 15.220085144042969, -4.079463481903076],
        "goal_position": [-185, 10, 0.6],
    },
    # 12 uFoods 食品店
    "12": {
        "id": "12",
        "name": "uFoods 食品店",
        "start_position": [-133.423, 9, -1],
        "initial_yaw_deg": -45,
        "turn_direction": "turn_left",
        "turn_degrees": 44.9998,
        "heading_deg": -89.9998,
        "altitude": 3,
        "target": "前方带红色番茄标志和 uFoods 字样、绿色遮阳棚下的入口附近",
        "clip_text": "green awning shop with a red tomato sign",
        "actor": "signs_sign_uFoods_225",
        "target_position": [-133.4229278564453, -17.014875411987305, -3.563486337661743],
        "goal_position": [-133.423, -13.015, 0.6],
    },
    # 13 UPSWAY 橘色遮阳棚
    "13": {
        "id": "13",
        "name": "UPSWAY 橘色遮阳棚",
        "start_position": [-90.598, 9, -1],
        "initial_yaw_deg": -45,
        "turn_direction": "turn_left",
        "turn_degrees": 44.9995,
        "heading_deg": -89.9995,
        "altitude": 3,
        "target": "前方 UPSWAY 招牌下橘色遮阳棚覆盖的小门口附近",
        "clip_text": "orange awning over a narrow shop doorway",
        "actor": "signs_sign_upsway2_229",
        "target_position": [-90.5977783203125, -16.80732536315918, -3.7266228199005127],
        "goal_position": [-90.598, -12.807, 0.6],
    },
    # 14 古董店门面
    "14": {
        "id": "14",
        "name": "古董店门面",
        "start_position": [44, -2, -1],
        "initial_yaw_deg": -45,
        "turn_direction": "turn_right",
        "turn_degrees": 166.6169,
        "heading_deg": 121.6169,
        "altitude": 3,
        "target": "前方棕红色遮阳棚下的店铺入口附近",
        "clip_text": "antique shop entrance with a dark red awning",
        "actor": "signs_sign_antiques2_232",
        "target_position": [33.67453384399414, 14.772706985473633, -3.5950474739074707],
        "goal_position": [37, 10, 0.6],
    },
    # 15 红框绿色棚餐厅
    "15": {
        "id": "15",
        "name": "红框绿色棚餐厅",
        "start_position": [-49, -8, -1],
        "initial_yaw_deg": -45,
        "turn_direction": "turn_right",
        "turn_degrees": 112.7623,
        "heading_deg": 67.7623,
        "altitude": 3,
        "target": "前方红色门窗框、绿色长遮阳棚和户外桌椅的餐厅入口附近",
        "clip_text": "restaurant with red window frames and a long green awning",
        "actor": "signs_sign_inkasipas_90",
        "target_position": [-40.260276794433594, 13.375778198242188, -3.6075525283813477],
        "goal_position": [-44, 9, 0.6],
    },
    # 16 红白条纹棚咖啡店
    "16": {
        "id": "16",
        "name": "红白条纹棚咖啡店",
        "start_position": [-194, 0, -1],
        "initial_yaw_deg": -45,
        "turn_direction": "turn_left",
        "turn_degrees": 19.794,
        "heading_deg": -64.794,
        "altitude": 3,
        "target": "前方白色咖啡店招牌、红白条纹小遮阳棚、黄色门框的入口附近",
        "clip_text": "cafe with a white sign and striped awning",
        "actor": "signs_sign_locafe3_116",
        "target_position": [-186.06468200683594, -16.858808517456055, -3.971292018890381],
        "goal_position": [-194, -10, 0.6],
    },
    # 17 CAFE VENDIE 条纹棚
    "17": {
        "id": "17",
        "name": "CAFE VENDIE 条纹棚",
        "start_position": [-154, -3, -1],
        "initial_yaw_deg": -45,
        "turn_direction": "turn_right",
        "turn_degrees": 138.2452,
        "heading_deg": 93.2452,
        "altitude": 3,
        "target": "前方 CAFE VENDIE 白色咖啡杯招牌、蓝白条纹遮阳棚下的咖啡店入口附近",
        "clip_text": "cafe entrance with blue and white striped awnings and a coffee cup sign",
        "actor": "signs_sign_cafe_358",
        "target_position": [-155.09999084472656, 16.399999618530273, -4.068830966949463],
        "goal_position": [-153.5, 12, 0.6],
    },
    # 18 THRIFTY 内凹入口
    "18": {
        "id": "18",
        "name": "THRIFTY 内凹入口",
        "start_position": [66.096, -9, -1],
        "initial_yaw_deg": -45,
        "turn_direction": "turn_right",
        "turn_degrees": 134.9998,
        "heading_deg": 89.9998,
        "altitude": 3,
        "target": "前方 THRIFTY 字样下方的内凹商店入口附近",
        "clip_text": "recessed shop doorway under a THRIFTY sign",
        "actor": "signs_sign_sleevo2",
        "target_position": [66.0960693359375, 14.626176834106445, -2.3370532989501953],
        "goal_position": [66.096, 10.626, 0.6],
    },
    # 19 LOCAFE 咖啡店
    "19": {
        "id": "19",
        "name": "LOCAFE 咖啡店",
        "start_position": [76.735, -9, -1],
        "initial_yaw_deg": -45,
        "turn_direction": "turn_right",
        "turn_degrees": 134.9992,
        "heading_deg": 89.9992,
        "altitude": 3,
        "target": "前方 LOCAFE 白色招牌和咖啡杯圆形标志下、白色框架的内凹入口附近",
        "clip_text": "recessed cafe entrance with white window frames and a coffee cup sign",
        "actor": "signs_sign_locafe4",
        "target_position": [76.73531341552734, 14.650792121887207, -3.4212920665740967],
        "goal_position": [76.735, 10.651, 0.6],
    },
    # 20 道路对面浅绿色棚
    "20": {
        "id": "20",
        "name": "道路对面浅绿色棚",
        "start_position": [-7, 39.642, -1],
        "initial_yaw_deg": -45,
        "turn_direction": "turn_right",
        "turn_degrees": 44.9989,
        "heading_deg": -0.0011,
        "altitude": 3,
        "target": "道路对面浅色建筑底层、浅绿色遮阳棚下的门口附近",
        "clip_text": "light green awning above a doorway in a pale stone building",
        "actor": "signs_sign_suni2",
        "target_position": [16.70755386352539, 39.641544342041016, -3.8485467433929443],
        "goal_position": [12.708, 39.642, 0.6],
    },
    # 21 鹰雕像花坛
    "21": {
        "id": "21",
        "name": "鹰雕像花坛",
        "start_position": [106, 9, -1],
        "initial_yaw_deg": -45,
        "turn_direction": "turn_left",
        "turn_degrees": 125.2137,
        "heading_deg": -170.2137,
        "altitude": 2,
        "target": "前方花坛中的鹰雕像附近，在花坛外侧空地降落",
        "clip_text": "eagle statue in a flower bed",
        "actor": "prop_statue_eagle_11",
        "target_position": [94.84300231933594, 7.075596809387207, 0.3154473900794983],
        "goal_position": [100, 8, 0.6],
    },
    # 22 鹿雕像花坛
    "22": {
        "id": "22",
        "name": "鹿雕像花坛",
        "start_position": [-162, -8, -1],
        "initial_yaw_deg": -45,
        "turn_direction": "turn_right",
        "turn_degrees": 171.0788,
        "heading_deg": 126.0788,
        "altitude": 2,
        "target": "前方花坛中的鹿雕像附近，在花坛外侧空地降落",
        "clip_text": "deer statue in a flower bed",
        "actor": "SM_prop_statue_deer_5",
        "target_position": [-169.0281982421875, 1.645551085472107, 0.2626204490661621],
        "goal_position": [-167, -3, 0.6],
    },
    # 23 公园喷泉
    "23": {
        "id": "23",
        "name": "公园喷泉",
        "start_position": [61.6, -20, -1],
        "initial_yaw_deg": -45,
        "turn_direction": "turn_left",
        "turn_degrees": 45.0277,
        "heading_deg": -90.0277,
        "altitude": 3,
        "target": "前方公园圆形喷泉附近，在喷泉外侧铺装空地降落",
        "clip_text": "circular fountain in a park",
        "actor": "FountainFX_2",
        "target_position": [61.5884895324707, -43.776023864746094, 1.0200068950653076],
        "goal_position": [61.6, -37, 0.6],
    },
    # 24 东段红黄伞餐车
    "24": {
        "id": "24",
        "name": "东段红黄伞餐车",
        "start_position": [35, -3, -1],
        "initial_yaw_deg": -45,
        "turn_direction": "turn_right",
        "turn_degrees": 61.5435,
        "heading_deg": 16.5435,
        "altitude": 3,
        "target": "前方步行街上红黄相间遮阳伞下的餐车附近，在餐车旁空地降落",
        "clip_text": "food cart with a red and yellow umbrella",
        "actor": "food_cart2",
        "target_position": [49.87188720703125, 1.417551875114441, 1.0045961141586304],
        "goal_position": [46, -1, 0.6],
    },
    # 25 西段红黄伞餐车
    "25": {
        "id": "25",
        "name": "西段红黄伞餐车",
        "start_position": [-142, -1, -1],
        "initial_yaw_deg": -45,
        "turn_direction": "turn_left",
        "turn_degrees": 149.1697,
        "heading_deg": 165.8303,
        "altitude": 3,
        "target": "前方步行街上红黄相间遮阳伞下的餐车附近，在餐车旁空地降落",
        "clip_text": "street food cart with a red and yellow umbrella",
        "actor": "food_cart5_278",
        "target_position": [-157.7163848876953, 2.968005895614624, 0.9942361116409302],
        "goal_position": [-155, 0, 0.6],
    },
    # 26 街角紫色遮阳伞
    "26": {
        "id": "26",
        "name": "街角紫色遮阳伞",
        "start_position": [8, -22.4, -1],
        "initial_yaw_deg": -45,
        "turn_direction": "turn_left",
        "turn_degrees": 134.8605,
        "heading_deg": -179.8605,
        "altitude": 3,
        "target": "前方米黄色街角建筑旁的紫色遮阳伞桌椅区附近，在桌椅外侧空地降落",
        "clip_text": "purple patio umbrellas outside a corner cafe",
        "actor": "prop_table_umbrella_a_7",
        "target_position": [-7.530877590179443, -22.437816619873047, 0.9942368865013123],
        "goal_position": [-5.5, -24.5, 0.6],
    },
    # 27 街角黑伞白桌椅
    "27": {
        "id": "27",
        "name": "街角黑伞白桌椅",
        "start_position": [5, 23.5, -1],
        "initial_yaw_deg": -45,
        "turn_direction": "turn_left",
        "turn_degrees": 134.9779,
        "heading_deg": -179.9779,
        "altitude": 3,
        "target": "前方街角建筑旁黑色遮阳伞、白色桌椅的休息区附近，在桌椅外侧空地降落",
        "clip_text": "black patio umbrellas over white cafe tables and chairs",
        "actor": "prop_table_umbrella_a8",
        "target_position": [-8.02847957611084, 23.494970321655273, 0.9942368865013123],
        "goal_position": [-4, 23.5, 0.6],
    },
    # 28 公园喷泉侧向接近
    "28": {
        "id": "28",
        "name": "公园喷泉侧向接近",
        "start_position": [80, -44, -1],
        "initial_yaw_deg": -45,
        "turn_direction": "turn_left",
        "turn_degrees": 135.697,
        "heading_deg": 179.303,
        "altitude": 3,
        "target": "前方圆形喷泉附近，从公园一侧接近，在喷泉外侧铺装空地降落",
        "clip_text": "circular fountain in a park",
        "actor": "FountainFX_2",
        "target_position": [61.5884895324707, -43.776023864746094, 1.0200068950653076],
        "goal_position": [69, -44, 0.6],
    },
    # 29 路口花盆之间的木长椅
    "29": {
        "id": "29",
        "name": "路口花盆之间的木长椅",
        "start_position": [114, -15, -1],
        "initial_yaw_deg": -45,
        "turn_direction": "turn_right",
        "turn_degrees": 44.4769,
        "heading_deg": -0.5231,
        "altitude": 1.5,
        "target": "前方路口小岛上、两个大花盆之间的木长椅附近，在长椅前方空地降落",
        "clip_text": "wooden bench between two large plant pots and trash cans",
        "actor": "bench_wood_a18",
        "target_position": [126.57122039794922, -15.114774703979492, 0.9942365884780884],
        "goal_position": [123, -15, 0.6],
    },
    # 30 街尾紫伞休息区
    "30": {
        "id": "30",
        "name": "街尾紫伞休息区",
        "start_position": [-202, -14, -1],
        "initial_yaw_deg": -45,
        "turn_direction": "turn_left",
        "turn_degrees": 131.0167,
        "heading_deg": -176.0167,
        "altitude": 3,
        "target": "前方街尾建筑旁紫色遮阳伞、白色桌椅的休息区附近，在桌椅外侧空地降落",
        "clip_text": "purple patio umbrella over white tables beside a shop",
        "actor": "prop_table_umbrella_a11",
        "target_position": [-214.7767333984375, -14.889687538146973, 0.9942368865013123],
        "goal_position": [-210, -14, 0.6],
    },
}
# END DOWNTOWN ROUTES


def initialize_route(drone, route, clock_speed, start_override=None):
    """统一应用一条路线；初始旋转使用绝对偏航闭环控制，避免小角度过转。"""
    initial_yaw = route["initial_yaw_deg"]
    orientation = airsim.to_quaternion(0, 0, math.radians(initial_yaw))
    start = tuple(start_override or route["start_position"])
    reset_to_start(drone.client, start, orientation, clock_speed)
    # 相机位置为机体 NED；正 Z 位于机身下方，防止俯视图拍到机身。
    drone.client.simSetCameraPose("front_center", airsim.Pose(
        airsim.Vector3r(.5, 0, 0), airsim.to_quaternion(0, 0, 0)))
    drone.client.simSetCameraPose("bottom_center", airsim.Pose(
        airsim.Vector3r(0, 0, .7), airsim.to_quaternion(-math.pi / 2, 0, 0)))
    drone.is_flying = False
    drone.takeoff(route["altitude"], settle_seconds=1.0 / clock_speed)
    sign = -1 if route["turn_direction"] == "turn_left" else 1
    heading = (initial_yaw + sign * route["turn_degrees"] + 180) % 360 - 180
    drone.client.rotateToYawAsync(heading, timeout_sec=15, margin=1).join()
    drone.client.moveByVelocityZAsync(0, 0, -route["altitude"], 3).join()
    actual = math.degrees(airsim.to_eularian_angles(drone.client.simGetVehiclePose().orientation)[2])
    error = abs((actual - heading + 180) % 360 - 180)
    if error > 3:
        raise RuntimeError(f"初始朝向误差 {error:.2f} 度，请检查路线起点和障碍物")
    actual_z = drone.client.simGetVehiclePose().position.z_val
    if abs(actual_z + route["altitude"]) > .35:
        raise RuntimeError(f"未达到路线起飞高度：实际 Z={actual_z:.2f}，预期 {-route['altitude']}")
    if drone.client.simGetCollisionInfo().has_collided:
        raise RuntimeError("路线初始化发生碰撞，请检查起点、起飞高度及地图版本")
    print(f"路线 {route['id']}：{route['name']}；朝向 {actual:.2f}°；NED Z={-route['altitude']}")
    return actual



def read_clock_speed(client):
    """读取服务器实际加载的配置；不能用尚未生效的本地配置缩短等待。"""
    try:
        speed = float(json.loads(client.getSettingsString()).get("ClockSpeed", 1.0))
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("ClockSpeed 必须是有限正数")
        return speed
    except Exception as exc:
        print(f"无法读取仿真倍率，稳定等待按 1 倍处理: {exc}")
        return 1.0


def save_profile(path, record):
    if path is not None:
        with open(path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


@contextmanager
def pause_for_inference(client, enabled=True):
    """推理期间冻结物理状态，避免网络耗时被仿真倍率放大；异常时也恢复。"""
    if enabled:
        client.simPause(True)
    try:
        yield
    finally:
        if enabled:
            client.simPause(False)

# 编号对应的位置描述（行优先，从上到下，从左到右）
POSITION_DESC = {
    1: "前上方偏左",
    2: "正前上方",
    3: "前上方偏右",
    4: "前方偏左",
    5: "正前方",
    6: "前方偏右",
    7: "前下方偏左",
    8: "正前下方",
    9: "前下方偏右"
}

def reset_to_start(client, start_position, start_orientation, clock_speed=1.0):
    """
    将无人机重置到指定的起点。
    :param client: AirSim 客户端
    :param start_position: 元组 (x, y, z)，NED坐标系（z负值表示高度）
    :param start_orientation: AirSim 四元数或 (x, y, z, w) 元组
    """
    # 1. 重置整个仿真到初始状态（清除物理效果、恢复车辆）
    client.reset()
    
    # 2. 重新获取API控制并解锁（reset后需要重新启用）
    client.enableApiControl(True)
    client.armDisarm(True)
    
    # 3. 构造目标位姿，四元数分量顺序为 x, y, z, w。
    orientation = airsim.Quaternionr(*start_orientation)

    target_pose = airsim.Pose(
        airsim.Vector3r(*start_position),
        orientation
    )
    
    # 4. 将无人机传送到指定位置（忽略碰撞，直接放置）
    client.simSetVehiclePose(target_pose, ignore_collision=True)
    
    # 5. 保持零速度稳定 2 个模拟秒；API duration 已受 ClockSpeed 控制。
    # 只 sleep 会让刚传送到空中的无人机在尚未起飞时自由下落。
    client.moveByVelocityAsync(0, 0, 0, 2.0).join()
    
    # 验证位置
    pose = client.simGetVehiclePose()
    print(f"已重置到起点: X={pose.position.x_val:.2f}, Y={pose.position.y_val:.2f}, Z={pose.position.z_val:.2f}")

# ========== CLIP函数 ==========
def find_best_patch(image_path, text, model_name="ViT-B/32", device="cuda:0", save_dir=None):
    """
    将输入图片分割成3x3九个小块，找出与文本最匹配的小块，并可选择保存所有小块。

    Args:
        image_path (str): 输入图片路径
        text (str): 文本描述，例如 "a dog"
        model_name (str): CLIP模型名称，默认 "ViT-B/32"
        device (str): 运行设备，默认 "cuda:0"
        save_dir (str, optional): 保存小图片的目录，默认为None（不保存）。如果指定，则将九个小块保存到该目录。

    Returns:
        best_index (int): 最匹配的小块编号，从1到9
        best_logit (float): 对应的logits值（越大表示越匹配）
        best_patch (PIL.Image): 最匹配的小块图像
        position (str): 该小块在原图中的位置描述，例如 "上方偏左"
    """
    # 设备设置
    device = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")

    # 加载模型（带缓存）
    cache_key = (model_name, device)
    if cache_key not in _model_cache:
        print(f"Loading CLIP model '{model_name}' on {device}...")
        model, preprocess = clip.load(model_name, device=device)
        _model_cache[cache_key] = (model, preprocess)
    else:
        model, preprocess = _model_cache[cache_key]

    # 打开图片，调整尺寸至能被3整除
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"无法打开图片 {image_path}: {e}")

    w, h = img.size
    new_w = (w // 3) * 3
    new_h = (h // 3) * 3
    if new_w != w or new_h != h:
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    tile_w, tile_h = new_w // 3, new_h // 3

    # 准备保存目录（如果需要保存）
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 分割九个小块并预处理
    patches = []
    image_tensors = []
    for row in range(3):
        for col in range(3):
            left = col * tile_w
            upper = row * tile_h
            right = left + tile_w
            lower = upper + tile_h
            patch = img.crop((left, upper, right, lower))
            patches.append(patch)
            tensor = preprocess(patch).unsqueeze(0)      # (1,3,224,224)
            image_tensors.append(tensor)

            # 如果需要保存，则保存当前小块
            if save_dir is not None:
                patch_index = row * 3 + col + 1
                save_path = os.path.join(save_dir, f"{base_name}_part_{patch_index}.png")
                patch.save(save_path)
                # print(f"已保存: {save_path}")

    images_batch = torch.cat(image_tensors, dim=0).to(device)  # (9,3,224,224)
    text_tokens = clip.tokenize([text]).to(device)

    # 推理
    with torch.no_grad():
        logits_per_image, _ = model(images_batch, text_tokens)  # (9,1)
        logits = logits_per_image.squeeze(-1).cpu().numpy()     # (9,)

    # 找出最佳小块
    best_idx = logits.argmax()          # 0~8
    best_index = best_idx + 1            # 转为1~9
    best_logit = float(logits[best_idx])
    best_patch = patches[best_idx]
    position = POSITION_DESC[best_index]

    return best_index, best_logit, best_patch, position

# ========== 图片编码函数 ==========
def encode_numpy_to_data_url(img_np, image_format='jpeg'):
    """
    将 numpy 数组编码为 base64 data URL
    支持灰度图 (H,W) 或彩色图 (H,W,3)
    """
    success, buffer = cv2.imencode(f'.{image_format}', img_np)
    if not success:
        raise RuntimeError("图像编码失败")
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/{image_format};base64,{b64_str}"

# ========== 动作解析 ==========
def parse_action(action_str):
    """
    从模型输出中提取动作和数值
    期望格式如: forward 2, turn_left 30, stop 等
    返回 (动作名称, 数值) 或 (None, None) 若解析失败
    """
    action_str = action_str.strip().lower()
    if action_str == 'stop':
        return 'stop', 0

    pattern = r'^(forward|backward|left|right|up|down|turn_left|turn_right)\s+(\d+(?:\.\d+)?)$'
    match = re.match(pattern, action_str)
    if match:
        action = match.group(1)
        value = float(match.group(2))
        return action, value
    else:
        print(f"警告：无法解析动作指令 '{action_str}'")
        return None, None

# ========== 动作执行 ==========
def execute_action(controller, action, value):
    """
    根据动作指令控制无人机移动
    """
    horizontal_speed = controller.current_velocity
    vertical_speed = controller.current_vertical_velocity
    yaw_rate = controller.current_yaw_rate
    client = controller.client

    if action in ['forward', 'backward', 'left', 'right']:
        duration = max(0.5, value / horizontal_speed)
        if action == 'forward':
            forward_vel = horizontal_speed
            right_vel = 0
        elif action == 'backward':
            forward_vel = -horizontal_speed
            right_vel = 0
        elif action == 'left':
            forward_vel = 0
            right_vel = -horizontal_speed
        elif action == 'right':
            forward_vel = 0
            right_vel = horizontal_speed
        else:
            return
        # 相对速度转世界坐标
        vx, vy = controller.rotate_vector_to_world(forward_vel, right_vel)
        print(f"执行 {action} {value:.1f} 米 (速度 {horizontal_speed:.1f} m/s, 持续时间 {duration:.1f}s)")
        client.moveByVelocityAsync(vx, vy, 0, duration).join()

    elif action in ['up', 'down']:
        duration = max(0.5, value / vertical_speed)
        if action == 'up':
            vz = -vertical_speed   # 上升为负
        else:
            vz = vertical_speed    # 下降为正
        print(f"执行 {action} {value:.1f} 米 (速度 {vertical_speed:.1f} m/s, 持续时间 {duration:.1f}s)")
        client.moveByVelocityAsync(0, 0, vz, duration).join()

    elif action in ['turn_left', 'turn_right']:
        duration = max(0.5, value / yaw_rate)
        if action == 'turn_left':
            rate = -yaw_rate
        else:
            rate = yaw_rate
        print(f"执行 {action} {value:.1f} 度 (速率 {yaw_rate:.1f} °/s, 持续时间 {duration:.1f}s)")
        client.moveByVelocityAsync(0, 0, 0, duration,
                                   yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=rate)).join()
    else:
        print(f"未知动作: {action}")

# ========== 主流程 ==========
def build_target_prompt(target):
    return (
    "你是一架无人机的飞手，现在正在依靠无人机上搭载的相机飞行。当前任务：请智能地导航到" + target + "并降落，不必太过精细，大概到达目标附近后即可输出stop"
    "我会按顺序给你三张无人机上搭载的相机拍到的图片：前向图、前向深度图、俯视图。请依据深度图像的信息尽量避免碰撞附近的物体并灵活的导航到目标附近后降落。"
    "请根据这些图片决定下一步动作，你可以采取的动作如下列所示。"
    "输出格式必须为以下之一（数值为数字）：\n"
    "- forward X（向前飞X米）\n"
    "- backward X（向后飞X米）\n"
    "- left X（向左飞X米）\n"
    "- right X（向右飞X米）\n"
    "- up X（上升X米）\n"
    "- down X（下降X米）\n"
    "- turn_left Y（左转Y度）\n"
    "- turn_right Y（右转Y度）\n"
    "- stop（到达目标，停止导航）\n"
    "只输出一条指令，不要附加任何解释。"
    )


def main():
    parser = argparse.ArgumentParser(description="Map2 CLIP + Qwen 导航")
    parser.add_argument("--route-id", choices=sorted(ROUTES), default="01", help="Downtown 路线编号 01–30")
    parser.add_argument("--flights", type=int, default=10)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--start-position", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="覆盖本次运行起点，使用 AirSim NED 米坐标")
    parser.add_argument("--profile", action="store_true", help="记录各阶段现实耗时和仿真时间")
    parser.add_argument("--clip", action=argparse.BooleanOptionalAction, default=True,
                        help="启用 CLIP 方位提示；--no-clip 为仅 Qwen 消融")
    parser.add_argument("--pause-during-inference", action=argparse.BooleanOptionalAction,
                        default=True, help="取图、CLIP 和 Qwen 推理期间暂停物理仿真，默认启用")
    args = parser.parse_args()
    route = ROUTES[args.route_id]
    if args.flights < 1 or args.max_iterations < 1:
        parser.error("flights 和 max-iterations 必须大于零")
    if args.start_position is not None and not all(math.isfinite(v) for v in args.start_position):
        parser.error("起点坐标必须为有限数值")
    # 公网 Qwen：运行前设置环境变量 OPENAI_API_KEY
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("请先设置环境变量 OPENAI_API_KEY，再运行公网 Qwen 导航脚本。")
    client_ai = OpenAI(
        api_key=api_key,
        base_url="https://api.desire-drive.com/v1",
        max_retries=0,
    )

    root_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root_dir = Path(__file__).resolve().parent / "map2" / f"route_{args.route_id}_flight_logs_{root_timestamp}"
    os.makedirs(root_dir, exist_ok=True)
    print(f"轨迹保存目录: {root_dir}")
    profile_path = root_dir / "profile.jsonl" if args.profile else None

    # 初始化无人机控制器（复用键盘控制脚本的类）
    drone = FixedDroneController()
    print("无人机已连接。")
    clock_speed = read_clock_speed(drone.client)
    print(f"服务器 ClockSpeed: {clock_speed:g}；每步稳定等待: {4.0 / clock_speed:.3f}s")
    print(f"推理期间暂停物理仿真: {args.pause_during_inference}")
    save_profile(profile_path, {"event": "config", "clock_speed": clock_speed,
                               "pause_during_inference": args.pause_during_inference,
                               "route_id": args.route_id, "route": route,
                               "start_position_override": args.start_position,
                               "flights": args.flights, "max_iterations": args.max_iterations})

    wind = airsim.Vector3r(0, 0, 0)
    drone.client.simSetWind(wind)

    # 执行十次飞行
    for flight_idx in range(1, args.flights + 1):
        print(f"\n{'='*60}")
        print(f"开始第 {flight_idx} 次飞行")
        print(f"{'='*60}")

        flight_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(root_dir, f"flight_{flight_idx:02d}_{flight_time}.csv")

        initialize_route(drone, route, clock_speed, args.start_position)

        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["X", "Y", "Z"])
            pose = drone.client.simGetVehiclePose()
            writer.writerow([pose.position.x_val, pose.position.y_val, pose.position.z_val])

        # 前向彩色、前向深度、机身下方俯视彩色。
        requests = [
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),      # 前向彩色
            airsim.ImageRequest("front_center", airsim.ImageType.DepthVis, False, False),   # 前向深度可视化
            airsim.ImageRequest("bottom_center", airsim.ImageType.Scene, False, False)       # 俯视彩色
        ]

        max_iterations = args.max_iterations
        # max_iterations = 25
        # max_iterations = 35
        # max_iterations = 40
        for i in range(max_iterations):
            print(f"\n===== 迭代 {i+1} =====")
            step_started = time.perf_counter()
            sim_started = drone.client.getMultirotorState().timestamp if args.profile else None

            # 1. 获取三张图像
            with pause_for_inference(drone.client, args.pause_during_inference):
                inference_state = drone.client.getMultirotorState() if args.profile else None
                responses = drone.client.simGetImages(requests)

                # 2. 将响应转换为 data URL 列表
                image_urls = []
                for idx, resp in enumerate(responses):
                    if not resp or resp.height == 0:
                        print(f"警告：第 {idx+1} 张图像获取失败")
                        # 如果某图像失败，可以跳过（但会破坏输入数量），这里直接退出本次迭代
                        # 更稳健的做法是重试，但为简洁，跳过本次迭代
                        continue

                    h, w = resp.height, resp.width
                    data = resp.image_data_uint8

                    # 根据数据长度判断单通道（深度）或三通道（彩色）
                    if len(data) == h * w:           # 单通道灰度
                        img_gray = np.frombuffer(data, dtype=np.uint8).reshape(h, w)
                        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)   # 转为三通道
                        cv2.imwrite(f"{idx}.png", img_rgb)
                    elif len(data) == h * w * 3:     # 三通道彩色
                        img_rgb = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
                        cv2.imwrite(f"{idx}.png", img_rgb)
                    else:
                        print(f"警告：图像数据尺寸异常，跳过")
                        continue

                    data_url = encode_numpy_to_data_url(img_rgb)
                    image_urls.append({"type": "image_url", "image_url": {"url": data_url}})

                images_finished = time.perf_counter()
                # 如果获取的图像不足三张，重试或结束
                if len(image_urls) < 3:
                    print("未能获取三张完整图像，等待后重试...")
                    time.sleep(2)
                    continue
            
                # 目标提示（可根据需要修改）

                target = route["target"]

                target_prompt = build_target_prompt(target)
                # print(target_prompt)

                clip_started = time.perf_counter()
                if args.clip:
                    idx, score, patch_img, pos = find_best_patch(
                        "0.png", route["clip_text"], save_dir="./frontcam_seg")
                    target_prompt += f'为辅助你的判断：你的导航目标有很大的可能性在你的{pos}'
                clip_finished = time.perf_counter()
                # print(pos)
 

                # 3. 构造消息
                messages = [
                    {
                        "role": "user",
                        "content": image_urls + [{"type": "text", "text": target_prompt}]
                    }
                ]

                # 4. 调用公网 Qwen 大模型
                print("正在请求公网 Qwen 决策...")
                qwen_started = time.perf_counter()
                try:
                    completion = client_ai.chat.completions.create(
                        model="qwen3.5-35b-a3b",
                        messages=messages,
                        max_tokens=256,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                        timeout=150
                    )
                    action_str = completion.choices[0].message.content
                    print(f"模型输出: {action_str}")
                except Exception as e:
                    print(f"模型调用失败: {e}")
                    save_profile(profile_path, {"event": "qwen_error", "flight": flight_idx,
                                               "iteration": i + 1, "error": str(e)})
                    break
                qwen_finished = time.perf_counter()
                if args.profile:
                    inference_end = drone.client.getMultirotorState()
                    save_profile(profile_path, {
                        "event": "decision", "flight": flight_idx, "iteration": i + 1,
                        "action_text": action_str, "qwen_seconds": qwen_finished - qwen_started,
                        "inference_sim_seconds": (inference_end.timestamp - inference_state.timestamp) / 1e9,
                        "inference_drift_m": math.dist(list(inference_state.kinematics_estimated.position),
                                                       list(inference_end.kinematics_estimated.position)),
                    })

                # 5. 解析动作
                action, value = parse_action(action_str)
                if action is None:
                    print("无法解析动作，跳过本次迭代")
                    time.sleep(1)
                    continue

                # 6. 检查是否停止
                if action == 'stop':
                    print("模型指示已到达目标，导航结束。")
                    break

            # 7. 执行动作
            action_pose = drone.client.simGetVehiclePose() if args.profile else None
            action_sim_started = drone.client.getMultirotorState().timestamp if args.profile else None
            action_started = time.perf_counter()
            execute_action(drone, action, value)
            action_sim_finished = drone.client.getMultirotorState().timestamp if args.profile else None
            action_finished = time.perf_counter()

            # 等待稳定
            time.sleep(4.0 / clock_speed)
            

            with open(csv_path, 'a', newline='') as file:
                writer = csv.writer(file)
                pose = drone.client.simGetVehiclePose()
                writer.writerow([pose.position.x_val, pose.position.y_val, pose.position.z_val])
            if args.profile:
                state = drone.client.getMultirotorState()
                collision = drone.client.simGetCollisionInfo()
                save_profile(profile_path, {
                    "event": "step", "flight": flight_idx, "iteration": i + 1,
                    "action": action, "value": value, "clock_speed": clock_speed,
                    "images_seconds": images_finished - step_started,
                    "clip_seconds": clip_finished - clip_started,
                    "qwen_seconds": qwen_finished - qwen_started,
                    "action_seconds": action_finished - action_started,
                    "action_sim_seconds": (action_sim_finished - action_sim_started) / 1e9,
                    "wall_seconds": time.perf_counter() - step_started,
                    "sim_seconds": (state.timestamp - sim_started) / 1e9,
                    "position": [pose.position.x_val, pose.position.y_val, pose.position.z_val],
                    "action_start_position": [action_pose.position.x_val, action_pose.position.y_val, action_pose.position.z_val],
                    "action_start_orientation": list(action_pose.orientation),
                    "orientation": list(pose.orientation),
                    "linear_velocity": list(state.kinematics_estimated.linear_velocity),
                    "has_collided": collision.has_collided,
                    "collision_timestamp": collision.time_stamp,
                    "collision_object": collision.object_name,
                })

        # 导航结束，降落
        print("导航结束，准备降落...")
        drone.land()
        drone.cleanup()
        print("任务完成。")

        with open(csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            pose = drone.client.simGetVehiclePose()
            writer.writerow([pose.position.x_val, pose.position.y_val, pose.position.z_val])



if __name__ == "__main__":
    main()
