"""Frozen experimental design, without simulator, GPU, or network side effects."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT
SUPP = PAPER / 'supplementary'
MODEL = 'qwen3.5-35b-a3b'
ENDPOINT = 'https://api.desire-drive.com/v1'
SEED = 20261027
CONDITION = 'Qwen3.5-35B-A3B + uniform direction prior'
ACTUAL = 'Qwen3.5-35B-A3B + CLIP'
NONE = 'Qwen3.5-35B-A3B'
POSITIONS = dict(enumerate(('前上方偏左', '正前上方', '前上方偏右', '前方偏左', '正前方',
                            '前方偏右', '前下方偏左', '正前下方', '前下方偏右'), 1))
SPECS = {
    'city': dict(environment='City Park', map_id=1, routes=14, steps=20, wait=4., pause=False,
                 module='map1.scripts.run_heading_ablation', port=41452,
                 route_source='map1/heading_ablation/20261015_heading_prediction_corrected_frame/routes.json',
                 frame='map1/heading_ablation/citypark_saved_default_frame.json',
                 settings='map1/heading_ablation/20261015_heading_prediction_corrected_frame/settings.json',
                 prompt='city_park_prompt_zh.txt', project=os.environ.get('CITYPARK_PROJECT', 'H:/CityParkEnvironmentCollec/CityPark.uproject')),
    'downtown': dict(environment='Downtown West', map_id=2, routes=30, steps=30, wait=.8, pause=True,
                     module='map2.scripts.run_heading_ablation', port=41453,
                     route_source='map2/heading_ablation/20261016_heading_prediction/routes.json',
                     frame='map2/heading_ablation/downtown_scene_frame.json',
                     settings='map2/formal_20260909_170427/settings.json',
                     prompt='downtown_west_prompt_zh.txt',
                     project=os.environ.get('DOWNTOWN_PROJECT', 'H:/airsimdowntown/airsimdowntown/airsimdowntown.uproject')),
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def cue_sequence(map_id, route, repeat, steps, kind='formal'):
    """Independent stream per episode; deliberately accepts no image/CLIP argument."""
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 1, map_id, route, repeat,
                                                       int(kind == 'smoke')]))
    return rng.integers(1, 10, size=steps).tolist()


def routes(map_name):
    spec = SPECS[map_name]
    source = json.loads((ROOT/spec['route_source']).read_text(encoding='utf-8'))
    scoring = [r for r in read_csv(SUPP/'routes.csv') if r['environment'] == spec['environment']]
    assert len(scoring) == spec['routes']
    output = {}
    for row in scoring:
        number = int(row['route'])
        r = source[str(number)]
        output[str(number)] = {
            'route': number, 'name': row['target'], 'start_position': json.loads(row['start']),
            'goal_position': json.loads(row['goal']), 'success_threshold': float(row['radius_m']),
            'reference_length_m': float(row['reference_length_m']), 'altitude': r['altitude'],
            'initial_yaw_deg': r.get('initial_yaw_deg', 0.),
            'baseline_turn_deg': r['baseline_turn_deg'],
            'target': row['target_text_for_mllm'], 'clip_text': row['target_text_for_clip'],
            'evaluation_version': row['evaluation_version'],
        }
    return output


def jobs(map_name, route_bank):
    spec = SPECS[map_name]
    ordered = sorted(route_bank.values(), key=lambda r: (r['reference_length_m'], r['route']))
    smoke = [ordered[i]['route'] for i in (0, len(ordered)//2, len(ordered)-1)]
    rows = []
    def add(kind, number, repeat):
        rows.append(dict(id=f'{kind}_{number:02d}_{repeat:02d}', kind=kind, route=number,
                         repeat=repeat, status='pending', attempts=0,
                         cues=cue_sequence(spec['map_id'], number, repeat,
                                           2 if kind == 'smoke' else spec['steps'], kind)))
    for number in smoke:
        add('smoke', number, 1)
    order_rng = np.random.default_rng(np.random.SeedSequence([SEED, 2, spec['map_id']]))
    for repeat in range(1, 11):
        for number in order_rng.permutation(np.arange(1, spec['routes']+1)):
            add('formal', int(number), repeat)
    return rows


def prompt(map_name, target, index):
    # Strip the file newline to reproduce the historical concatenation exactly.
    template = (SUPP/'interfaces'/SPECS[map_name]['prompt']).read_text(encoding='utf-8').rstrip('\r\n')
    return template.replace('{TARGET}', target) + f'为辅助你的判断：你的导航目标有很大的可能性在你的{POSITIONS[index]}'


def historical_entries():
    rows = [r for r in read_csv(SUPP/'episodes_manifest.csv') if r['condition'] in (ACTUAL, NONE)]
    assert len(rows) == 880
    assert len({(r['environment'], r['condition'], r['route'], r['repeat']) for r in rows}) == 880
    return rows
