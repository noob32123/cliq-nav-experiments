"""Reproduce the five endpoint/trajectory metrics and route-clustered intervals.

Requires Python 3.10+ and NumPy. Run from any directory:
  python reproduce_results.py --manifest episodes_manifest.csv --output results
No simulator, model service, event labels, or network access is required.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import hashlib
from pathlib import Path
import numpy as np

METRICS = ('SR', 'NE', 'SPL', 'OR', 'TL')
WEIGHTS = np.asarray([.10, .15, .20, .25, .30])

def wrap(degrees):
    if not math.isfinite(degrees):
        raise ValueError('Non-finite heading')
    return (degrees + 180.) % 360. - 180.

def heading(points, current_yaw):
    p = np.asarray(points, dtype=float)
    if p.shape != (6, 3) or not np.isfinite(p).all():
        raise ValueError('Expected six finite XYZ history points')
    segments = np.diff(p[:, :2], axis=0)
    weighted = WEIGHTS @ segments
    latest = segments[-1]
    if min(np.linalg.norm(weighted), np.linalg.norm(latest)) < 1e-5:
        raise ValueError('Heading undefined for a near-zero vector')
    predicted = math.degrees(math.atan2(weighted[1], weighted[0]))
    last = math.degrees(math.atan2(latest[1], latest[0]))
    return {'predicted_yaw_deg': predicted, 'last_yaw_deg': last,
            'weighted_turn_deg': wrap(predicted-current_yaw),
            'last_minus_weighted_deg': wrap(last-predicted)}

def score(points, start, goal, radius, metadata=None):
    """Metadata may supply a saved navigation endpoint; other fields are unused."""
    xyz = np.asarray(points, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('Expected a nonempty finite XYZ trajectory')
    xy = xyz[:, :2]
    if metadata and metadata.get('navigation_end_position') is not None:
        endpoint = np.asarray(metadata['navigation_end_position'][:2], dtype=float)
        hits = np.flatnonzero(np.linalg.norm(xy-endpoint, axis=1) < 1e-6)
        if not len(hits):
            raise ValueError('Saved navigation endpoint is absent from the trajectory')
        xy = xy[:hits[-1]+1]
    goal, start = np.asarray(goal[:2], dtype=float), np.asarray(start[:2], dtype=float)
    ne = float(np.linalg.norm(xy[-1]-goal))
    length = float(np.linalg.norm(start-goal))
    path = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    success = float(ne <= radius)
    spl = success*length/max(path,length) if max(path,length) else 0.
    oracle = float(np.linalg.norm(xy-goal, axis=1).min() <= radius)
    return np.asarray([success,ne,spl,oracle,path])

def record(matrix, samples):
    routes = matrix.mean(axis=1)
    mean = routes.mean(axis=0)
    lo, hi = np.quantile(routes[samples].mean(axis=1), [.025,.975], axis=0)
    return {name: {'mean':float(mean[i]),'low':float(lo[i]),'high':float(hi[i])}
            for i,name in enumerate(METRICS)}

def read_points(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return [[float(row[k]) for k in ('X','Y','Z')] for row in csv.DictReader(f)]

def run(manifest, output):
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    with manifest.open(encoding='utf-8',newline='') as f:
        entries = list(csv.DictReader(f))
    groups, episode_rows = {}, []
    for row in entries:
        env, condition = row['environment'], row['condition']
        route, repeat = int(row['route']), int(row['repeat'])
        n = 14 if env == 'City Park' else 30
        key = (env,condition)
        matrix = groups.setdefault(key,np.full((n,10,5),np.nan))
        if np.isfinite(matrix[route-1,repeat-1]).any():
            raise ValueError(f'Duplicate episode {key} {route} {repeat}')
        metadata = json.loads(row['endpoint_metadata'])
        if hashlib.sha256((manifest.parent/row['trajectory_file']).read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Trajectory checksum mismatch: '+row['trajectory_file'])
        values = score(read_points(manifest.parent/row['trajectory_file']),
                       json.loads(row['start']),json.loads(row['goal']),float(row['radius_m']),metadata)
        matrix[route-1,repeat-1] = values
        episode_rows.append({'environment':env,'condition':condition,'route':route,'repeat':repeat,
                             **dict(zip(METRICS,map(float,values)))})
    assert all(np.isfinite(m).all() for m in groups.values()), 'Missing episode or invalid metric'
    result = {'metrics':METRICS,'bootstrap_resamples':10000,'episodes':len(entries),
              'metric_inputs':'endpoint and XY trajectory; event labels have no scoring role',
              'environments':{},'heading':{}}
    versions = {row.get('evaluation_version') for row in entries}
    if versions != {None}:
        assert len(versions) == 1 and None not in versions
        result['evaluation_version'] = versions.pop()
        result['manifest_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    route_rows=[]
    for env,n,seed in [('City Park',14,20260823),('Downtown West',30,20261014)]:
        samples=np.random.default_rng(seed).integers(0,n,(10000,n))
        methods={k[1]:m for k,m in groups.items() if k[0]==env and k[1]!='Last-segment heading'}
        differences={}
        for cue,control in [('Qwen3.5-35B-A3B + CLIP','Qwen3.5-35B-A3B'),('Qwen3.5-Flash + CLIP','Qwen3.5-Flash')]:
            if cue in methods and control in methods:
                differences[cue+' minus '+control]=record(methods[cue]-methods[control],samples)
        result['environments'][env]={'routes':n,'repeats':10,'seed':seed,
            'methods':{k:record(m,samples) for k,m in methods.items()},'differences':differences}
        heading_samples=np.random.default_rng(20260823).integers(0,n,(10000,n))
        full=groups[(env,'Qwen3.5-35B-A3B + CLIP')]
        last=groups[(env,'Last-segment heading')]
        result['heading'][env]={'seed':20260823,'methods':{'historical_full':record(full,heading_samples),
            'last_segment':record(last,heading_samples)},'difference':record(full-last,heading_samples)}
        for (environment,condition),matrix in groups.items():
            if environment!=env:continue
            for i,values in enumerate(matrix.mean(axis=1)):
                route_rows.append({'environment':env,'condition':condition,'route':i+1,
                                  **dict(zip(METRICS,map(float,values)))})
    output.mkdir(parents=True,exist_ok=True)
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    for name,rows in [('episode_metrics.csv',episode_rows),('routewise_metrics.csv',route_rows)]:
        with (output/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    plot_rows=[]
    for env, data in result['environments'].items():
        for condition, metrics in data['methods'].items():
            for metric, values in metrics.items():
                scale=100 if metric in ('SR','SPL','OR') else 1
                plot_rows.append({'environment':env,'condition':condition,'metric':metric,
                                  **{k:v*scale for k,v in values.items()}})
    with (output/'plot_ci.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(plot_rows[0]));w.writeheader();w.writerows(plot_rows)
    return result

def self_test():
    path=[[0,0,0],[5,0,-2],[10,0,-2]]
    baseline=score(path,[0,0,0],[10,0,0],1)
    a=score(path,[0,0,0],[10,0,0],1,{'reason':'collision','has_collided':True})
    b=score(path,[0,0,0],[10,0,0],1,{'reason':'model_stop','has_collided':False})
    assert np.array_equal(a,b) and np.array_equal(a,baseline)
    assert np.array_equal(baseline,[1,0,1,1,10])
    landed=path+[[10,0,0],[18,0,0]]
    for label in ['collision','model_stop','step_limit']:
        saved=score(landed,[0,0,0],[10,0,0],1,
            {'navigation_end_position':[10,0,-2],'reason':label,'has_collided':label=='collision'})
        assert np.array_equal(saved,baseline)
    # A route enters the goal region and then leaves it: OR differs from SR.
    c=score(path+[[20,0,-2]],[0,0,0],[10,0,0],1)
    assert c[0]==0 and c[2]==0 and c[3]==1 and c[4]==20
    for current,predicted,expected in [(0,90,90),(90,0,-90),(179,-179,2),(-179,179,-2),(90,90,0)]:
        assert abs(wrap(predicted-current)-expected)<1e-10
    assert wrap(180)==-180 and wrap(-180)==-180
    for target in [0,90,-90,179,-179]:
        vector=np.asarray([math.cos(math.radians(target)),math.sin(math.radians(target)),0.])
        points=np.arange(6)[:,None]*vector
        h=heading(points,35.)
        assert abs(wrap(h['predicted_yaw_deg']-target))<1e-10
        assert abs(h['weighted_turn_deg']-wrap(target-35.))<1e-10
    for points in [np.zeros((6,3)),np.full((6,3),np.nan)]:
        try:heading(points,0.)
        except ValueError:pass
        else:raise AssertionError('Invalid history accepted')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=Path(__file__).with_name('episodes_manifest.csv'))
    p.add_argument('--output',type=Path,default=Path(__file__).with_name('results'))
    args=p.parse_args();self_test();r=run(args.manifest,args.output)
    print(f'Reproduced {r["episodes"]} episodes, all five metrics and route-clustered intervals; self-tests passed.')
