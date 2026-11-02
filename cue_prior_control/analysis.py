"""Offline route-matched analysis, requiring every prespecified formal episode."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import numpy as np
from . import design as d
from map1.scripts.heading_ablation import atomic_json, write_csv

_spec = importlib.util.spec_from_file_location('cue_frozen_scoring', d.SUPP/'reproduce_results.py')
_scoring = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scoring)
score, read_points, METRICS = _scoring.score, _scoring.read_points, _scoring.METRICS


def interval(route_values, samples, coverage=.95):
    estimates = np.asarray(route_values)[samples].mean(axis=1)
    alpha = (1-coverage)/2
    lo, hi = np.quantile(estimates, [alpha, 1-alpha], axis=0)
    mean = np.asarray(route_values).mean(axis=0)
    return {name: dict(mean=float(mean[i]), low=float(lo[i]), high=float(hi[i]), coverage=coverage)
            for i, name in enumerate(METRICS)}


def analyze(batch_dir):
    batch_dir = Path(batch_dir).resolve()
    output = batch_dir/'analysis'
    # Verify everything before publishing any numerical result.
    states = {}
    for map_name, spec in d.SPECS.items():
        folder = batch_dir/map_name
        state = json.loads((folder/'state.json').read_text(encoding='utf-8'))
        manifest = json.loads((folder/'input_manifest.json').read_text(encoding='utf-8'))
        for name, digest in manifest['sha256'].items():
            if d.sha256(d.ROOT/name) != digest:
                raise ValueError('Protected file changed: '+name)
        expected = {(r, k) for r in range(1, spec['routes']+1) for k in range(1, 11)}
        formal = [j for j in state['jobs'] if j['kind'] == 'formal']
        if {(j['route'], j['repeat']) for j in formal} != expected or len(formal) != len(expected):
            raise ValueError('Unexpected formal schedule')
        if not all(j['status'] == 'complete' for j in formal):
            raise ValueError('Formal batch incomplete; no final statistics produced')
        states[map_name] = state
    entries = d.historical_entries()
    origins = [(row, d.SUPP/row['trajectory_file']) for row in entries]
    integrity = dict(historical_episodes=880, new_formal_episodes=440, old_hashes_unchanged=True,
                     cue_checks=0, request_image_hash_checks=0, event_label_invariance_checks=0)
    for map_name, state in states.items():
        spec = d.SPECS[map_name]
        route_bank = json.loads((batch_dir/map_name/'routes.json').read_text(encoding='utf-8'))
        for job in state['jobs']:
            if job['kind'] != 'formal':
                continue
            result_path = batch_dir/map_name/job['result_path']
            folder = result_path.parent
            result = json.loads(result_path.read_text(encoding='utf-8'))
            if not result['complete'] or d.sha256(folder/'trajectory.csv') != result['trajectory_sha256']:
                raise ValueError('Incomplete or changed episode')
            route = route_bank[str(job['route'])]
            if job['cues'] != d.cue_sequence(spec['map_id'], job['route'], job['repeat'], spec['steps']):
                raise ValueError('Cue sequence differs from precommitment')
            for step in range(1, result['decisions']+1):
                step_folder = folder/f'step_{step:02d}'
                cue = json.loads((step_folder/'cue.json').read_text(encoding='utf-8'))
                request = json.loads((step_folder/'request.json').read_text(encoding='utf-8'))
                if cue['index'] != job['cues'][step-1] or request['prompt'] != d.prompt(map_name, route['target'], cue['index']):
                    raise ValueError('Transmitted cue/prompt mismatch')
                integrity['cue_checks'] += 1
                for item in request['images']:
                    if d.sha256(step_folder/item['file']) != item['sha256']:
                        raise ValueError('Request image changed')
                    integrity['request_image_hash_checks'] += 1
            row = dict(environment=spec['environment'], condition=d.CONDITION, route=str(job['route']),
                       repeat=str(job['repeat']), start=json.dumps(route['start_position']),
                       goal=json.dumps(route['goal_position']), radius_m=str(route['success_threshold']),
                       endpoint_metadata=json.dumps({'navigation_end_position': result['navigation_end_position']}),
                       sha256=result['trajectory_sha256'], original_radius_m=str(route['success_threshold']),
                       evaluation_version=route['evaluation_version'])
            origins.append((row, folder/'trajectory.csv'))
    output.mkdir(parents=True, exist_ok=True)
    matrices, episodes, provenance, manifest_rows = {}, [], [], []
    for row, path in origins:
        if d.sha256(path) != row['sha256']:
            raise ValueError('Source trajectory checksum mismatch')
        env, condition = row['environment'], row['condition']
        map_name = 'city' if env == 'City Park' else 'downtown'
        n = d.SPECS[map_name]['routes']
        r, k = int(row['route']), int(row['repeat'])
        matrix = matrices.setdefault((env, condition), np.full((n, 10, 5), np.nan))
        if np.isfinite(matrix[r-1, k-1]).any():
            raise ValueError('Duplicate episode')
        points, start, goal = read_points(path), json.loads(row['start']), json.loads(row['goal'])
        metadata, radius = json.loads(row['endpoint_metadata']), float(row['radius_m'])
        values = score(points, start, goal, radius, metadata)
        for label in ('collision', 'model_stop', 'max_iterations'):
            changed = score(points, start, goal, radius, dict(metadata, reason=label, has_collided=label == 'collision'))
            if not np.array_equal(values, changed):
                raise ValueError('Event label changed a metric')
            integrity['event_label_invariance_checks'] += 1
        matrix[r-1, k-1] = values
        episodes.append(dict(environment=env, condition=condition, route=r, repeat=k, **dict(zip(METRICS, map(float, values)))))
        arm = 'prior' if condition == d.CONDITION else 'actual' if condition == d.ACTUAL else 'none'
        relative = Path('data')/map_name/arm/f'r{r:02d}_e{k:02d}.csv'
        target = output/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        manifest_rows.append(dict(row, trajectory_file=relative.as_posix()))
        provenance.append(dict(environment=env, condition=condition, route=r, repeat=k,
                               original_path=str(path.relative_to(d.ROOT)), sha256=row['sha256']))
    if len(episodes) != 1320 or not all(np.isfinite(m).all() for m in matrices.values()):
        raise ValueError('Incomplete group')
    results = dict(episodes=1320, new_episodes=440, bootstrap_resamples=10000,
                   independent_units='routes; ten repeats nested within each condition and route',
                   primary='actual minus uniform-prior SR per map, 97.5% intervals (two-map Bonferroni)',
                   secondary='other metrics/contrasts: exploratory 95% intervals',
                   interpretation='historical cross-batch route-matched association; no causal randomization claim',
                   environments={})
    route_rows, plot_rows, differences = [], [], []
    for map_name, spec in d.SPECS.items():
        env, n = spec['environment'], spec['routes']
        rng = np.random.default_rng(np.random.SeedSequence([d.SEED, 3, spec['map_id']]))
        samples = rng.integers(0, n, (10000, n))
        means = {c: matrices[(env, c)].mean(axis=1) for c in (d.ACTUAL, d.NONE, d.CONDITION)}
        data = dict(routes=n, repeats=10, methods={c: interval(m, samples) for c, m in means.items()}, differences={})
        for a, b, label in ((d.ACTUAL, d.CONDITION, 'actual minus prior'),
                            (d.CONDITION, d.NONE, 'prior minus no cue'),
                            (d.ACTUAL, d.NONE, 'actual minus no cue')):
            result = interval(means[a]-means[b], samples)
            if label == 'actual minus prior':
                result['SR'] = interval(means[a]-means[b], samples, .975)['SR']
            data['differences'][label] = result
            for i in range(n):
                differences.append(dict(environment=env, contrast=label, route=i+1,
                                        **dict(zip(METRICS, map(float, means[a][i]-means[b][i])))))
        results['environments'][env] = data
        for condition, m in means.items():
            for i in range(n):
                route_rows.append(dict(environment=env, condition=condition, route=i+1, **dict(zip(METRICS, map(float, m[i])))))
            for metric, values in data['methods'][condition].items():
                scale = 100 if metric in ('SR', 'SPL', 'OR') else 1
                plot_rows.append(dict(environment=env, condition=condition, metric=metric,
                                      **{key: values[key]*scale for key in ('mean', 'low', 'high')}))
    atomic_json(output/'results.json', results)
    atomic_json(output/'integrity.json', integrity)
    atomic_json(output/'provenance.json', provenance)
    for name, rows in (('episodes_manifest.csv', manifest_rows), ('episode_metrics.csv', episodes),
                       ('routewise_metrics.csv', route_rows), ('routewise_differences.csv', differences), ('plot_ci.csv', plot_rows)):
        write_csv(output/name, rows)
    print('Analysis complete: 1320 episodes (440 new), 44 route clusters; results: '+str(output), flush=True)
    return results
