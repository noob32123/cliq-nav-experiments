"""Replay all 1,320 cue-control episodes and route-matched bootstrap intervals.

Python 3.10+ and NumPy; no simulator, model service or network access required.
Default output is a separate replay directory, preserving the published files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from reproduce_results import METRICS, read_points, score

ACTUAL = 'Qwen3.5-35B-A3B + CLIP'
PRIOR = 'Qwen3.5-35B-A3B + uniform direction prior'
NONE = 'Qwen3.5-35B-A3B'
SEED = 20261027


def interval(values, samples, coverage=.95):
    estimates = values[samples].mean(axis=1)
    alpha = (1 - coverage) / 2
    low, high = np.quantile(estimates, [alpha, 1-alpha], axis=0)
    mean = values.mean(axis=0)
    return {name: dict(mean=float(mean[i]), low=float(low[i]), high=float(high[i]), coverage=coverage)
            for i, name in enumerate(METRICS)}


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(folder, output):
    folder, output = Path(folder).resolve(), Path(output).resolve()
    with (folder/'episodes_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        entries = list(csv.DictReader(stream))
    assert len(entries) == 1320
    matrices, episode_rows = {}, []
    for row in entries:
        env, condition = row['environment'], row['condition']
        assert env in ('City Park', 'Downtown West') and condition in (ACTUAL, PRIOR, NONE)
        n = 14 if env == 'City Park' else 30
        r, k = int(row['route']), int(row['repeat'])
        assert 1 <= r <= n and 1 <= k <= 10
        matrix = matrices.setdefault((env, condition), np.full((n, 10, 5), np.nan))
        assert np.isnan(matrix[r-1, k-1]).all(), 'Duplicate episode'
        path = folder/row['trajectory_file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], 'Trajectory changed'
        metadata = json.loads(row['endpoint_metadata'])
        points = read_points(path)
        start, goal, radius = json.loads(row['start']), json.loads(row['goal']), float(row['radius_m'])
        values = score(points, start, goal, radius, metadata)
        for reason in ('collision', 'model_stop', 'max_iterations'):
            changed = score(points, start, goal, radius,
                            dict(metadata, reason=reason, has_collided=reason == 'collision'))
            assert np.array_equal(values, changed), 'Event label changed a score'
        matrix[r-1, k-1] = values
        episode_rows.append(dict(environment=env, condition=condition, route=r, repeat=k,
                                 **dict(zip(METRICS, map(float, values)))))
    assert len(matrices) == 6 and all(np.isfinite(m).all() for m in matrices.values())
    result = dict(episodes=1320, new_episodes=440, bootstrap_resamples=10000,
                  independent_units='routes; ten repeats nested within each condition and route',
                  primary='actual minus uniform-prior SR per map, 97.5% intervals (two-map Bonferroni)',
                  secondary='other metrics/contrasts: exploratory 95% intervals',
                  environments={})
    route_rows, difference_rows, plot_rows = [], [], []
    for map_id, (env, n) in enumerate((('City Park', 14), ('Downtown West', 30)), 1):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 3, map_id]))
        samples = rng.integers(0, n, (10000, n))
        means = {c: matrices[(env, c)].mean(axis=1) for c in (ACTUAL, NONE, PRIOR)}
        data = dict(routes=n, repeats=10,
                    methods={c: interval(m, samples) for c, m in means.items()}, differences={})
        for a, b, label in ((ACTUAL, PRIOR, 'actual minus prior'),
                            (PRIOR, NONE, 'prior minus no cue'),
                            (ACTUAL, NONE, 'actual minus no cue')):
            delta = means[a]-means[b]
            estimates = interval(delta, samples)
            if label == 'actual minus prior':
                estimates['SR'] = interval(delta, samples, .975)['SR']
            data['differences'][label] = estimates
            for i in range(n):
                difference_rows.append(dict(environment=env, contrast=label, route=i+1,
                                            **dict(zip(METRICS, map(float, delta[i])))))
        result['environments'][env] = data
        for condition, values in means.items():
            for i in range(n):
                route_rows.append(dict(environment=env, condition=condition, route=i+1,
                                       **dict(zip(METRICS, map(float, values[i])))))
            for metric, values in data['methods'][condition].items():
                scale = 100 if metric in ('SR', 'SPL', 'OR') else 1
                plot_rows.append(dict(environment=env, condition=condition, metric=metric,
                                      **{key: values[key]*scale for key in ('mean', 'low', 'high')}))
    output.mkdir(parents=True, exist_ok=True)
    (output/'results.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    for name, rows in (('episode_metrics.csv', episode_rows), ('routewise_metrics.csv', route_rows),
                       ('routewise_differences.csv', difference_rows), ('plot_ci.csv', plot_rows)):
        write_csv(output/name, rows)
    archived = json.loads((folder/'results.json').read_text(encoding='utf-8'))
    assert result['environments'] == archived['environments'], 'Published statistics do not reproduce'
    print('Verified 1320 episodes, six complete groups, all means and all pairwise intervals.')


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=here/'cue_prior_control')
    parser.add_argument('--output', type=Path, default=here/'cue_prior_control/replay')
    args = parser.parse_args()
    run(args.input, args.output)
