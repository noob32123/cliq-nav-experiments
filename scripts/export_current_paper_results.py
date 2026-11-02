"""Export and display only the current paper's revised-radius results.

No statistics are refitted. The current benchmark, unified cue and heading
results retain the bootstrap streams used in their manuscript comparisons.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUPP = ROOT / 'supplementary'
METRICS = ('SR', 'NE', 'SPL', 'OR', 'TL')
ACTUAL = 'Qwen3.5-35B-A3B + CLIP'
NONE = 'Qwen3.5-35B-A3B'
RANDOM = 'Qwen3.5-35B-A3B + uniform direction prior'
REVISED_ROUTES = {9, 14, 16, 17, 21, 25, 29}
START = '<!-- CURRENT_PAPER_RESULTS_BEGIN -->'
END = '<!-- CURRENT_PAPER_RESULTS_END -->'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def scaled(value, metric):
    return value * (100 if metric in ('SR', 'SPL', 'OR') else 1)


def check_radii():
    routes = read_csv(SUPP/'routes.csv')
    assert len(routes) == 44
    radii = {}
    for row in routes:
        number = int(row['route'])
        if row['environment'] == 'Downtown West':
            expected = 3. if number in REVISED_ROUTES else 6.
        else:
            expected = {3: 15., 5: 20.}.get(number, 10.)
        assert float(row['radius_m']) == expected, 'Unexpected current radius: ' + str(row)
        radii[(row['environment'], row['route'])] = expected
    for manifest in (SUPP/'episodes_manifest.csv', SUPP/'cue_prior_control/episodes_manifest.csv'):
        for row in read_csv(manifest):
            assert row['evaluation_version'] == 'revised_v2'
            assert float(row['radius_m']) == radii[(row['environment'], row['route'])]


def metric_row(env, condition, stats, source, comparison='benchmark'):
    routes = 14 if env == 'City Park' else 30
    row = dict(environment=env, comparison=comparison, condition=condition,
               evaluation_version='revised_v2', routes=routes, episodes=routes*10,
               result_source=source)
    for metric in METRICS:
        for field in ('mean', 'low', 'high'):
            row[f'{metric}_{field}'] = scaled(stats[metric][field], metric)
    return row


def csv_text(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def build():
    check_radii()
    base = json.loads((SUPP/'results/results.json').read_text(encoding='utf-8'))
    cue = json.loads((SUPP/'cue_prior_control/results.json').read_text(encoding='utf-8'))
    assert base['evaluation_version'] == 'revised_v2'
    summary, heading, differences = [], [], []
    for env in ('City Park', 'Downtown West'):
        methods = list(base['environments'][env]['methods']) + [RANDOM]
        for condition in methods:
            if condition in (ACTUAL, NONE, RANDOM):
                stats = cue['environments'][env]['methods'][condition]
                source = 'supplementary/cue_prior_control/results.json'
                if condition != RANDOM:
                    assert all(math.isclose(stats[m]['mean'], base['environments'][env]['methods'][condition][m]['mean'], abs_tol=1e-12) for m in METRICS)
            else:
                stats = base['environments'][env]['methods'][condition]
                source = 'supplementary/results/results.json'
            summary.append(metric_row(env, condition, stats, source))
        for condition, key in [('Unablated (weighted-history heading)', 'historical_full'),
                               ('Ablated (last-segment heading)', 'last_segment')]:
            row = metric_row(env, condition, base['heading'][env]['methods'][key],
                'supplementary/results/results.json#heading', 'heading')
            heading.append(row)
            if key == 'last_segment':
                unique = dict(row, condition='Last-segment heading')
                summary.append(unique)
        for metric in METRICS:
            r = base['heading'][env]['difference'][metric]
            differences.append(dict(environment=env, contrast='Unablated minus Ablated',
                metric=metric, mean=scaled(r['mean'], metric), low=scaled(r['low'], metric),
                high=scaled(r['high'], metric), coverage=.95,
                unit='percentage points' if metric in ('SR', 'SPL', 'OR') else 'm',
                evaluation_version='revised_v2'))
    assert len(summary) == 19 and sum(row['episodes'] for row in summary) == 3780
    sources = ['supplementary/routes.csv', 'supplementary/episodes_manifest.csv',
               'supplementary/cue_prior_control/episodes_manifest.csv',
               'supplementary/results/results.json', 'supplementary/cue_prior_control/results.json']
    metadata = dict(evaluation_version='revised_v2', unique_episodes=3780,
        city_park_radii_m={'03': 15, '05': 20, 'all_other_routes': 10},
        downtown_radii_m={'3_m_routes': sorted(REVISED_ROUTES), 'all_other_routes': 6},
        radius_revision_timing='During manuscript revision, after original outcomes were available.',
        units={'SR': 'percent', 'NE': 'm', 'SPL': 'percent', 'OR': 'percent', 'TL': 'm'},
        marginal_confidence_level=.95, heading_difference_confidence_level=.95,
        interval_sources='Actual/no/random cue: unified three-condition streams; other benchmark conditions: original benchmark seeds; heading: seed 20260823.',
        heading_full_rows_reuse_benchmark_episodes=True,
        historical_6m_results='Retained separately for optional sensitivity analysis; not displayed in the current results.',
        source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources})
    return summary, heading, differences, metadata


def display_name(condition):
    return 'Qwen3.5-35B-A3B + random cue' if condition == RANDOM else condition


def formatted(row):
    return [f'{row[m+"_mean"]:.{1 if m in ("SR", "SPL", "OR") else 2}f}' for m in METRICS]


def readme_block(summary, heading):
    lines = [START, '## Current paper results (revised success radii)', '',
        '**The tables below use the current KBS manuscript scoring version, `revised_v2`.**', '',
        '- **City Park:** 10 m, except route 03 (15 m) and route 05 (20 m).',
        '- **Downtown West:** **3 m on routes 09, 14, 16, 17, 21, 25 and 29; 6 m on all other routes.**', '',
        'The same route-specific rule applies to every method and heading condition. SR, SPL and OR are percentages; NE and TL are metres. Values are means, with percentages rounded to one decimal and distances to two. The radius change occurred during manuscript revision, after the original outcomes were available.', '']
    for env, n in [('City Park', 140), ('Downtown West', 300)]:
        lines += [f'### {env}: benchmark and cue conditions', '', f'Each row contains {n} episodes (ten per route).', '',
                  '| Method / condition | SR (%) | NE (m) | SPL (%) | OR (%) | TL (m) |',
                  '|---|---:|---:|---:|---:|---:|']
        for row in summary:
            if row['environment'] == env and row['comparison'] == 'benchmark':
                lines.append('| ' + ' | '.join([display_name(row['condition']), *formatted(row)]) + ' |')
        lines.append('')
    lines += ['### Heading ablation', '',
        'Both conditions retain Qwen3.5-35B-A3B and the actual CLIP cue. The unablated rows reuse the full-method benchmark episodes and are not additional experiments.', '',
        '| Environment | Heading condition | SR (%) | NE (m) | SPL (%) | OR (%) | TL (m) |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in heading:
        lines.append('| ' + ' | '.join([row['environment'],row['condition'],*formatted(row)]) + ' |')
    lines += ['',
        'Full-precision means and marginal 95% intervals: [current summary](supplementary/current_paper_results/summary.csv), [heading conditions](supplementary/current_paper_results/heading.csv), and [heading differences](supplementary/current_paper_results/heading_differences.csv). The summary covers all 19 distinct formal condition/map groups (3,780 episodes).', '',
        'The cue rows retain the unified three-condition bootstrap, and heading comparisons retain their own manuscript bootstrap stream. The two primary actual-minus-random SR contrasts use 97.5% intervals, available in `supplementary/cue_prior_control/results.json`; these differ from the marginal 95% intervals in the summary CSV.', '',
        '**Original uniform-6-m outputs are historical sensitivity records. They are not the displayed current results.** Historical route-layout summaries and frozen batch settings remain unchanged for provenance; use the current result files linked above for manuscript values.', '',
        'Regenerate these CSV files and this README section with:', '',
        '```bash', 'python -B scripts/export_current_paper_results.py --update-readme', '```', END]
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=SUPP/'current_paper_results')
    parser.add_argument('--update-readme', action='store_true')
    args = parser.parse_args()
    summary, heading, differences, metadata = build()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in [('summary', summary), ('heading', heading), ('heading_differences', differences)]:
        (args.output/(name+'.csv')).write_text(csv_text(rows), encoding='utf-8', newline='\n')
    (args.output/'scoring_version.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    if args.update_readme:
        path=ROOT/'README.md'
        text=path.read_text(encoding='utf-8')
        block=readme_block(summary,heading)
        if START in text:
            start=text.index(START)
            end=text.index(END,start)+len(END)
            text=text[:start]+block.rstrip('\n')+text[end:]
        else:
            text=text.replace('## Experiment coverage',block+'\n## Experiment coverage',1)
        path.write_text(text,encoding='utf-8',newline='\n')
    print('Exported revised_v2 results: 19 formal condition/map groups, 3780 distinct episodes.')
    print('Output: '+str(args.output.resolve()))


if __name__ == '__main__':
    main()
