"""Verify the complete public archive and reproduce both scoring versions and cue contrasts."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SUPP = ROOT / 'supplementary'

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def verify_archive():
    provenance = json.loads((SUPP/'publication_provenance.json').read_text(encoding='utf-8'))
    for name, record in provenance['files'].items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record['published_sha256']:
            raise ValueError('Published file changed or missing: ' + name)
    base = read_csv(SUPP/'episodes_manifest.csv')
    cue = read_csv(SUPP/'cue_prior_control/episodes_manifest.csv')
    assert len(base) == 3340 and len(cue) == 1320
    bank = {}
    for rows, folder in ((base, SUPP), (cue, SUPP/'cue_prior_control')):
        seen = set()
        groups = Counter()
        for row in rows:
            identity = tuple(row[k] for k in ('environment','condition','route','repeat'))
            assert identity not in seen, 'Duplicate episode identity'
            seen.add(identity)
            groups[identity[:2]] += 1
            routes = 14 if row['environment'] == 'City Park' else 30
            assert 1 <= int(row['route']) <= routes and 1 <= int(row['repeat']) <= 10
            assert hashlib.sha256((folder/row['trajectory_file']).read_bytes()).hexdigest() == row['sha256']
            if identity in bank:
                previous = bank[identity]
                assert all(previous[k] == row[k] for k in ('sha256','start','goal','radius_m','endpoint_metadata'))
            else:
                bank[identity] = row
        assert all(n == (140 if env == 'City Park' else 300) for (env,_),n in groups.items())
    assert len(bank) == 3780 and len({r['condition'] for r in base}) == 11
    assert len(read_csv(SUPP/'routes.csv')) == 44
    audits = read_csv(SUPP/'heading_episode_audit.csv')
    assert len(audits) == 440
    assert len(read_csv(SUPP/'heading_route_audit.csv')) == 44
    print(f'Integrity verified: {len(provenance["files"])} files, 3780 distinct episodes, 440 heading checks.', flush=True)

def same(expected, actual, location='result'):
    if isinstance(expected, dict):
        assert set(expected) == set(actual), location + ': object fields differ'
        for key in expected:
            same(expected[key], actual[key], location+'/'+key)
    elif isinstance(expected, list):
        assert len(expected) == len(actual), location + ': lengths differ'
        for i,(a,b) in enumerate(zip(expected,actual)):
            same(a,b,f'{location}/{i}')
    elif isinstance(expected, (int,float)):
        assert math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-10), location + ': number differs'
    else:
        assert expected == actual, location + ': value differs'

def compare(saved, generated):
    a,b = (json.loads(p.read_text(encoding='utf-8')) for p in (saved,generated))
    for key in ('environments','heading'):
        if key in a:
            same(a[key],b[key],key)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reproduced_results')
    args = parser.parse_args()
    output = args.output.resolve()
    assert output != SUPP and not output.is_relative_to(SUPP), 'Choose an output outside the published supplementary data'
    verify_archive()
    for manifest, saved, target in (
        ('episodes_manifest.csv','results','current'),
        ('episodes_manifest_original_6m.csv','results_original_6m','original_6m')):
        subprocess.run([sys.executable,'-B',str(SUPP/'reproduce_results.py'),
            '--manifest',str(SUPP/manifest),'--output',str(output/target)],check=True)
        compare(SUPP/saved/'results.json',output/target/'results.json')
    subprocess.run([sys.executable,'-B',str(SUPP/'reproduce_cue_control.py'),
        '--output',str(output/'cue_control')],check=True)
    compare(SUPP/'cue_prior_control/results.json',output/'cue_control/results.json')
    a = read_csv(output/'current/episode_metrics.csv')
    b = read_csv(output/'original_6m/episode_metrics.csv')
    assert len(a) == len(b) == 3340
    for x,y in zip(a,b):
        assert all(x[k] == y[k] for k in ('environment','condition','route','repeat','NE','TL'))
    print('All published means, interval bounds, cue contrasts and radius-sensitivity checks reproduced.')
    print('Output: '+str(output))

if __name__ == '__main__':
    main()
