"""python -m cue_prior_control.runner --map all --batch-dir PATH --run

Modes: --preflight, --smoke, --run, --resume (with a mode), --analyze.
Only the uniform-prior condition is flown. Existing baselines are never rerun.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

sys.dont_write_bytecode = True
from . import design as d
from map1.scripts.heading_ablation import atomic_json, write_csv, wrap


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


class PauseBatch(RuntimeError):
    pass


class CueBatch:
    """First MRO entry; inherits only tested simulator primitives from each map."""
    def __init__(self, args, map_name):
        self.args, self.map_name = args, map_name
        self.spec = d.SPECS[map_name]
        self.directory = (args.batch_dir/map_name).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.directory/'state.json'
        self.engine = self.drone = self.ai = None
        self.context = {}
        self.args.rpc_port = self.spec['port']
        if self.state_path.exists():
            if not args.resume:
                raise PauseBatch('Batch exists; explicitly use --resume')
            self.state = json.loads(self.state_path.read_text(encoding='utf-8'))
            self.routes = json.loads((self.directory/'routes.json').read_text(encoding='utf-8'))
            self.verify_manifest()
            for job in self.state['jobs']:
                for path in sorted((self.directory/'attempts'/job['id']).glob('*/result.json')):
                    result = json.loads(path.read_text(encoding='utf-8'))
                    if result.get('complete'):
                        if d.sha256(path.parent/'trajectory.csv') != result['trajectory_sha256']:
                            raise PauseBatch('Completed trajectory checksum mismatch')
                        if job['status'] == 'complete' and job['result_path'] != str(path.relative_to(self.directory)):
                            raise PauseBatch('Multiple completed results for one job')
                        job.update(status='complete', result_path=str(path.relative_to(self.directory)))
                if job['status'] != 'complete':
                    job['status'] = 'pending'
        else:
            self.routes = d.routes(map_name)
            frame = json.loads((d.ROOT/self.spec['frame']).read_text(encoding='utf-8'))
            config = dict(protocol='uniform-prior-v1', map=map_name, environment=self.spec['environment'],
                          model=d.MODEL, endpoint=d.ENDPOINT, rpc_port=self.spec['port'],
                          seed=d.SEED, cue_distribution={str(k): 1/9 for k in d.POSITIONS},
                          max_iterations=self.spec['steps'], post_action_wall_seconds=self.spec['wait'],
                          pause_during_inference=self.spec['pause'], clock_speed=5,
                          horizontal_speed=2, vertical_speed=1.5, yaw_rate=30, action_min_seconds=.5,
                          max_tokens=256, enable_thinking=False, request_timeout_seconds=150,
                          request_attempts=3, bootstrap_resamples=10000,
                          temperature='provider default (not explicitly supplied)', provider_seed=None,
                          scene_frame_reference=self.spec['frame'],
                          vehicle_spawn_settings=frame['vehicle_spawn_settings'],
                          engine=os.environ.get('UE4_EDITOR', 'H:/UE_4.27/Engine/Binaries/Win64/UE4Editor.exe'), project=self.spec['project'],
                          heading_policy='original full-method signed initialization turn',
                          comparison='historical cross-batch route-matched; repeats are not matched pairs',
                          invalid_response_policy='consume one decision, execute no action',
                          scoring_policy='endpoint/XY trajectory only; event labels never alter scoring or inclusion')
            self.state = dict(created_at=now(), status='created', config=config,
                              jobs=d.jobs(map_name, self.routes))
            atomic_json(self.directory/'routes.json', self.routes)
            self.freeze_manifest()
        self.save()

    def save(self):
        atomic_json(self.state_path, self.state)

    def event(self, name, **fields):
        row = dict(time=now(), event=name, **self.context, **fields)
        with (self.directory/'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        if name in ('engine_started', 'flight_completed', 'batch_paused', 'batch_complete', 'smoke_passed'):
            print(json.dumps(row, ensure_ascii=False), flush=True)

    def freeze_manifest(self):
        inputs = [d.SUPP/'routes.csv', d.SUPP/'episodes_manifest.csv',
                  d.SUPP/'reproduce_results.py', d.ROOT/self.spec['route_source'],
                  d.ROOT/self.spec['frame'], d.ROOT/self.spec['settings'],
                  d.SUPP/'interfaces'/self.spec['prompt'], d.SUPP/'interfaces/nine_region_mapping.csv']
        inputs += [d.SUPP/r['trajectory_file'] for r in d.historical_entries()]
        inputs += [p for p in (d.ROOT/'cue_prior_control').glob('*.py')]
        inputs += [d.ROOT/p for p in ('map1/scripts/run_heading_ablation.py', 'map2/scripts/run_heading_ablation.py',
                                     'map1/scripts/heading_ablation.py', 'map2/scripts/heading_ablation.py',
                                     'map1/scripts/heading_scene_frame.py', 'map2/scripts/heading_scene_frame.py',
                                     'map1/scripts/evaluate_nav_metrics.py', 'map1/scripts/validate_map1.py',
                                     'nav_qwen.py', 'nav_qwen_zgc.py', 'fly_on_keyboard.py', 'setup_path.py')]
        paths = sorted(set(p.resolve() for p in inputs))
        hashes = {str(p.relative_to(d.ROOT)): d.sha256(p) for p in paths}
        frame = json.loads((d.ROOT/self.spec['frame']).read_text(encoding='utf-8'))
        map_path = os.environ.get('CITYPARK_MAP' if self.spec['map_id'] == 1 else 'DOWNTOWN_MAP', frame['map_path'])
        if d.sha256(map_path) != frame['map_sha256']:
            raise PauseBatch('Map asset differs from verified scene frame')
        # Save the complete precommitted schedule separately from mutable status.
        schedule = [{k: j[k] for k in ('id', 'kind', 'route', 'repeat', 'cues')} for j in self.state['jobs']]
        atomic_json(self.directory/'schedule.json', schedule)
        atomic_json(self.directory/'frozen_config.json', self.state['config'])
        hashes.update({str(p.relative_to(d.ROOT)): d.sha256(p)
                       for p in (self.directory/'routes.json', self.directory/'schedule.json', self.directory/'frozen_config.json')})
        manifest = dict(created_at=now(), sha256=hashes, historical_episodes=880,
                        external_sha256={map_path: frame['map_sha256']})
        atomic_json(self.directory/'input_manifest.json', manifest)
        for p in paths:
            if p.suffix == '.py':
                target = self.directory/'source_snapshot'/p.relative_to(d.ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)

    def verify_manifest(self):
        manifest = json.loads((self.directory/'input_manifest.json').read_text(encoding='utf-8'))
        bad = [name for name, digest in manifest['sha256'].items()
               if not (d.ROOT/name).is_file() or d.sha256(d.ROOT/name) != digest]
        bad += [name for name, digest in manifest['external_sha256'].items()
                if not Path(name).is_file() or d.sha256(name) != digest]
        schedule = json.loads((self.directory/'schedule.json').read_text(encoding='utf-8'))
        current = [{k: j[k] for k in ('id', 'kind', 'route', 'repeat', 'cues')} for j in self.state['jobs']]
        if current != schedule or self.state['config'] != json.loads((self.directory/'frozen_config.json').read_text(encoding='utf-8')):
            bad.append('mutable state disagrees with frozen schedule/config')
        if bad:
            raise PauseBatch('Frozen inputs changed: '+', '.join(bad[:5]))

    def snapshot(self):
        pose = super().snapshot()
        pose['sim_timestamp_ns'] = int(self.drone.client.getMultirotorState().timestamp)
        return pose

    def start_engine(self):
        # The inherited City loader otherwise reads mutable global AirSim settings.
        # Use the already verified, hashed camera/spawn template instead.
        if self.spec['map_id'] == 1:
            importlib.import_module(self.spec['module']).GLOBAL_SETTINGS = d.ROOT/self.spec['settings']
        return super().start_engine()

    def request(self, urls, prompt, folder):
        messages = [{'role': 'user', 'content': urls+[{'type': 'text', 'text': prompt}]}]
        request = dict(model=d.MODEL, endpoint=d.ENDPOINT, max_tokens=256,
                       extra_body={'chat_template_kwargs': {'enable_thinking': False}},
                       prompt=prompt, role='user', content_order=['0.jpeg', '1.jpeg', '2.jpeg', 'text'],
                       images=[{'file': f'{i}.jpeg', 'sha256': d.sha256(folder/f'{i}.jpeg')} for i in range(3)])
        atomic_json(folder/'request.json', request)
        for attempt in range(1, 4):
            started = time.monotonic()
            self.event('request_started', request_attempt=attempt)
            try:
                response = self.ai.chat.completions.create(model=d.MODEL, messages=messages, max_tokens=256,
                    extra_body=request['extra_body'], timeout=150)
            except Exception as exc:
                status = getattr(exc, 'status_code', None)
                atomic_json(folder/f'request_error_{attempt}.json', dict(error_type=type(exc).__name__,
                            http_status=status, elapsed_seconds=time.monotonic()-started))
                self.event('request_failed', request_attempt=attempt, error_type=type(exc).__name__, http_status=status)
                if status in (400, 401, 402, 403, 404, 422) or attempt == 3:
                    raise PauseBatch('Model transport/API failed; all attempts preserved') from None
                time.sleep(5)
                continue
            atomic_json(folder/f'response_{attempt}.json', response.model_dump(mode='json'))
            if response.model != d.MODEL:
                raise PauseBatch('Returned model identifier differs from requested identifier')
            text = response.choices[0].message.content or ''
            action, value = self.nav.parse_action(text)
            if action is None or not math.isfinite(value):
                action, value = 'invalid', 0.
            self.event('request_completed', request_attempt=attempt, elapsed_seconds=time.monotonic()-started,
                       action=action, value=value, returned_model=response.model)
            return action, value, text

    @contextmanager
    def inference_context(self):
        if self.spec['pause']:
            self.drone.client.simPause(True)
        try:
            yield
        finally:
            if self.spec['pause']:
                self.drone.client.simPause(False)

    def flight(self, job, folder):
        from .analysis import score, read_points
        route = self.routes[str(job['route'])]
        atomic_json(folder/'route_definition.json', route)
        self.initialize(route, folder/'reset_takeoff.json')
        initialization = self.precise_turn(route['baseline_turn_deg'], folder/'full_initialization.json')
        c = self.drone.client
        contact = self.collision()
        collision_epoch = max(int(c.getMultirotorState().timestamp), contact['time_stamp_ns'])
        atomic_json(folder/'navigation_takeover.json', dict(initialization_collision=contact,
                    epoch_sim_ns=collision_epoch, pose=self.snapshot()))
        result = dict(id=job['id'], kind=job['kind'], route=job['route'], repeat=job['repeat'],
                      condition=d.CONDITION, complete=False, started_at=now(), model=d.MODEL,
                      initialization=initialization, decisions=0, actions_executed=0,
                      reason='smoke_step_limit' if job['kind'] == 'smoke' else 'max_iterations')
        trajectory = folder/'trajectory.csv'
        trajectory.write_text('X,Y,Z\n', encoding='utf-8')
        def append(pose=None):
            pose = pose or self.snapshot()
            with trajectory.open('a', encoding='utf-8') as f:
                f.write(','.join(str(v) for v in pose['position'])+'\n')
            return pose
        append()
        for step, cue in enumerate(job['cues'], 1):
            self.context['step'] = step
            step_folder = folder/f'step_{step:02d}'
            with self.inference_context():
                urls, observation = self.capture(step_folder)
                if step == 1:
                    yaw_error = abs(wrap(observation['vehicle']['yaw_deg']-initialization['end']['yaw_deg']))
                    camera_error = max(abs(wrap(cam['yaw_deg']-observation['vehicle']['yaw_deg']))
                                       for cam in observation['cameras'][:2])
                    atomic_json(folder/'first_frame_check.json', dict(yaw_error_deg=yaw_error,
                                camera_error_deg=camera_error, passed=yaw_error <= 1 and camera_error <= 1))
                    if yaw_error > 1 or camera_error > 1:
                        raise PauseBatch('First-frame heading/camera gate failed before model call')
                index, similarity, patch, position = self.nav.find_best_patch(
                    str(step_folder/'0.png'), route['clip_text'], save_dir=str(step_folder/'clip_tiles'))
                patch.save(step_folder/'clip_best.png')
                atomic_json(step_folder/'clip.json', dict(model='ViT-B/32', text=route['clip_text'],
                            best_index=int(index), score=float(similarity), position=position,
                            role='shadow computation; never used to choose transmitted cue'))
                atomic_json(step_folder/'cue.json', dict(index=cue, position=d.POSITIONS[cue],
                            source='precommitted image-independent uniform nine-region draw',
                            clip_index=int(index), coincidental_agreement=cue == int(index)))
                action, value, text = self.request(urls, d.prompt(self.map_name, route['target'], cue), step_folder)
            result['decisions'] += 1
            before = self.snapshot()
            atomic_json(step_folder/'action.json', dict(action=action, value=value, text=text, before=before))
            contact = self.collision()
            atomic_json(step_folder/'collision_before_action.json', contact)
            if contact['has_collided'] and contact['time_stamp_ns'] > collision_epoch:
                result['reason'] = 'collision'
                break
            if action == 'stop':
                result['reason'] = 'model_stop'
                break
            if action != 'invalid':
                self.nav.execute_action(self.drone, action, value)
                result['actions_executed'] += 1
                time.sleep(self.spec['wait'])
            after = append()
            atomic_json(step_folder/'action_end.json', dict(**after, executed=action != 'invalid'))
            self.event('action_executed' if action != 'invalid' else 'invalid_response_consumed', action=action, value=value)
            contact = self.collision()
            atomic_json(step_folder/'collision_after_action.json', contact)
            if contact['has_collided'] and contact['time_stamp_ns'] > collision_epoch:
                result['reason'] = 'collision'
                break
        endpoint = append()
        result.update(navigation_end=endpoint, navigation_end_position=endpoint['position'],
                      navigation_ended_at=now(), trajectory_sha256=d.sha256(trajectory))
        values = score(read_points(trajectory), route['start_position'], route['goal_position'], route['success_threshold'])
        result['metrics'] = dict(zip(('SR', 'NE', 'SPL', 'OR', 'TL'), map(float, values)))
        # Smoke acceptance concerns functioning interfaces, never navigation success/actions chosen.
        if job['kind'] == 'smoke':
            result['smoke_acceptance'] = dict(passed=result['decisions'] >= 1,
                criteria='verified initialization and images, shadow CLIP, real exact-model response and parser')
        result['complete'] = True
        atomic_json(folder/'result.json', result)
        try:
            c.landAsync(timeout_sec=30).join()
            atomic_json(folder/'landing.json', self.snapshot())
        except Exception as exc:
            atomic_json(folder/'landing.json', {'error_type': type(exc).__name__})
        self.drone.is_flying = False
        return result

    def run_job(self, job):
        job.update(status='running', attempts=job['attempts']+1)
        self.context = dict(job=job['id'], attempt=job['attempts'])
        folder = self.directory/'attempts'/job['id']/f"{job['attempts']:03d}"
        folder.mkdir(parents=True)
        self.save()
        self.event('flight_started', route=job['route'], kind=job['kind'])
        try:
            self.ensure_engine_ready()
            result = self.flight(job, folder)
        except Exception as exc:
            job['status'] = 'interrupted'
            atomic_json(folder/'failure.json', dict(at=now(), error_type=type(exc).__name__, complete=False))
            self.save()
            raise
        job.update(status='complete', result_path=str((folder/'result.json').relative_to(self.directory)))
        self.save()
        self.event('flight_completed', decisions=result['decisions'], metrics=result['metrics'],
                   formal_completed=sum(j['kind'] == 'formal' and j['status'] == 'complete' for j in self.state['jobs']))

    def preflight(self):
        self.verify_manifest()
        if self.args.endpoint != d.ENDPOINT:
            raise PauseBatch('Endpoint must equal the frozen approved endpoint')
        for path in (self.state['config']['engine'], self.state['config']['project']):
            if not Path(path).is_file():
                raise PauseBatch('Missing simulator executable/project')
        if not os.environ.get('OPENAI_API_KEY'):
            raise PauseBatch('OPENAI_API_KEY is absent')
        atomic_json(self.directory/'preflight.json', dict(passed=True, at=now(),
                    formal_episodes=self.spec['routes']*10, smoke_episodes=3,
                    note='Offline checks; API/simulator checked by short flights; no secret persisted'))

    def run(self, mode):
        self.preflight()
        if mode == 'preflight':
            self.state['status'] = 'preflight_complete'
            self.save()
            return
        self.load_runtime()
        self.state.update(status='running', runner_pid=os.getpid())
        self.save()
        for job in self.state['jobs']:
            if mode == 'smoke' and job['kind'] != 'smoke':
                continue
            if job['status'] != 'complete':
                self.run_job(job)
        self.verify_manifest()
        self.context = {}
        self.state.update(status='smoke_complete' if mode == 'smoke' else 'complete', finished_at=now())
        self.save()
        self.event('smoke_passed' if mode == 'smoke' else 'batch_complete')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map', choices=['city', 'downtown', 'all'], default='all')
    p.add_argument('--batch-dir', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--endpoint', default=d.ENDPOINT)
    modes = p.add_mutually_exclusive_group(required=True)
    for mode in ('preflight', 'smoke', 'run', 'analyze'):
        modes.add_argument('--'+mode, action='store_true')
    args = p.parse_args(argv)
    args.batch_dir = args.batch_dir.resolve()
    if not args.batch_dir.is_relative_to(d.ROOT):
        p.error('Batch output must stay inside this workspace')
    if args.analyze:
        from .analysis import analyze
        analyze(args.batch_dir)
        return 0
    names = ['city', 'downtown'] if args.map == 'all' else [args.map]
    batches = []
    try:
        for name in names:
            module = importlib.import_module(d.SPECS[name]['module'])
            cls = type('PriorBatch', (CueBatch, module.Batch), {})
            batches.append(cls(argparse.Namespace(**vars(args)), name))
        for batch in batches:
            try:
                batch.run('preflight' if args.preflight else 'smoke' if args.smoke else 'run')
            finally:
                batch.close()
        if args.run and args.map == 'all':
            from .analysis import analyze
            analyze(args.batch_dir)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        for batch in batches:
            if batch.state['status'] not in ('complete', 'smoke_complete'):
                batch.state.update(status='paused', failure_type=type(exc).__name__)
                batch.save()
                batch.event('batch_paused', error_type=type(exc).__name__)
        # API errors are sanitized in request(); local tracebacks aid simulator diagnosis.
        if batches:
            (args.batch_dir/'traceback.log').write_text(traceback.format_exc(), encoding='utf-8')
        print('Batch paused: '+type(exc).__name__+': '+str(exc), file=sys.stderr)
        return 1
    finally:
        for batch in batches:
            batch.close()


if __name__ == '__main__':
    raise SystemExit(main())
