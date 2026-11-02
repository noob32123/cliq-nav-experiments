"""Resumable downtown evaluation of the three project-adapted baselines."""
import argparse
import contextlib
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime

import cv2
import numpy as np
import formal_downtown as shared
from formal_downtown import nav, ROOT, atomic_json, NetworkAbort
from downtown_baseline_adapters import (
    POLICIES, ENGLISH_TARGETS, LocalAerialModel, PoseAdapter,
    AerialVLAFlightController, execute_discrete)

FROZEN_ROUTES = ROOT/'map2/formal_20260909_170427/routes.json'
LABELS = {'aerialvla': 'AerialVLA_INT8_Adapted',
          'citynav': 'CityNavAgent_Qwen_Adapted', 'finecog': 'FineCogNav_Qwen_Adapted'}
NORMAL = {'model_stop', 'collision', 'max_iterations', 'stagnation'}


def sources():
    paths = [ROOT/n for n in ['formal_baselines_downtown.py', 'downtown_baseline_adapters.py',
             'baseline_qwen_worker.py', 'baseline_runtime_requirements.txt',
             'formal_downtown.py', 'nav_qwen_zgc.py', 'fly_on_keyboard.py']]
    for folder in ['city_nav/adapted', 'finecog_nav/adapted', 'finecog_nav/prompts', 'aerialvla/adapted']:
        paths.extend(p for p in (ROOT/folder).iterdir() if p.suffix in ('.py', '.txt'))
    paths.extend(p for p in (ROOT/'aerialvla/models/openvla-7b').iterdir() if p.suffix in ('.py', '.json'))
    paths.append(ROOT/'aerialvla/models/aerialvla_lora/aerial_vla/adapter_config.json')
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def make_jobs(baselines, routes, repeats):
    return [{'id': f'{rid}_{name}_{repeat:02}', 'route_id': rid, 'baseline': name,
             'repeat': repeat, 'status': 'pending', 'attempts': 0}
            for name in baselines for rid in routes for repeat in range(1, repeats+1)]


class BaselineBatch(shared.Batch):
    def __init__(self, directory, args):
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=bool(args.resume))
        self.state_path = self.directory/'state.json'
        self.context = {}; self.step = 0
        self.ai = None; self.engine = None; self.drone = None; self.local_model = None
        self.sources = sources()
        if args.resume:
            self.state = json.loads(self.state_path.read_text(encoding='utf-8'))
            if self.state['source_hashes'] != self.sources:
                raise RuntimeError('Source differs from frozen batch: inspect and record change before resuming')
            self.routes = json.loads((self.directory/'routes.json').read_text(encoding='utf-8'))
            self.english = json.loads((self.directory/'english_targets.json').read_text(encoding='utf-8'))
            by_id = {j['id']: j for j in self.state['jobs']}
            for job in by_id.values():
                if job['status'] == 'running': job['status'] = 'pending'
            parent = self.directory/'smoke_results' if self.state['smoke_test'] else ROOT/'map2'
            for path in parent.glob('data_*/*/flight_*.json'):
                record = json.loads(path.read_text(encoding='utf-8'))
                if record.get('batch_id') != self.state['batch_id']: continue
                if not path.with_suffix('.csv').exists():
                    shutil.copy2(ROOT/record['attempt_directory']/'trajectory.csv', path.with_suffix('.csv'))
                by_id[record['id']].update(status='complete', result=record, trajectory=record['trajectory'])
        else:
            self.routes = json.loads(FROZEN_ROUTES.read_text(encoding='utf-8'))
            self.english = {rid: ENGLISH_TARGETS[int(rid)-1] for rid in self.routes}
            names = list(POLICIES) if args.baseline == 'all' else [args.baseline]
            rids = [f'{int(x):02}' for x in args.routes.split(',')] if args.routes else list(self.routes)
            if args.smoke_test and not args.routes: rids = ['01', '23', '30']
            if len(set(rids)) != len(rids) or not set(rids) <= self.routes.keys(): raise ValueError('Invalid routes')
            repeats = 1 if args.smoke_test else args.repeats
            self.state = {'batch_id': directory.name, 'status': 'created',
                'jobs': make_jobs(names, rids, repeats), 'repeats': repeats,
                'smoke_test': args.smoke_test, 'clock_speed': args.clock_speed,
                'max_iterations': args.max_decisions, 'source_hashes': self.sources,
                'started_at': datetime.now().isoformat(), 'completed': 0,
                'network_errors': 0, 'consecutive_request_failures': 0,
                'route_slugs': dict(zip(self.routes, shared.SLUGS)),
                'api_model': getattr(args, 'model', shared.MODEL), 'goal_information': 'evaluation_only',
                'model_output_label': getattr(args, 'model', shared.MODEL),
                'base_url': 'https://api.desire-drive.com/v1',
                'observation_mode': ('front_rgb_only' if getattr(args, 'model', shared.MODEL) == 'internvl2-26b' else 'three_views'),
                'request_wall_timeout_seconds': 180, 'aerial_quantization': 'int8',
                'weight_files': {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                    for p in (ROOT/'aerialvla/models').rglob('*.safetensors')},
                'comparison_batch': 'formal_20260909_170427'}
            atomic_json(self.directory/'routes.json', self.routes)
            atomic_json(self.directory/'english_targets.json', self.english)
            for name in self.sources:
                dest = self.directory/'sources'/name
                dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/name, dest)
            old = json.loads((ROOT/'map2/formal_20260909_170427/state.json').read_text(encoding='utf-8'))
            atomic_json(self.directory/'existing_600_hashes.json', {
                j['trajectory']: hashlib.sha256((ROOT/j['trajectory']).read_bytes()).hexdigest() for j in old['jobs']})
        self.state['status'] = 'running'
        self.state['pid'] = os.getpid()
        self.save()

    def start_engine(self):
        # The agreed batch uses ClockSpeed=5; shared startup also verifies the loaded value.
        if self.state['clock_speed'] != 5: raise ValueError('This evaluation protocol requires ClockSpeed=5')
        super().start_engine()
        self.ai.close(); self.ai = None

    def request(self, messages, max_tokens=1000):
        # Keep synthetic image payloads in pipes, not command arguments or logs.
        prompt = [m['content'] if isinstance(m['content'], str) else
                  [c['text'] for c in m['content'] if c.get('type') == 'text'] for m in messages]
        self.event('decision_input', step=self.step, prompt=prompt, max_tokens=max_tokens)
        for attempt in (1, 2):
            start = time.perf_counter()
            self.event('request_started', step=self.step, request_attempt=attempt)
            try:
                proc = subprocess.run([sys.executable, str(ROOT/'baseline_qwen_worker.py')],
                    input=json.dumps({'messages': messages, 'max_tokens': max_tokens,
                                      'model': self.state.get('api_model', shared.MODEL)}),
                    capture_output=True, text=True, encoding='utf-8', timeout=180,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                reply = json.loads(proc.stdout) if proc.returncode == 0 else {
                    'ok': False, 'error_type': 'RequestWorkerExit', 'http_status': None}
            except subprocess.TimeoutExpired:
                reply = {'ok': False, 'error_type': 'RequestWallTimeout', 'http_status': None}
            except (ValueError, OSError) as exc:
                reply = {'ok': False, 'error_type': type(exc).__name__, 'http_status': None}
            elapsed = time.perf_counter()-start
            if reply['ok']:
                self.state['consecutive_request_failures'] = 0
                self.event('request_completed', step=self.step, request_attempt=attempt,
                           elapsed_seconds=elapsed, action_text=reply['text'], usage=reply.get('usage'),
                           requested_model=self.state.get('api_model', shared.MODEL), response_model=reply.get('response_model'))
                self.save()
                return reply['text']
            self.state['network_errors'] += 1
            self.state['consecutive_request_failures'] += 1
            self.event('qwen_error', step=self.step, request_attempt=attempt,
                error_type=reply['error_type'], http_status=reply.get('http_status'),
                elapsed_seconds=elapsed, consecutive=self.state['consecutive_request_failures'])
            fatal = reply.get('http_status') in (400, 401, 402, 403, 404, 422)
            if fatal or self.state['consecutive_request_failures'] >= 3:
                self.state['status'] = 'paused_network'
            self.save()
            if self.state['status'] == 'paused_network' or attempt == 2:
                raise NetworkAbort('Request failed; archive attempt and preserve checkpoint')
            time.sleep(5)

    def capture(self, folder):
        c = self.drone.client; a = nav.airsim
        reqs = [a.ImageRequest('front_center', a.ImageType.Scene, False, False),
                a.ImageRequest('front_center', a.ImageType.DepthVis, False, False),
                a.ImageRequest('bottom_center', a.ImageType.Scene, False, False),
                a.ImageRequest('front_center', a.ImageType.DepthPlanar, True, False)]
        result = {}
        for key, r in zip(['front', 'depth_vis', 'down', 'depth_m'], c.simGetImages(reqs)):
            if r.height <= 0 or r.width <= 0: raise RuntimeError('Camera returned empty image')
            if key == 'depth_m':
                arr = np.array(r.image_data_float, dtype=np.float32).reshape(r.height, r.width)
                if not np.isfinite(arr).any(): raise RuntimeError('Metric depth invalid')
                np.save(folder/'latest_depth_m.npy', arr)
            else:
                arr = np.frombuffer(r.image_data_uint8, np.uint8).reshape(r.height, r.width, -1)
                if arr.shape[2] != 3: raise RuntimeError('Unexpected camera channels')
                cv2.imwrite(str(folder/f'latest_{key}.png'), arr)
            result[key] = arr
        return result

    def flight(self, job, folder):
        c = self.drone.client; route = self.routes[job['route_id']]
        result = {'batch_id': self.state['batch_id'], **job, 'reason': 'max_iterations',
            'decisions': 0, 'actions': 0, 'valid_decisions': 0, 'invalid_decisions': 0,
            'model': 'OpenVLA-7B+AerialVLA-LoRA-INT8' if job['baseline'] == 'aerialvla' else self.state.get('api_model', shared.MODEL),
            'max_iterations': self.state['max_iterations'], 'clock_speed': 5,
            'goal_information': 'evaluation_only', 'smoke_test': self.state['smoke_test']}
        path = folder/'trajectory.csv'
        with path.open('w', newline='') as f: csv.writer(f).writerow(['X', 'Y', 'Z'])
        def position():
            p = list(c.simGetVehiclePose().position)
            with path.open('a', newline='') as f: csv.writer(f).writerow(p)
            return p
        if job['baseline'] == 'citynav':
            result['observation_mode'] = self.state.get('observation_mode', 'three_views')
        start = time.perf_counter(); self.step = 0
        try:
            nav.initialize_route(self.drone, route, 5)
            c.simSetWind(nav.airsim.Vector3r(0, 0, 0))
            initial = position()
            if math.dist(initial[:2], route['start_position'][:2]) > 1:
                raise RuntimeError('Reset XY differs from intended route by >1 m')
            initial_yaw = nav.airsim.to_eularian_angles(c.simGetVehiclePose().orientation)[2]
            target = self.english[job['route_id']] if job['baseline'] == 'aerialvla' else route['target']
            policy = POLICIES[job['baseline']](self, target, initial_yaw)
            controller = AerialVLAFlightController(PoseAdapter(c)) if job['baseline'] == 'aerialvla' else None
            stagnation = 0
            for step in range(1, self.state['max_iterations']+1):
                self.step = step; result['decisions'] = step
                with nav.pause_for_inference(c, True):
                    images = self.capture(folder)
                    yaw = nav.airsim.to_eularian_angles(c.simGetVehiclePose().orientation)[2]
                    inference_start = time.perf_counter()
                    action = policy.decide(images, yaw)
                    self.event('policy_output', step=step, action=action, elapsed_seconds=time.perf_counter()-inference_start)
                valid = action.get('valid', action.get('code', -1) in range(8))
                if not valid:
                    result['invalid_decisions'] += 1
                    self.event('invalid_action', step=step)
                    continue
                result['valid_decisions'] += 1
                if action['stop']:
                    result['reason'] = 'model_stop'; break
                before = list(c.simGetVehiclePose().position)
                t = time.perf_counter()
                if controller:
                    controller.execute_continuous_action(action['fwd'], action['down'], action['yaw'])
                else: execute_discrete(c, action['code'])
                time.sleep(.8)
                after = position(); ci = c.simGetCollisionInfo(); result['actions'] += 1
                self.event('action', step=step, action=action, position=after,
                           wall_seconds=time.perf_counter()-t, collision=bool(ci.has_collided), collision_object=ci.object_name)
                if ci.has_collided:
                    result.update(reason='collision', collision_object=ci.object_name); break
                if job['baseline'] == 'finecog':
                    stagnation = stagnation+1 if action['code'] not in (2, 3) and math.dist(before, after) < .1 else 0
                    if stagnation >= 3: result['reason'] = 'stagnation'; break
            if result['valid_decisions'] == 0:
                result['reason'] = 'model_error'; result['error'] = 'No valid model decisions in this attempt'
        except NetworkAbort as exc:
            result.update(reason='network_interrupted', error=str(exc))
        except Exception as exc:
            result.update(reason='simulation_error', error_type=type(exc).__name__, error=str(exc))
            (folder/'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        finally:
            with contextlib.suppress(Exception): result['navigation_end_position'] = position()
            try: result['landing'] = self.land()
            except Exception as exc: result['landing'] = {'landed': False, 'error_type': type(exc).__name__}
            with contextlib.suppress(Exception): position()
            result['wall_seconds'] = time.perf_counter()-start
            atomic_json(folder/'result.json', result)
        return result

    def publish(self, job, folder, result):
        parent = self.directory/'smoke_results' if self.state['smoke_test'] else ROOT/'map2'
        label = ('CityNavAgent_Adapted' if job['baseline'] == 'citynav' and
                 self.state.get('api_model') == 'internvl2-26b' else LABELS[job['baseline']])
        group = label+'_'+self.state['route_slugs'][job['route_id']]
        if self.state.get('model_output_label') and job['baseline'] != 'aerialvla':
            group = label+'_'+self.state['model_output_label']+'_'+self.state['route_slugs'][job['route_id']]
        destination = parent/f'data_{int(job["route_id"])}'/group
        destination.mkdir(parents=True, exist_ok=True)
        path = destination/f'flight_{job["repeat"]:02}_{datetime.now():%Y%m%d_%H%M%S_%f}.csv'
        result.update(status='complete', trajectory=str(path.relative_to(ROOT)), attempt_directory=str(folder.relative_to(ROOT)))
        atomic_json(path.with_suffix('.json'), result)
        shutil.copy2(folder/'trajectory.csv', path)
        job.update(status='complete', result=result, trajectory=result['trajectory'])

    def report(self):
        if self.state.get('api_model') == 'internvl2-26b' and not self.state['smoke_test']:
            atomic_json(ROOT/'map2/citynav_current_results.json', {
                'batch':str(self.directory.relative_to(ROOT)), 'model':self.state['api_model'],
                'status':self.state['status'], 'completed':self.state['completed'],
                'total':len(self.state['jobs']), 'updated_at':datetime.now().isoformat()})
        lines = ['# Downtown baseline experiments', '',
            f'Batch: {self.state["batch_id"]}; {self.state["status"]}; {self.state["completed"]}/{len(self.state["jobs"])}',
            f'ClockSpeed=5; max decisions={self.state["max_iterations"]}; smoke={self.state["smoke_test"]}',
            f'Network errors: {self.state["network_errors"]}',
            'Project-adapted policies, not strict official reproductions. Goal coordinates are evaluation-only.',
            '', '| Route | AerialVLA INT8 | CityNavAgent | FineCogNav Qwen |', '|---|---:|---:|---:|']
        for rid in self.routes:
            counts = [sum(j['status'] == 'complete' and j['route_id'] == rid and j['baseline'] == name for j in self.state['jobs']) for name in POLICIES]
            lines.append(f'| {rid} | {counts[0]} | {counts[1]} | {counts[2]} |')
        (self.directory/'experiment_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')

    def run(self):
        try:
            if any(j['baseline'] == 'aerialvla' and j['status'] != 'complete' for j in self.state['jobs']):
                self.event('model_loading', model='AerialVLA', quantization='int8')
                self.local_model = LocalAerialModel()
                self.event('model_loaded', model='AerialVLA')
            self.start_engine()
            while True:
                job = next((j for j in self.state['jobs'] if j['status'] != 'complete'), None)
                if job is None: self.state['status'] = 'complete'; break
                if job['baseline'] != 'aerialvla' and self.local_model is not None:
                    self.local_model = None; gc.collect(); nav.torch.cuda.empty_cache()
                job['attempts'] += 1; job['status'] = 'running'
                self.context = {'job': job['id'], 'route_id': job['route_id'], 'baseline': job['baseline'],
                                'repeat': job['repeat'], 'attempt': job['attempts']}
                folder = self.directory/'attempts'/job['id']/f'{job["attempts"]:03}'
                folder.mkdir(parents=True); self.save(); self.event('flight_started')
                result = self.flight(job, folder)
                if result['reason'] in NORMAL: self.publish(job, folder, result)
                else: job['status'] = 'pending'
                if result['reason'] in ('simulation_error', 'model_error'):
                    self.state['status'] = 'paused_'+result['reason'].removesuffix('_error')
                self.event('flight_finished', reason=result['reason'], landed=result['landing']['landed'])
                self.save(); self.report()
                if self.state['status'].startswith('paused'): break
                if result['reason'] == 'network_interrupted': time.sleep(10)
        except Exception as exc:
            self.state['status'] = 'paused_infrastructure'
            self.event('infrastructure_error', error_type=type(exc).__name__, error=str(exc))
            (self.directory/'infrastructure_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        finally:
            self.save(); self.report()
            self.event('batch_finished', status=self.state['status'], completed=self.state['completed'])
            if self.drone:
                with contextlib.suppress(Exception): self.drone.client.simPause(False)
            if self.ai: self.ai.close()
            shared.terminate_owned(self.engine)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', choices=['all', *POLICIES], default='all')
    p.add_argument('--routes'); p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--max-decisions', type=int, default=30)
    p.add_argument('--clock-speed', type=float, default=5)
    p.add_argument('--smoke-test', action='store_true')
    p.add_argument('--resume', type=Path)
    p.add_argument('--model', default=shared.MODEL)
    args = p.parse_args()
    if not args.model or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._' for c in args.model):
        p.error('Model identifier must use letters, digits, hyphen, dot or underscore')
    if args.repeats < 1 or not 1 <= args.max_decisions <= 30 or args.clock_speed != 5:
        p.error('Positive repeats, 1..30 decisions and ClockSpeed=5 are required')
    if not os.getenv('OPENAI_API_KEY'): raise RuntimeError('OPENAI_API_KEY is missing')
    # Windows advisory byte lock prevents two new/resume runners sharing one simulator.
    import msvcrt
    lock_path = ROOT/'map2/baseline_runner.lock'
    with lock_path.open('a+b') as lock:
        if lock.tell() == 0: lock.write(b'0'); lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        directory = args.resume or ROOT/'map2'/f'baselines_{"smoke_" if args.smoke_test else "formal_"}{datetime.now():%Y%m%d_%H%M%S}'
        BaselineBatch(directory, args).run()


if __name__ == '__main__': main()
