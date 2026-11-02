"""Resumable 30-route, 10-repeat paired experiment; real Qwen requests only."""
import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime

import cv2
import numpy as np
from openai import OpenAI
import nav_qwen_zgc as nav
from benchmark_map2 import ENGINE, PROJECT, SETTINGS, port_open, terminate_owned

ROOT=Path(__file__).resolve().parent
MODEL='qwen3.5-35b-a3b'
SLUGS=['corner_striped_awning','brun_perez','striped_bistro','weefit','blue_frame_entrance',
       'europa','jacobs_deli','blue_awning','yellow_shop_sign','sleevo','corner_green_awning',
       'ufoods','upsway','antiques','red_frame_restaurant','striped_cafe','cafe_vendie',
       'thrifty','locafe','roadside_green_awning','eagle_statue','deer_statue','park_fountain',
       'east_food_cart','west_food_cart','purple_umbrellas','black_umbrellas','fountain_side',
       'traffic_island_bench','west_purple_umbrella']
NORMAL={'model_stop','max_iterations','collision'}


def atomic_json(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    # Windows readers can briefly deny replacement while inspecting a checkpoint.
    # Keep the old checkpoint intact, and retry only this transient access error.
    for attempt in range(21):
        try:
            os.replace(tmp,path)
            return
        except PermissionError:
            if attempt==20:
                raise
            time.sleep(0.1)


def make_jobs():
    jobs=[]
    for i in range(1,31):
        for repeat in range(1,11):
            for clip in ([True,False] if repeat%2 else [False,True]):
                jobs.append({'id':f'{i:02}_{"clip" if clip else "qwen"}_{repeat:02}',
                             'route_id':f'{i:02}','clip':clip,'repeat':repeat,'status':'pending','attempts':0})
    return jobs


class NetworkAbort(Exception):
    pass


class Batch:
    def __init__(self,directory,resume=False):
        self.directory=directory.resolve()
        self.directory.mkdir(parents=True,exist_ok=resume)
        self.state_path=self.directory/'state.json'
        if resume:
            self.state=json.loads(self.state_path.read_text(encoding='utf-8'))
            self.routes=json.loads((self.directory/'routes.json').read_text(encoding='utf-8'))
            for job in self.state['jobs']:
                if job['status']=='running':job['status']='pending'
            # Recover a publish interrupted between result, CSV, and checkpoint writes.
            by_id={job['id']:job for job in self.state['jobs']}
            for path in (ROOT/'map2').glob('data_*/*/flight_*.json'):
                record=json.loads(path.read_text(encoding='utf-8'))
                if record.get('batch_id')!=self.state['batch_id']:continue
                if not path.with_suffix('.csv').exists():
                    shutil.copy2(ROOT/record['attempt_directory']/'trajectory.csv',path.with_suffix('.csv'))
                job=by_id[record['id']];job.update(status='complete',result=record,trajectory=record['trajectory'])
        else:
            self.routes=nav.ROUTES
            atomic_json(self.directory/'routes.json',self.routes)
            shutil.copy2(ROOT/'nav_qwen_zgc.py',self.directory/'nav_qwen_zgc_snapshot.py.txt')
            shutil.copy2(Path(__file__),self.directory/'formal_downtown_snapshot.py.txt')
            self.state={'batch_id':directory.name,'status':'created','jobs':make_jobs(),
                        'consecutive_request_failures':0,'network_errors':0,'max_iterations':30,
                        'clock_speed':5,'model':MODEL,'route_slugs':dict(zip(self.routes,SLUGS)),
                        'started_at':datetime.now().isoformat(),'completed':0}
        self.context={}
        self.ai=None
        self.engine=None
        self.drone=None
        self.state['status']='running'
        self.save()

    def save(self):
        self.state['completed']=sum(j['status']=='complete' for j in self.state['jobs'])
        self.state['updated_at']=datetime.now().isoformat()
        atomic_json(self.state_path,self.state)

    def event(self,event,**fields):
        data={'time':datetime.now().isoformat(),'event':event,**self.context,**fields}
        with (self.directory/'events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(data,ensure_ascii=False)+'\n')
        print(json.dumps(data,ensure_ascii=False),flush=True)

    def start_engine(self):
        if port_open():raise RuntimeError('RPC 41451 occupied; refusing to replace existing simulation')
        config=json.loads(SETTINGS.read_text(encoding='utf-8-sig'));config['ClockSpeed']=5
        path=self.directory/'settings.json';atomic_json(path,config)
        startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
        self.engine=subprocess.Popen([str(ENGINE),str(PROJECT),'/Game/maps/airsim_downtown','-game',
            '-windowed','-ResX=1280','-ResY=720','-nosound','-unattended',f'-settings={path}',
            f'-abslog={self.directory/"unreal.log"}'],startupinfo=startup,
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        self.event('engine_started',pid=self.engine.pid)
        deadline=time.monotonic()+120
        while not port_open():
            if self.engine.poll() is not None or time.monotonic()>deadline:raise RuntimeError('UE startup failed')
            time.sleep(1)
        self.drone=nav.FixedDroneController()
        c=self.drone.client
        assert nav.read_clock_speed(c)==5
        t=c.getMultirotorState().timestamp;w=time.perf_counter();time.sleep(2)
        measured=(c.getMultirotorState().timestamp-t)/1e9/(time.perf_counter()-w)
        self.event('clock_verified',configured=5,measured=measured)
        self.ai=OpenAI(api_key=os.environ['OPENAI_API_KEY'],base_url='https://api.desire-drive.com/v1',max_retries=0,timeout=150)

    def request(self,messages,step):
        for attempt in range(1,3):
            start=time.perf_counter();self.event('request_started',step=step,request_attempt=attempt)
            try:
                response=self.ai.chat.completions.create(model=MODEL,messages=messages,max_tokens=256,
                    extra_body={'chat_template_kwargs':{'enable_thinking':False}},timeout=150)
                text=response.choices[0].message.content or ''
            except Exception as exc:
                status=getattr(exc,'status_code',None)
                self.state['consecutive_request_failures']+=1
                self.state['network_errors']+=1
                # Do not record raw exception text: SDK errors can include request data.
                self.event('qwen_error',step=step,request_attempt=attempt,error_type=type(exc).__name__,
                           http_status=status,elapsed_seconds=time.perf_counter()-start,
                           consecutive=self.state['consecutive_request_failures'])
                self.save()
                fatal=status in (400,401,402,403,404,422)
                if fatal or self.state['consecutive_request_failures']>=3:
                    self.state['status']='paused_network';self.save()
                    raise NetworkAbort('Authentication/configuration error or three consecutive request failures')
                if attempt==2:raise NetworkAbort('Two request attempts failed; archive and requeue flight')
                time.sleep(5)
            else:
                self.state['consecutive_request_failures']=0
                self.event('request_completed',step=step,request_attempt=attempt,
                           elapsed_seconds=time.perf_counter()-start,action_text=text)
                self.save()
                return text

    def land(self):
        """Land using only current local depth, never route goal coordinates."""
        c=self.drone.client;c.simPause(False)
        state=c.getMultirotorState()
        pos=list(state.kinematics_estimated.position)
        r=c.simGetImages([nav.airsim.ImageRequest('bottom_center',nav.airsim.ImageType.DepthPlanar,True,False)])[0]
        d=np.asarray(r.image_data_float).reshape(r.height,r.width)
        center=float(np.median(d[r.height//2-4:r.height//2+4,r.width//2-4:r.width//2+4]))
        if not math.isfinite(center) or not 0<=center<=200:raise RuntimeError('Invalid landing depth')
        ground=pos[2]+.7+center
        c.moveToZAsync(ground-.7,.8,timeout_sec=35).join()
        c.moveByVelocityAsync(0,0,.5,2).join()
        c.landAsync(timeout_sec=15).join()
        ci=c.simGetCollisionInfo();p=list(c.simGetVehiclePose().position)
        # A contact and low speed are required before shutting down motors.
        speed=np.linalg.norm(list(c.getMultirotorState().kinematics_estimated.linear_velocity))
        contact=bool(ci.has_collided and abs(p[2]-ground)<1.2 and speed<.4)
        if contact:c.armDisarm(False);time.sleep(.25)
        final=c.getMultirotorState()
        result={'landed':int(final.landed_state)==0,'ground_z_from_depth':ground,
                'position':list(final.kinematics_estimated.position),'contact_object':ci.object_name,
                'contact_timestamp':ci.time_stamp,'speed_before_disarm':float(speed)}
        self.drone.is_flying=not result['landed']
        return result

    def flight(self,job,folder):
        route=self.routes[job['route_id']];c=self.drone.client
        result={'batch_id':self.state['batch_id'],**job,'reason':'max_iterations','decisions':0,'actions':0,
                'clip_enabled':job['clip'],'model':MODEL,'max_iterations':30,'clock_speed':5}
        csv_path=folder/'trajectory.csv'
        def position():
            p=list(c.simGetVehiclePose().position)
            with csv_path.open('a',newline='') as f:csv.writer(f).writerow(p)
            return p
        with csv_path.open('w',newline='') as f:csv.writer(f).writerow(['X','Y','Z'])
        start=time.perf_counter()
        try:
            nav.initialize_route(self.drone,route,5)
            c.simSetWind(nav.airsim.Vector3r(0,0,0));position()
            for step in range(1,31):
                result['decisions']=step
                with nav.pause_for_inference(c,True):
                    requests=[nav.airsim.ImageRequest('front_center',nav.airsim.ImageType.Scene,False,False),
                              nav.airsim.ImageRequest('front_center',nav.airsim.ImageType.DepthVis,False,False),
                              nav.airsim.ImageRequest('bottom_center',nav.airsim.ImageType.Scene,False,False)]
                    responses=c.simGetImages(requests);images=[]
                    for i,r in enumerate(responses):
                        raw=np.frombuffer(r.image_data_uint8,dtype=np.uint8)
                        if r.height==0:raise RuntimeError('Camera image missing')
                        image=raw.reshape(r.height,r.width,-1)
                        if image.shape[2]==1:image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
                        if image.shape[2]!=3:raise RuntimeError('Unexpected camera channels')
                        path=folder/f'latest_{i}.png';cv2.imwrite(str(path),image)
                        images.append({'type':'image_url','image_url':{'url':nav.encode_numpy_to_data_url(image)}})
                    prompt=nav.build_target_prompt(route['target'])
                    if job['clip']:
                        t=time.perf_counter();idx,score,patch,pos=nav.find_best_patch(str(folder/'latest_0.png'),route['clip_text'])
                        patch.save(folder/'latest_clip_patch.png')
                        prompt+=f'为辅助你的判断：你的导航目标有很大的可能性在你的{pos}'
                        self.event('clip',step=step,index=int(idx),score=float(score),position=pos,elapsed_seconds=time.perf_counter()-t)
                    self.event('decision_input',step=step,prompt=prompt)
                    text=self.request([{'role':'user','content':images+[{'type':'text','text':prompt}]}],step)
                    action,value=nav.parse_action(text)
                if action is None:continue
                if action=='stop':result['reason']='model_stop';break
                t=time.perf_counter();nav.execute_action(self.drone,action,value);time.sleep(.8)
                p=position();ci=c.simGetCollisionInfo();result['actions']+=1
                self.event('action',step=step,action=action,value=value,position=p,
                           wall_seconds=time.perf_counter()-t,collision=bool(ci.has_collided),collision_object=ci.object_name)
                if ci.has_collided:
                    result['reason']='collision';result['collision_object']=ci.object_name;break
        except NetworkAbort as exc:
            result['reason']='network_interrupted';result['error']=str(exc)
        except Exception as exc:
            result['reason']='simulation_error';result['error_type']=type(exc).__name__;result['error']=str(exc)
        finally:
            result['navigation_end_position']=position()
            try:result['landing']=self.land()
            except Exception as exc:result['landing']={'landed':False,'error_type':type(exc).__name__}
            position();result['wall_seconds']=time.perf_counter()-start
            atomic_json(folder/'result.json',result)
        return result

    def publish(self,job,folder,result):
        slug=self.state['route_slugs'][job['route_id']]
        group=MODEL+('_CLIP' if job['clip'] else '')+'_'+slug
        destination=ROOT/'map2'/f'data_{int(job["route_id"])}'/group
        destination.mkdir(parents=True,exist_ok=True)
        stem=f'flight_{job["repeat"]:02}_{datetime.now():%Y%m%d_%H%M%S_%f}'
        path=destination/(stem+'.csv');assert not path.exists()
        result['trajectory']=str(path.relative_to(ROOT))
        result['attempt_directory']=str(folder.relative_to(ROOT))
        atomic_json(path.with_suffix('.json'),result)
        shutil.copy2(folder/'trajectory.csv',path)
        job['status']='complete';job['result']=result;job['trajectory']=str(path.relative_to(ROOT))

    def report(self):
        lines=['# Downtown 正式实验运行记录','',f'- 批次：`{self.state["batch_id"]}`',
               f'- 完成：{self.state["completed"]}/600；状态：{self.state["status"]}',
               '- 模型：qwen3.5-35b-a3b；ClockSpeed=5；每次最多 30 次决策。',
               '- 两组的差异仅为 CLIP 推理及其方位提示；不以模型 stop 判定导航成功。',
               f'- 公网请求错误：{self.state["network_errors"]} 次；中断与补跑保留在 attempts 和 events.jsonl。','',
               '| 路线 | Qwen+CLIP 完成 | Qwen 完成 |','|---|---:|---:|']
        for rid in self.routes:
            counts=[sum(j['status']=='complete' and j['route_id']==rid and j['clip']==flag for j in self.state['jobs']) for flag in [True,False]]
            lines.append(f'| {rid} {self.routes[rid]["name"]} | {counts[0]} | {counts[1]} |')
        (self.directory/'experiment_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

    def run(self):
        try:
            self.start_engine()
            while True:
                pending=[j for j in self.state['jobs'] if j['status']!='complete']
                if not pending:self.state['status']='complete';break
                job=pending[0];job['attempts']+=1;job['status']='running'
                self.context={'job':job['id'],'route_id':job['route_id'],'clip':job['clip'],'repeat':job['repeat'],'attempt':job['attempts']}
                folder=self.directory/'attempts'/job['id']/f'{job["attempts"]:03}'
                folder.mkdir(parents=True)
                self.save();self.event('flight_started')
                result=self.flight(job,folder)
                if result['reason'] in NORMAL:self.publish(job,folder,result)
                else:job['status']='pending'
                if result['reason']=='simulation_error':self.state['status']='paused_simulation'
                self.event('flight_finished',reason=result['reason'],landed=result['landing']['landed'])
                self.save();self.report()
                if self.state['status'].startswith('paused'):break
                if result['reason']=='network_interrupted':time.sleep(10)
            self.save();self.report();self.event('batch_finished',status=self.state['status'],completed=self.state['completed'])
        finally:
            if self.drone:
                with contextlib.suppress(Exception):self.drone.client.simPause(False)
            if self.ai:self.ai.close()
            terminate_owned(self.engine)


def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',type=Path)
    args=p.parse_args()
    if not os.getenv('OPENAI_API_KEY'):raise RuntimeError('OPENAI_API_KEY is missing')
    directory=args.resume or ROOT/'map2'/f'formal_{datetime.now():%Y%m%d_%H%M%S}'
    Batch(directory,resume=args.resume is not None).run()


if __name__=='__main__':main()
