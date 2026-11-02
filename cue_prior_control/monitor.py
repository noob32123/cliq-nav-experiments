"""Read-only experiment observer; writes only its own status/result-view files.

python -m cue_prior_control.monitor --batch-dir PATH [--once]
No simulator RPC, network requests, scientific input edits or paper updates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import psutil


def load(path):
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return None


def last_event(path):
    try:
        with path.open('rb') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size-65536))
            lines = f.read().decode('utf-8', errors='replace').splitlines()
        for line in reversed(lines):
            try:
                return json.loads(line)
            except ValueError:
                pass
    except OSError:
        pass
    return None


def process_alive(pid):
    if not pid:
        return False
    try:
        p = psutil.Process(int(pid))
        return p.is_running() and 'cue_prior_control.runner' in ' '.join(p.cmdline())
    except psutil.AccessDenied:
        return psutil.pid_exists(int(pid))
    except (psutil.NoSuchProcess, ValueError):
        return False


def publish(path, text):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(text, encoding='utf-8-sig')
    for retry in range(20):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            time.sleep(.1)
    raise OSError('Status file locked: '+str(path))


def snapshot(root):
    current = datetime.now().astimezone()
    maps, complete = {}, 0
    active_ages = []
    for name, total in [('city', 140), ('downtown', 300)]:
        state = load(root/name/'state.json') or {}
        jobs = state.get('jobs', [])
        finished = sum(j['kind'] == 'formal' and j['status'] == 'complete' for j in jobs)
        smoke = sum(j['kind'] == 'smoke' and j['status'] == 'complete' for j in jobs)
        running = [j['id'] for j in jobs if j['status'] == 'running']
        event = last_event(root/name/'events.jsonl') or {}
        age = None
        if event.get('time'):
            age = max(0., (current-datetime.fromisoformat(event['time'])).total_seconds())
        if running and age is not None:
            active_ages.append(age)
        maps[name] = dict(status=state.get('status', 'not initialized'), complete=finished, total=total,
                          smoke_complete=smoke, current_job=running[0] if running else None,
                          last_event=event, activity_age_seconds=age)
        complete += finished
    try:
        pid = int((root/'formal.pid').read_text(encoding='utf-8-sig').strip())
    except (OSError, ValueError):
        pid = None
    alive = process_alive(pid)
    analyzed = load(root/'analysis/integrity.json')
    if complete == 440 and analyzed and (root/'analysis/results.json').is_file():
        status = 'COMPLETE'
    elif complete == 440:
        status = 'ANALYZING' if alive else 'ANALYSIS_REQUIRED'
    elif not alive:
        status = 'PAUSED_OR_EXITED'
    elif active_ages and max(active_ages) > 600:
        status = 'CHECK_STALL'
    else:
        status = 'RUNNING'
    return dict(updated_at=current.isoformat(), overall_status=status, complete=complete, total=440,
                runner_pid=pid, runner_alive=alive, monitor_pid=os.getpid(), maps=maps,
                manuscript_policy='HOLD: present results first; do not update manuscript/PDF/figures/submission archives')


def render(status):
    lines = ['均匀方向先验 cue 实验：实时状态', f"更新时间：{status['updated_at']}",
             f"总体状态：{status['overall_status']}",
             f"正式回合：{status['complete']}/440 ({100*status['complete']/440:.1f}%)",
             f"运行进程：PID {status['runner_pid']}，存活={status['runner_alive']}",
             '自动刷新：每10秒；本文件不调用模拟器或模型接口。',
             '论文保持不更新；完成后先查看 RESULTS.txt。', '']
    for name, label in [('city', 'City Park'), ('downtown', 'Downtown West')]:
        m = status['maps'][name]
        event = m['last_event']
        age = m['activity_age_seconds']
        lines += [f"{label}: {m['complete']}/{m['total']}，短检查 {m['smoke_complete']}/3，状态 {m['status']}",
                  f"  当前任务：{m['current_job'] or '无（等待/已完成/暂停）'}",
                  f"  最新事件：{event.get('event', '-')}；step={event.get('step', '-')}；时间={event.get('time', '-')}",
                  f"  距最新活动：{age:.0f} 秒" if age is not None else '  距最新活动：无记录']
        if event.get('event') == 'batch_paused':
            lines.append('  异常类型：'+event.get('error_type', 'unknown'))
        lines.append('')
    lines += ['说明：正常失败、碰撞、提前停止和步数耗尽均保留；仅按终点/轨迹计算五指标。',
              '固定440个正式回合；原实际CLIP cue与无cue两组均不重跑。',
              '日志：各地图 events.jsonl；历史启动故障见对应 attempt 和 traceback.log。']
    return '\n'.join(lines)+'\n'


def result_view(root):
    results = load(root/'analysis/results.json')
    if not results:
        return
    lines = ['均匀方向先验 cue 实验结果（先展示，论文未更新）',
             '复用880回合，新增440回合；跨批次路线匹配比较。',
             'SR/SPL/OR单位为%；NE/TL单位为米。', '']
    for env, data in results['environments'].items():
        lines += [env, '条件 | SR | NE | SPL | OR | TL']
        for c, metrics in data['methods'].items():
            values = [metrics[k]['mean']*(100 if k in ('SR', 'SPL', 'OR') else 1)
                      for k in ('SR', 'NE', 'SPL', 'OR', 'TL')]
            lines.append(c+' | '+' | '.join(f'{v:.2f}' for v in values))
        for contrast, metrics in data['differences'].items():
            s = metrics['SR']
            lines.append(f"{contrast}: SR差值 {100*s['mean']:.2f}个百分点，{100*s['coverage']:.1f}%区间 [{100*s['low']:.2f}, {100*s['high']:.2f}]")
        lines.append('')
    lines += ['主要比较使用97.5%区间；其它差值为探索性95%区间。',
              '不能将跨批次比较解释为同批次随机对照或纯因果机制验证。',
              '完整结果、路线差值、回合指标及核验记录见 analysis/。']
    publish(root/'RESULTS.txt', '\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-dir', type=Path, required=True)
    p.add_argument('--once', action='store_true')
    args = p.parse_args()
    root = args.batch_dir.resolve()
    (root/'monitor.pid').write_text(str(os.getpid()), encoding='utf-8')
    previous, last_logged = None, 0.
    while True:
        state = snapshot(root)
        publish(root/'LIVE_STATUS.txt', render(state))
        publish(root/'live_status.json', json.dumps(state, ensure_ascii=False, indent=2)+'\n')
        key = (state['overall_status'], state['complete'],
               tuple((m['current_job'], m['last_event'].get('event'), m['last_event'].get('step')) for m in state['maps'].values()))
        if key != previous or time.monotonic()-last_logged >= 60:
            with (root/'monitor.log').open('a', encoding='utf-8-sig') as f:
                f.write(f"{state['updated_at']} {state['overall_status']} {state['complete']}/440 " +
                        ' | '.join(f"{name} {m['complete']}/{m['total']} {m['current_job'] or '-'} {m['last_event'].get('event', '-')} step={m['last_event'].get('step', '-')}"
                                   for name, m in state['maps'].items())+'\n')
            previous, last_logged = key, time.monotonic()
        if state['overall_status'] == 'COMPLETE':
            result_view(root)
            return
        if args.once:
            return
        time.sleep(10)


if __name__ == '__main__':
    main()
