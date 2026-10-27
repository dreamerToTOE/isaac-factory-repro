#!/usr/bin/env python3
"""[ENGINEERING] Independent 10Hz GPU guard, sampling before gated training exec.

1024MiB free / three consecutive low samples remains fixed. Runtime samples and
heartbeats are flushed/atomic without per-sample fsync; immutable metadata and
final status are fsynced. The guard also detects supervisor loss independently.
"""
from __future__ import annotations
import argparse
import csv
import ctypes as C
import json
import math
import os
from pathlib import Path
import signal
import threading
import time
try:
    from supervision_common import BOOT_ID, atomic_json, fresh, identity, read_json, same_process, stamp, stop_owned
except ModuleNotFoundError:  # package import from repository-root tests
    from scripts.supervision_common import BOOT_ID, atomic_json, fresh, identity, read_json, same_process, stamp, stop_owned

class Memory(C.Structure):
    _fields_=[('version',C.c_uint)]+[(n,C.c_ulonglong) for n in ('total','reserved','free','used')]

class Nvml:
    def __init__(self):
        self.lib=C.CDLL('libnvidia-ml.so.1')
        if self.lib.nvmlInit_v2()!=0:raise RuntimeError('NVML init failed')
        self.dev=C.c_void_p()
        if self.lib.nvmlDeviceGetHandleByIndex_v2(C.c_uint(0),C.byref(self.dev))!=0:raise RuntimeError('GPU0 unavailable')
    def read(self):
        m=Memory();m.version=C.sizeof(m)|(2<<24)
        rc=self.lib.nvmlDeviceGetMemoryInfo_v2(self.dev,C.byref(m))
        if rc:raise RuntimeError(f'NVML memory status {rc}')
        return {'total_MiB':m.total/2**20,'reserved_MiB':m.reserved/2**20,
                'free_MiB':m.free/2**20,'used_MiB':m.used/2**20}
    def close(self):self.lib.nvmlShutdown()

class Mock:
    def __init__(self,path):self.rows=json.loads(path.read_text());self.n=0
    def read(self):
        row=self.rows[min(self.n,len(self.rows)-1)];self.n+=1
        if 'delay_seconds'in row:time.sleep(row['delay_seconds'])
        if 'error'in row:raise RuntimeError(row['error'])
        return {'free_MiB':float(row['free_MiB']),'used_MiB':8192-float(row['free_MiB']),
                'total_MiB':8192,'reserved_MiB':0}
    def close(self):pass


def gap_exceeds_limit(previous, current, limit):
    """Preserve strict greater-than semantics for the sampled gap."""
    return current - previous > limit


class HeartbeatPublisher:
    """Publish the newest heartbeat without blocking the sampling loop.

    The pending slot is bounded to one item. Replacing an older pending
    heartbeat is safe because the payload carries its real sample timestamp;
    the replacement count is retained in final evidence. A publisher failure
    or stale publication is reported to the sampler and remains fail-closed.
    """

    def __init__(self, path, write_fn=atomic_json, stale_seconds=2.0):
        self.path = Path(path)
        self.write_fn = write_fn
        self.stale_seconds = float(stale_seconds)
        self._condition = threading.Condition()
        self._pending = None
        self._stop = False
        self._failure = None
        self._published_count = 0
        self._coalesced_count = 0
        self._first_submitted_sample = None
        self._last_submitted_sample = None
        self._last_published_sample = None
        self._last_publish_monotonic = None
        self._timings = {
            'count': 0,
            'total_seconds': 0.0,
            'max_seconds': 0.0,
        }
        self._thread = threading.Thread(target=self._run, name='heartbeat-publisher', daemon=True)
        self._thread.start()

    def submit(self, payload, sample_monotonic):
        with self._condition:
            if self._failure is not None:
                raise RuntimeError(f'heartbeat publisher failed: {self._failure}')
            if self._stop:
                raise RuntimeError('heartbeat publisher is stopped')
            if self._pending is not None:
                self._coalesced_count += 1
            self._pending = (payload, float(sample_monotonic))
            if self._first_submitted_sample is None:
                self._first_submitted_sample = float(sample_monotonic)
            self._last_submitted_sample = float(sample_monotonic)
            self._condition.notify()

    def health(self, now=None):
        now = time.monotonic() if now is None else float(now)
        with self._condition:
            if self._failure is not None:
                return 'failed'
            if self._last_submitted_sample is None:
                return 'starting'
            publication_anchor = self._last_published_sample or self._first_submitted_sample
            if publication_anchor is not None and now - publication_anchor > self.stale_seconds:
                return 'stale'
            return 'healthy'

    def _run(self):
        while True:
            with self._condition:
                while self._pending is None and not self._stop:
                    self._condition.wait()
                if self._pending is None and self._stop:
                    return
                payload, sample_monotonic = self._pending
                self._pending = None
            started = time.monotonic()
            try:
                self.write_fn(self.path, payload, durable=False)
            except BaseException as exc:  # surfaced through health(), then fail closed
                with self._condition:
                    self._failure = f'{type(exc).__name__}: {exc}'
                    self._condition.notify_all()
                return
            elapsed = time.monotonic() - started
            with self._condition:
                self._published_count += 1
                self._last_published_sample = sample_monotonic
                self._last_publish_monotonic = time.monotonic()
                self._timings['count'] += 1
                self._timings['total_seconds'] += elapsed
                self._timings['max_seconds'] = max(self._timings['max_seconds'], elapsed)

    def close(self, timeout=0.2):
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        self._thread.join(max(0.0, float(timeout)))
        with self._condition:
            return {
                'published_count': self._published_count,
                'coalesced_count': self._coalesced_count,
                'failure': self._failure,
                'thread_alive_after_close': self._thread.is_alive(),
                'first_submitted_sample': self._first_submitted_sample,
                'last_submitted_sample': self._last_submitted_sample,
                'last_published_sample': self._last_published_sample,
                'last_publish_monotonic': self._last_publish_monotonic,
                'timing': dict(self._timings),
            }


def run(args):
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    who=identity();started=stamp();super_id=None;target=None;reason=None;error=None;signals=[]
    samples=0;low=0;max_low=0;minimum=None;peak=None;first=None;last=None;gaps=[];source=None
    attached_record=None
    supervisor_heartbeat_seen=False
    # Constant-space aggregates; no extra per-sample disk writes. The same
    # sample-to-sample gap and stop thresholds remain authoritative.
    timings={}
    previous_cycle=None
    previous_sleep=None
    def record_timing(name, duration):
        entry=timings.setdefault(name, {'count':0,'total_seconds':0.0,'max_seconds':0.0})
        entry['count']+=1;entry['total_seconds']+=duration;entry['max_seconds']=max(entry['max_seconds'],duration)
    requested_stop=[];handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    for s in handlers:signal.signal(s,lambda n,frame:requested_stop.append(n))
    metadata={'schema_version':2,'identity':who,'boot_id':BOOT_ID,'started':started,
              'supervisor_directory':str(args.supervisor_directory),
              'policy':{'floor_MiB':1024,'consecutive_samples':3,'interval_seconds':args.interval,
                        'gap_limit_seconds':args.gap_limit,'stale_seconds':args.stale_seconds},
              'mock_source':str(args.mock_samples) if args.mock_samples else None}
    atomic_json(args.output/'metadata.json',metadata,replace=False)
    publisher = None
    try:
        source=Mock(args.mock_samples) if args.mock_samples else Nvml()
        publisher = HeartbeatPublisher(args.output/'heartbeat.json', stale_seconds=args.stale_seconds)
        with (args.output/'samples.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['utc','unix_seconds','monotonic_seconds','elapsed_seconds',
                    'used_MiB','free_MiB','total_MiB','reserved_MiB','target_pid','low_consecutive'])
            w.writeheader();f.flush();deadline=time.monotonic()
            while True:
                query_start=time.monotonic()
                try: mem=source.read()
                finally:
                    query_seconds=time.monotonic()-query_start
                    record_timing('memory_query',query_seconds)
                at=stamp();cycle_start=time.monotonic()
                if not all(math.isfinite(v) and v>=0 for v in mem.values()):raise RuntimeError('Invalid NVML memory values')
                if last and gap_exceeds_limit(last['monotonic_seconds'],at['monotonic_seconds'],args.gap_limit):
                    gaps.append({'from':last,'to':at,'duration_seconds':at['monotonic_seconds']-last['monotonic_seconds'],
                                 'timing_detail':{'previous_cycle':previous_cycle,'previous_sleep':previous_sleep,
                                                  'current_memory_query_seconds':query_seconds}})
                first=first or at;last=at;samples+=1
                low=low+1 if mem['free_MiB']<1024 else 0;max_low=max(max_low,low)
                minimum=mem['free_MiB'] if minimum is None else min(minimum,mem['free_MiB'])
                peak=mem['used_MiB'] if peak is None else max(peak,mem['used_MiB'])
                # Bind only immutable identities generated by this supervisor.
                super_record=read_json(args.supervisor_directory/'identity.json')
                if super_record and super_id is None:super_id=super_record['identity']
                child_record=read_json(args.supervisor_directory/'child_identity.json')
                if child_record and target is None:
                    candidate=child_record['identity']
                    if same_process(candidate):
                        target=candidate
                        attached_record={'target_identity':target,
                                         'supervisor_identity':super_id,**at}
                before_csv=time.monotonic()
                coordination_seconds=before_csv-cycle_start
                record_timing('processing_and_coordination_before_csv',coordination_seconds)
                w.writerow({**at,'elapsed_seconds':at['monotonic_seconds']-started['monotonic_seconds'],
                            **mem,'target_pid':target['pid'] if target else '', 'low_consecutive':low});f.flush()
                csv_seconds=time.monotonic()-before_csv;record_timing('csv_write_flush',csv_seconds)
                heartbeat_start=time.monotonic()
                state='bound' if target else ('ready' if mem['free_MiB']>=1024 else 'low_headroom')
                heartbeat_payload={'identity':who,'boot_id':BOOT_ID,**at,
                            'target_identity':target,'supervisor_identity':super_id,'state':state,
                            'sample_count':samples,'free_MiB':mem['free_MiB'],'low_consecutive':low}
                publisher.submit(heartbeat_payload, at['monotonic_seconds'])
                heartbeat_seconds=time.monotonic()-heartbeat_start;record_timing('heartbeat_submit',heartbeat_seconds)
                tail_start=time.monotonic()
                exited=read_json(args.supervisor_directory/'child_exit.json')
                final=read_json(args.supervisor_directory/'status.json')
                if exited and exited.get('identity')==target and target:
                    reason='child_wait_observed';break
                if final and not target:
                    reason='supervisor_finished_before_launch';break
                if requested_stop:reason='guard_signal';break
                if low>=3:reason='gpu_headroom_floor';break
                if gaps:reason='monitoring_gap';break
                publisher_health=publisher.health()
                if publisher_health != 'healthy':
                    reason='heartbeat_publisher_'+publisher_health
                    break
                if super_id:
                    shb=read_json(args.supervisor_directory/'heartbeat.json')
                    supervisor_heartbeat_seen=supervisor_heartbeat_seen or bool(shb)
                    startup_pending=not supervisor_heartbeat_seen and target is None and time.monotonic()-started['monotonic_seconds']<args.startup_timeout
                    if not same_process(super_id) or (not startup_pending and (not fresh(shb,args.stale_seconds) or shb.get('identity')!=super_id)):
                        reason='supervisor_lost_or_stale';break
                elif time.monotonic()-started['monotonic_seconds']>args.startup_timeout:
                    reason='supervisor_startup_timeout';break
                if target and not same_process(target):
                    # Give its parent wait a bounded chance to publish actual status.
                    # Continue sampling; do not invent a child return code here.
                    if not super_id or not same_process(super_id):reason='supervisor_lost_or_stale';break
                tail_seconds=time.monotonic()-tail_start;record_timing('processing_and_coordination_after_heartbeat',tail_seconds)
                previous_cycle={'processing_and_coordination_before_csv_seconds':coordination_seconds,
                                'csv_write_flush_seconds':csv_seconds,'heartbeat_submit_seconds':heartbeat_seconds,
                                'processing_and_coordination_after_heartbeat_seconds':tail_seconds}
                deadline+=args.interval
                sleep_start=time.monotonic();requested=max(0,deadline-sleep_start)
                time.sleep(requested)
                sleep_end=time.monotonic();actual=sleep_end-sleep_start
                previous_sleep={'requested_seconds':requested,'actual_seconds':actual,
                                'oversleep_seconds':max(0.0,actual-requested),
                                'deadline_lateness_seconds':max(0.0,sleep_end-deadline)}
                record_timing('sleep_requested',requested);record_timing('sleep_actual',actual)
                record_timing('sleep_overshoot',previous_sleep['oversleep_seconds'])
                record_timing('deadline_lateness',previous_sleep['deadline_lateness_seconds'])
            f.flush();os.fsync(f.fileno())
    except BaseException as exc:
        error={'type':type(exc).__name__,'message':str(exc)};reason='monitor_failed'
    finally:
        if reason not in ('child_wait_observed','supervisor_finished_before_launch') and target:
            try:signals=stop_owned(target)
            except Exception as exc:error=error or {'type':type(exc).__name__,'message':str(exc)}
        if source:
            try:source.close()
            except Exception as exc:error=error or {'type':type(exc).__name__,'message':str(exc)}
        publisher_status = publisher.close() if publisher is not None else None
        if attached_record is not None:
            try:
                atomic_json(args.output/'attached_identity.json', attached_record, replace=False)
            except BaseException as exc:
                error=error or {'type':type(exc).__name__,'message':str(exc)}
        for s,h in handlers.items():signal.signal(s,h)
        publisher_good=(publisher_status is None or
                        (publisher_status['failure'] is None and
                         not publisher_status['thread_alive_after_close'] and
                         publisher_status['published_count'] > 0))
        good=reason=='child_wait_observed' and not gaps and error is None and publisher_good
        result={'schema_version':2,'finalized':True,'identity':who,'target_identity':target,
                'supervisor_identity':super_id,'started':started,'ended':stamp(),
                'sample_count':samples,'first_sample':first,'last_sample':last,
                'min_free_MiB':minimum,'peak_used_MiB':peak,'max_low_consecutive':max_low,
                'low_memory_stop_condition_observed':max_low>=3,
                'monitoring_gaps':gaps,'stop_reason':reason,'error':error,'stop_signals':signals,
                'continuous_monitoring':good,'coverage_note':'Sampled coverage: regular intervals are not continuous measurement.',
                'policy':metadata['policy'], 'timing_diagnostics':{'aggregates':timings,
                    'heartbeat_publisher': publisher_status,
                    'note':'Sampling and heartbeat publication are separate. Wall-clock segments include scheduling delays; sleep overshoot is not proof of scheduler cause. Gap detail is partial and instrumentation adds overhead.'}}
        atomic_json(args.output/'status.json',result,replace=False)
    return 0 if good else 124


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--supervisor-directory',type=Path,required=True)
    p.add_argument('--interval',type=float,default=.1)
    p.add_argument('--gap-limit',type=float,default=.5)
    p.add_argument('--stale-seconds',type=float,default=2)
    p.add_argument('--startup-timeout',type=float,default=30)
    p.add_argument('--mock-samples',type=Path)
    a=p.parse_args(argv)
    if min(a.interval,a.gap_limit,a.stale_seconds,a.startup_timeout)<=0:p.error('Positive times required')
    return run(a)
if __name__=='__main__':raise SystemExit(main())
