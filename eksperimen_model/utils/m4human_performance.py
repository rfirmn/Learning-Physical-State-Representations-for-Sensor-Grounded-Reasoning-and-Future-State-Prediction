"""Shared, explicit performance policy; never changes scientific batch policy."""
import math
import random
import threading
import time
from .m4human_runtime import process_tree_resources
import numpy as np
import torch
from torch.utils.data import DataLoader


def _integer(settings, key, default, minimum):
    value = settings.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{key} must be an integer >= {minimum}')
    return value


def validate_performance(settings):
    for key, default, minimum in [('workers',0,0),('prefetch_factor',2,1),('cpu_threads',1,1),
                                 ('micro_batch',4,1),('accumulation',1,1),('eval_batch_size',4,1),('seed',42,0),
                                 ('ram_reserve_mib',2048,0),('vram_reserve_mib',1024,0),('resource_poll_ms',250,50)]:
        _integer(settings,key,default,minimum)
    for key in ('persistent_workers','pin_memory','non_blocking'):
        if key in settings and not isinstance(settings[key],bool):
            raise ValueError(f'{key} must be boolean')
    if settings.get('multiprocessing_context','spawn') not in ('spawn','fork','forkserver'):
        raise ValueError('invalid multiprocessing_context')
    if settings.get('persistent_workers',False) and not settings.get('workers',0):
        raise ValueError('persistent_workers requires workers > 0')
    if settings.get('precision','fp32') not in ('fp32','fp16_amp_with_grad_scaler','fp16_amp','fp16_amp_grad_scaler','bf16'):
        raise ValueError('unsupported precision')
    effective = settings.get('effective_batch',settings.get('micro_batch',4)*settings.get('accumulation',1))
    if isinstance(effective,bool) or not isinstance(effective,int) or effective < 1:
        raise ValueError('effective_batch must be a positive integer')
    if effective != settings.get('micro_batch',4)*settings.get('accumulation',1):
        raise ValueError('effective_batch must equal micro_batch * accumulation')


def configure_runtime(settings, device):
    validate_performance(settings)
    if settings.get('precision','fp32') != 'fp32' and torch.device(device).type != 'cuda':
        raise ValueError('fp16 AMP requires CUDA; choose fp32 explicitly')
    torch.set_num_threads(settings.get('cpu_threads',1))


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(dataset,batch_size,settings,device,shuffle=False,collate_fn=None,generator=None):
    validate_performance(settings)
    if isinstance(batch_size,bool) or not isinstance(batch_size,int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    workers = settings.get('workers',0)
    kwargs = dict(batch_size=batch_size,shuffle=shuffle,collate_fn=collate_fn,num_workers=workers,
                  pin_memory=settings.get('pin_memory',torch.device(device).type=='cuda'),
                  worker_init_fn=seed_worker,generator=generator)
    if workers:
        kwargs.update(prefetch_factor=settings.get('prefetch_factor',2),
                      persistent_workers=settings.get('persistent_workers',False),
                      multiprocessing_context=settings.get('multiprocessing_context','spawn'))
    return DataLoader(dataset,**kwargs)


def finish_optimizer_step(optimizer,scaler,parameters,max_norm,clipping_groups=None,norms_out=None):
    """Recover scaled overflow before clipping; reject nonfinite unscaled training."""
    parameters = list(parameters)
    groups = [(parameters,max_norm)] if clipping_groups is None else [(list(group),limit) for group,limit in clipping_groups]
    if not groups or any(not group for group,limit in groups):
        raise ValueError('clipping groups must contain parameters')
    if any(isinstance(limit,bool) or not isinstance(limit,(int,float)) or not math.isfinite(limit) or limit <= 0 for group,limit in groups):
        raise ValueError('clip max_norm must be finite and positive')
    optimizer_parameters = [p for group in optimizer.param_groups for p in group['params']]
    allowed = {id(p) for p in optimizer_parameters}
    clipped = [id(p) for group,limit in groups for p in group]
    if any(identity not in allowed for identity in clipped) or len(set(clipped)) != len(clipped):
        raise ValueError('clipping groups must be disjoint optimizer parameters')
    if not {id(p) for p in parameters}.issubset(set(clipped)):
        raise ValueError('clipping groups must cover supplied parameters')
    scaler.unscale_(optimizer)
    checks = [torch.isfinite(p.grad).all() for p in optimizer_parameters if p.grad is not None]
    finite = bool(torch.stack(checks).all().item()) if checks else True
    if not finite and scaler.is_enabled():
        old_scale = scaler.get_scale()
        scaler.step(optimizer)  # GradScaler found_inf prevents the update.
        scaler.update()
        if scaler.get_scale() >= old_scale:
            raise RuntimeError('nonfinite gradients were not recognized by GradScaler')
        return None,False
    norms = [torch.nn.utils.clip_grad_norm_(group,limit,error_if_nonfinite=True) for group,limit in groups]
    if norms_out is not None:
        norms_out.update({index:float(value) for index,value in enumerate(norms)})
    norm = max(float(value) for value in norms)
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return float(norm),scaler.get_scale() >= old_scale


class ResourceMonitor:
    """Bounded polling; explicit budgets opt in, CPU fixtures remain unchanged."""
    def __init__(self, settings, device):
        validate_performance(settings)
        self.enabled = any(key in settings for key in ('ram_reserve_mib', 'vram_reserve_mib', 'resource_poll_ms'))
        self.device = torch.device(device)
        self.ram_reserve = settings.get('ram_reserve_mib', 2048) * 1024**2
        self.vram_reserve = settings.get('vram_reserve_mib', 1024) * 1024**2
        self.poll_seconds = settings.get('resource_poll_ms', 250) / 1000.
        self._lock, self._stop = threading.Lock(), threading.Event()
        self._thread = None
        self._report = {'status': 'not_measured' if not self.enabled else 'pending',
                        'enabled': self.enabled, 'device': str(self.device),
                        'cuda_measured': False, 'sample_count': 0,
                        'ram_reserve_bytes': self.ram_reserve if self.enabled else None,
                        'vram_reserve_bytes': self.vram_reserve if self.enabled and self.device.type == 'cuda' else None,
                        'resource_poll_ms': round(self.poll_seconds * 1000), 'sampling_seconds': 0.}
        self._error, self._swap_initial = None, None
        self._started = None

    def _sample(self):
        started = time.monotonic()
        try:
            row = process_tree_resources()
            cuda = None
            if self.device.type == 'cuda':
                if not torch.cuda.is_available():
                    raise RuntimeError('configured CUDA resource monitor has no CUDA device')
                with torch.cuda.device(self.device):
                    free, total = torch.cuda.mem_get_info(self.device)
                    allocated = torch.cuda.memory_allocated(self.device)
                    reserved = torch.cuda.memory_reserved(self.device)
                    cuda = {'cuda_total_bytes': total, 'cuda_free_bytes': free,
                            'cuda_allocated_bytes': allocated, 'cuda_reserved_bytes': reserved,
                            'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(self.device),
                            'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(self.device),
                            'cuda_headroom_bytes': free + max(0, reserved - allocated)}
            with self._lock:
                report = self._report
                report['sample_count'] += 1
                report['status'] = 'measured' if self._error is None else 'budget_exceeded'
                for key in ('ram_available_bytes',):
                    report['min_' + key] = min(row[key], report.get('min_' + key, row[key]))
                for key in ('process_tree_rss_sum_bytes', 'process_tree_uss_sum_bytes'):
                    value = row[key]
                    if value is not None:
                        report['peak_' + key] = max(value, report.get('peak_' + key, value))
                if self._swap_initial is None:
                    self._swap_initial = {key: row[key] for key in ('swap_used_bytes', 'swap_in_bytes', 'swap_out_bytes')}
                for key, initial in self._swap_initial.items():
                    report[key + '_delta'] = row[key] - initial if row[key] is not None and initial is not None else None
                report.update(row)
                if cuda is not None:
                    report['cuda_measured'] = True
                    report['cuda_headroom_semantics'] = 'device_free_plus_unused_process_allocator_reservation'
                    for key in ('cuda_free_bytes', 'cuda_headroom_bytes'):
                        report['min_' + key] = min(cuda[key], report.get('min_' + key, cuda[key]))
                    for key in ('cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes'):
                        cuda[key] = max(cuda[key], report.get(key, cuda[key]))
                    report.update(cuda)
                if row['ram_available_bytes'] < self.ram_reserve:
                    self._error = 'system available RAM below configured reserve'
                if cuda is not None and cuda['cuda_headroom_bytes'] < self.vram_reserve:
                    self._error = 'CUDA usable headroom below configured reserve'
                if self._error:
                    report.update(status='budget_exceeded', reason=self._error)
                report['sampling_seconds'] += time.monotonic() - started
        except Exception as error:
            with self._lock:
                self._error = 'resource sampling failed: ' + str(error)
                self._report.update(status='measurement_failed', reason=self._error)

    def _poll(self):
        while not self._stop.wait(self.poll_seconds):
            self._sample()

    def __enter__(self):
        self._started = time.monotonic()
        if self.enabled:
            self._sample()
            self.check()
            self._thread = threading.Thread(target=self._poll, name='m4human-resource-monitor', daemon=True)
            self._thread.start()
        return self

    def check(self):
        with self._lock:
            if self._error:
                raise RuntimeError(self._error)
        return self.snapshot()

    def snapshot(self):
        with self._lock:
            report = dict(self._report)
        elapsed = time.monotonic() - self._started if self._started is not None else 0.
        report['monitor_elapsed_seconds'] = elapsed
        report['sampling_wall_fraction'] = report['sampling_seconds'] / elapsed if elapsed > 0 else None
        report['sampling_scope'] = 'poll_samples; minima_and_peaks_between_polls_may_be_missed'
        return report

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1., self.poll_seconds * 2))
            if self._thread.is_alive():
                with self._lock:
                    self._error = 'resource sampling did not stop before final measurement'
                    self._report.update(status='measurement_failed',reason=self._error)
                if exc_type is None:
                    self.check()
                return False
        if self.enabled:
            self._sample()
        if exc_type is None:
            self.check()
        return False
