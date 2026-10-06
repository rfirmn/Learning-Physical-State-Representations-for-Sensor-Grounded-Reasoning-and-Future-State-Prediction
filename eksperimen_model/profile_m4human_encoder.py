"""Bounded E train-only performance sweep; no checkpoints or overflow retries.
Timed overflow disqualifies candidates even if training retry could recover.
"""
import argparse
import copy
import gc
from contextlib import nullcontext
import itertools
import json
from pathlib import Path
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import time
import numpy as np
import torch
from torch.utils.data import Subset
import yaml
from eksperimen_model.train_m4human_encoder import make_datasets,collate_samples,to_device,fit_input_normalizer,encoder_group_denominators
from eksperimen_model.models.m4human_encoder import M4HumanSetEncoderV2,encoder_loss
from eksperimen_model.utils.m4human_performance import configure_runtime,make_loader,finish_optimizer_step,ResourceMonitor
from eksperimen_model.utils.m4human_runtime import atomic_json,seed_everything,resource_snapshot,canonical_hash,adamw_parameters
from eksperimen_model.utils.m4human_gates import encoder_readiness_config_hash


def fixed_batch_candidates(effective_batch,micro_batches):
    if isinstance(effective_batch,bool) or not isinstance(effective_batch,int) or effective_batch<1:
        raise ValueError('effective_batch must be a positive integer')
    if any(isinstance(b,bool) or not isinstance(b,int) or b<1 for b in micro_batches):
        raise ValueError('micro batches must be positive integers')
    return [{'micro_batch':b,'accumulation':effective_batch//b} for b in micro_batches if effective_batch%b==0]


def synchronize(device):
    if torch.device(device).type=='cuda':
        torch.cuda.synchronize(device)


def numerical_parity(model,batch,device,amp,atol=.002,rtol=.02):
    """Same weights, sensor and masks; include forward, loss and gradients."""
    values = []
    sensor,target = to_device(batch['sensor'],device),to_device(batch['targets'],device)
    for use_amp in (False,amp):
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16) if use_amp else nullcontext():
            pred = model(**sensor)
        loss = encoder_loss(pred,target,model.pelvis_index)['total']
        loss.backward()
        values.append(torch.cat([pred['P_enc_relative_m'].detach().flatten(),pred['r_enc_m'].detach().flatten(),loss.detach().reshape(1),
                                 *[p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]]).float())
    passed = torch.isfinite(values[1]).all().item() and torch.allclose(*values,atol=atol,rtol=rtol)
    return {'passed':bool(passed),'max_abs_error':float((values[0]-values[1]).abs().max()),'atol':atol,'rtol':rtol,
            'scope':'same_batch_forward_loss_backward_fp32_reference','cuda_measured':torch.device(device).type=='cuda'}


def profile_candidate(dataset,pelvis_index,settings,device,warmup_steps=3,timed_steps=3,reserve_bytes=0):
    resources=dict(settings)
    resources.setdefault('ram_reserve_mib',2048)
    resources.setdefault('vram_reserve_mib',max(1024,(reserve_bytes+1024**2-1)//1024**2))
    resources.setdefault('resource_poll_ms',250)
    with ResourceMonitor(resources,device) as monitor:
        result=_profile_candidate(dataset,pelvis_index,resources,device,warmup_steps,timed_steps,reserve_bytes,monitor)
    result['resource_safety']=monitor.snapshot()
    return result


def _profile_candidate(dataset,pelvis_index,settings,device,warmup_steps,timed_steps,reserve_bytes,monitor):
    if any(isinstance(v,bool) or not isinstance(v,int) or v<minimum for v,minimum in ((warmup_steps,0),(timed_steps,1),(reserve_bytes,0))):
        raise ValueError('invalid profiling bounds')
    configure_runtime(settings,device)
    seed_everything(settings.get('seed',42))
    effective = settings['micro_batch']*settings['accumulation']
    bound = (warmup_steps+timed_steps)*effective
    if len(dataset)<bound:
        raise ValueError(f'profile requires {bound} train samples; reduce bounded steps or effective batch')
    dataset = Subset(dataset,range(bound))
    generator = torch.Generator().manual_seed(settings.get('seed',42))
    loader = make_loader(dataset,settings['micro_batch'],settings,device,collate_fn=collate_samples,generator=generator)
    if not settings.get('profile_normalizer'):
        raise ValueError('profile requires a supplied locked or train-fitted normalizer')
    model = M4HumanSetEncoderV2(pelvis_index,settings['profile_normalizer']).to(device)
    model.train()
    optimizer = torch.optim.AdamW(adamw_parameters(model,settings.get('weight_decay',.01)),lr=settings.get('learning_rate',.0002))
    amp = settings.get('precision')=='fp16_amp_with_grad_scaler'
    scaler = torch.amp.GradScaler('cuda',enabled=amp)
    is_cuda = torch.device(device).type=='cuda'
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        free,total = torch.cuda.mem_get_info(device)
        available_budget = free + torch.cuda.memory_reserved(device)
        if free <= reserve_bytes:
            return {'status':'memory_reserve_rejected','free_bytes':free,'reserve_bytes':reserve_bytes}
    iterator = iter(loader)
    elapsed,wait,samples,updates,skips = 0.,0.,0,0,0
    parity = None
    warmup_skips = 0
    timed_update_seconds = []
    for step in range(warmup_steps+timed_steps):
        if is_cuda and step == warmup_steps:
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
        started = time.perf_counter()
        batches = [next(iterator) for _ in range(settings['accumulation'])]
        monitor.check()
        synchronize(device)
        waited = time.perf_counter()-started
        if parity is None:
            parity = numerical_parity(model,batches[0],device,amp)
            synchronize(device)
            started = time.perf_counter()-waited
        denominators = encoder_group_denominators(batches,pelvis_index)
        if not sum(denominators.values()):
            raise ValueError('bounded profile has unsupervised accumulation group')
        optimizer.zero_grad(set_to_none=True)
        for batch in batches:
            sensor,target = to_device(batch['sensor'],device,settings.get('non_blocking',False)),to_device(batch['targets'],device,settings.get('non_blocking',False))
            with torch.autocast('cuda',dtype=torch.float16) if amp else nullcontext():
                pred = model(**sensor)
            loss = encoder_loss(pred,target,pelvis_index,settings.get('smooth_l1_beta_m',.05))
            weighted = sum(loss[k]*loss[k+'_count']/denominators[k] if denominators[k] else loss[k]*0 for k in denominators)
            if not torch.isfinite(weighted):
                raise ValueError('nonfinite profile loss')
            scaler.scale(weighted).backward()
        _,did_step = finish_optimizer_step(optimizer,scaler,model.parameters(),settings.get('clip_grad_norm',1.))
        synchronize(device)
        monitor.check()
        if step<warmup_steps:
            warmup_skips += int(not did_step)
        if step>=warmup_steps:
            duration = time.perf_counter()-started
            timed_update_seconds.append(duration)
            elapsed += duration
            wait += waited
            samples += effective
            updates += int(did_step); skips += int(not did_step)
    peak = torch.cuda.max_memory_allocated(device) if is_cuda else None
    reserved = torch.cuda.max_memory_reserved(device) if is_cuda else None
    memory_ok = not is_cuda or reserved+reserve_bytes <= available_budget
    return {'status':'ok' if memory_ok else 'memory_reserve_rejected','samples':samples,'loaded_samples_bound':bound,
            'warmup_steps':warmup_steps,'timed_steps':timed_steps,'seconds':elapsed,'samples_per_second':samples/elapsed,
            'loader_wait_seconds':wait,'timed_logical_update_seconds':timed_update_seconds,
            'median_logical_update_seconds':float(np.median(timed_update_seconds)),
            'p95_logical_update_seconds':float(np.percentile(timed_update_seconds,95)),'successful_updates':updates,'overflow_skips':skips,'warmup_overflow_skips':warmup_skips,'peak_allocated_bytes':peak,
            'peak_reserved_bytes':reserved,'available_cuda_budget_bytes':available_budget if is_cuda else None,'reserve_bytes':reserve_bytes,'numerical_parity':parity,
            'resources':resource_snapshot(),'normalizer_policy':'supplied_or_fitted_train_only'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True); p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--micro-batches',type=int,nargs='+',default=[4,8,16,32,64,128,256,512,1024])
    p.add_argument('--workers',type=int,nargs='+',default=[0,2,4])
    p.add_argument('--precisions',nargs='+',choices=['fp32','fp16_amp_with_grad_scaler'],default=['fp32','fp16_amp_with_grad_scaler'])
    policy=p.add_mutually_exclusive_group()
    policy.add_argument('--effective-batch',type=int,help='Explicit new batch policy; default preserves config effective batch')
    policy.add_argument('--effective-batches',type=int,nargs='+',help='Explicit batch-policy sweep, e.g.16 64 128 256 512 1024; compare selected policy on train/validation')
    p.add_argument('--selected-config',help='Write full measured recommended config to a new YAML path')
    p.add_argument('--warmup-steps',type=int,default=3,help='Excluded warmup updates; increase if initial GradScaler overflow persists (recommendation rejects timed overflow)'); p.add_argument('--timed-steps',type=int,default=3)
    p.add_argument('--reserve-mib',type=int,default=1024); p.add_argument('--ram-reserve-mib',type=int,default=2048)
    p.add_argument('--synthetic-smoke',action='store_true')
    a=p.parse_args()
    for output in (a.output,a.selected_config):
        if output and Path(output).exists():
            raise FileExistsError(output)
    config=yaml.safe_load(Path(a.config).read_text())
    original=config['training']['micro_batch']*config['training']['accumulation']
    policies=list(dict.fromkeys(a.effective_batches or [a.effective_batch if a.effective_batch is not None else original]))
    candidates=[dict(candidate,effective_batch=effective) for effective in policies for candidate in fixed_batch_candidates(effective,a.micro_batches)]
    if not candidates:
        p.error('no micro batch divides effective batch; supply divisors or explicitly choose --effective-batch')
    outcomes=[]
    train=val=None
    prep_started=time.perf_counter()
    try:
        preparation_settings=dict(config['training'],ram_reserve_mib=a.ram_reserve_mib,vram_reserve_mib=a.reserve_mib,resource_poll_ms=config['training'].get('resource_poll_ms',250))
        with ResourceMonitor(preparation_settings,a.device) as preparation_monitor:
            train,val,joint_map,lineage=make_datasets(config,allow_synthetic=a.synthetic_smoke)
            preparation_monitor.check()
            normalizer=config.get('normalizer') or fit_input_normalizer(train.sensor_dataset)
        prep_seconds=time.perf_counter()-prep_started
        for batch,workers,precision in itertools.product(candidates,a.workers,a.precisions):
            settings=dict(config['training'],**batch,workers=workers,precision=precision,
                          ram_reserve_mib=a.ram_reserve_mib,vram_reserve_mib=a.reserve_mib,resource_poll_ms=config['training'].get('resource_poll_ms',250),
                          persistent_workers=workers>0 and config['training'].get('persistent_workers',False),profile_normalizer=normalizer)
            result={'settings':{key:value for key,value in settings.items() if key!='profile_normalizer'}}
            try:
                result.update(profile_candidate(train,joint_map['pelvis_index'],settings,a.device,a.warmup_steps,a.timed_steps,a.reserve_mib*1024**2))
            except torch.cuda.OutOfMemoryError as exc:
                result.update(status='oom',error=str(exc))
            except (ValueError,RuntimeError) as exc:
                result.update(status='rejected',error=str(exc))
            outcomes.append(result)
            gc.collect()
            if torch.device(a.device).type=='cuda': torch.cuda.empty_cache()
    finally:
        if train is not None: train.sensor_dataset.reader.close()
        if val is not None: val.sensor_dataset.reader.close()
    eligible=[r for r in outcomes if r['status']=='ok' and r['numerical_parity']['passed'] and not r['overflow_skips']]
    recommended_settings = dict(max(eligible,key=lambda r:r['samples_per_second'])['settings']) if eligible else None
    if recommended_settings is not None:
        recommended_settings.pop('profile_normalizer',None)
    recommended_config = dict(copy.deepcopy(config),training=recommended_settings) if recommended_settings else None
    if recommended_config:
        recommended_config.pop('real_smoke_passed',None)
        recommended_config.pop('resource_profile_passed',None)
        if 'performance_policy' in recommended_config:
            policy=recommended_config['performance_policy']
            policy.setdefault('original_effective_batch',original)
            policy.update(status='resource_profile_selected',effective_batch=recommended_settings['effective_batch'],
                          batch_policy_change='explicit_new_protocol_requires_validation' if recommended_settings['effective_batch']!=policy['original_effective_batch'] else 'preserve_baseline_effective_batch',
                          learning_rate_policy='unchanged_no_automatic_scaling',
                          rationale='measured_throughput_with_resource_reserve_and_precision_parity; quality_requires_pilot')
    effective=recommended_settings['effective_batch'] if recommended_settings else None
    report={'data_kind':config['data_kind'],'kind':'bounded_train_only_resource_profile','scientific_gates_passed':False,'config_hash':canonical_hash(config),
            'lineage':lineage,'normalizer_hash':canonical_hash(normalizer),'preparation_seconds':prep_seconds,'normalizer_policy':'supplied_locked' if config.get('normalizer') else 'fitted_train_only','device':a.device,'original_effective_batch':original,'effective_batch':effective,
            'batch_policy_changed':effective is not None and effective!=original,'effective_batch_candidates':policies,
            'excluded_micro_batches':{str(policy):[b for b in a.micro_batches if policy%b] for policy in policies},
            'preparation_resource_safety':preparation_monitor.snapshot(),
            'outcomes':outcomes,'recommended_settings':recommended_settings,
            'recommended_config_hash':canonical_hash(recommended_config) if recommended_config else None,
            'readiness_config_hash':encoder_readiness_config_hash(recommended_config) if recommended_config else None}
    if a.selected_config and recommended_config:
        target=Path(a.selected_config)
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('x',encoding='utf-8') as handle:
            yaml.safe_dump(recommended_config,handle,sort_keys=False)
        report['selected_config_path']=str(target)
    atomic_json(a.output,report); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
