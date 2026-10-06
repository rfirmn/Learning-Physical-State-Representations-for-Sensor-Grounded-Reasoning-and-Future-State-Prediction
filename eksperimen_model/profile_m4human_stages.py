"""Bounded train-only M/C/probe/L hardware profiling; emits no scientific checkpoints."""
import argparse
import copy
import gc
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
from eksperimen_model import m4human_training as physical
from eksperimen_model.train_m4human_projector import _projector_logical_step
from eksperimen_model.profile_m4human_encoder import fixed_batch_candidates
from eksperimen_model.utils.m4human_performance import configure_runtime,make_loader
from eksperimen_model.utils.m4human_runtime import (atomic_json,canonical_hash,read_json,seed_everything,
    adamw_parameters,state_dict_hash,load_checkpoint,file_sha256)


def sync(device):
    if torch.device(device).type=='cuda': torch.cuda.synchronize(device)


def cleanup():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def _settings(config,stage):
    return config['training'] if stage=='L' else config


def _context(settings,device):
    enabled=settings['precision']!='fp32'
    return torch.autocast(torch.device(device).type,dtype=torch.bfloat16 if settings['precision']=='bf16' else torch.float16,enabled=enabled)


def _signature(model,batch,stage,condition,stats,config,device,precision):
    """Unclipped, unscaled same-input forward/loss/backward witness."""
    settings=dict(_settings(config,stage),precision=precision)
    model.train(); model.zero_grad(set_to_none=True)
    seed_everything(settings.get('seed',42))
    with _context(settings,device):
        if stage=='L':
            inputs,labels=model.assemble(batch,training=True)
            from eksperimen_model.models.m4human_projector import answer_only_loss
            logits=model.llm(**inputs,use_cache=False).logits
            loss=answer_only_loss(logits,labels)
            forward=logits.detach().float().flatten().cpu()
        else:
            values=physical._device_batch(batch,device)
            loss,_,prediction=physical._forward(model,stage,condition,values['sensor'],values['targets'],stats,config)
            forward=torch.cat([v.detach().float().flatten().cpu() for v in prediction.values() if isinstance(v,torch.Tensor) and v.is_floating_point()])
    if not torch.isfinite(loss): raise ValueError('nonfinite numerical parity loss')
    loss.backward()
    parameters=model.projector.parameters() if stage=='L' else model.parameters()
    gradients=torch.cat([p.grad.detach().float().flatten().cpu() for p in parameters if p.grad is not None])
    return {'forward':forward,'loss':loss.detach().float().reshape(1).cpu(),'gradients':gradients}


def parity(reference,candidate,device,batch_size,atol=.002,rtol=.02):
    comparisons={}
    for key in reference:
        a,b=reference[key],candidate[key]
        comparisons[key]={'passed':bool(torch.isfinite(b).all() and torch.allclose(a,b,atol=atol,rtol=rtol)),
                          'max_abs_error':float((a-b).abs().max())}
    return {'passed':all(v['passed'] for v in comparisons.values()),'components':comparisons,'atol':atol,'rtol':rtol,
            'cuda_measured':torch.device(device).type=='cuda','reference_precision':'fp32','reference_batch_size':batch_size,
            'scope':'same_train_input_initial_weights_forward_loss_unclipped_backward; no batch-size parity claim'}


def _monitor(settings,device):
    from eksperimen_model.utils.m4human_performance import ResourceMonitor
    resources=dict(settings)
    resources.setdefault('ram_reserve_mib',2048); resources.setdefault('vram_reserve_mib',1024); resources.setdefault('resource_poll_ms',250)
    return ResourceMonitor(resources,device)


def profile_stage(dataset,builder,stage,condition,config,stats,device,warmup_steps=3,timed_steps=3):
    """Measures loaded train logical groups, all microsteps and actual optimizer updates."""
    settings=_settings(config,stage); device=torch.device(device)
    configure_runtime(settings,device)
    if type(warmup_steps)!=int or warmup_steps<0 or type(timed_steps)!=int or timed_steps<1:
        raise ValueError('invalid profile bounds')
    effective=settings['micro_batch']*settings['accumulation']; bound=effective*(warmup_steps+timed_steps)
    if len(dataset)<bound: raise ValueError(f'need {bound} train samples; reduce steps/batch explicitly')
    if stage=='L' and settings.get('workers',0): raise ValueError('L loads QA synchronously; workers must be zero')
    model=optimizer=scaler=loader=iterator=None
    try:
        parity_batch=[dataset[0]] if stage=='L' else physical._collate([dataset[i] for i in range(settings['micro_batch'])])
        # Release the fp32 model before constructing the timed model, important for Qwen VRAM.
        with _monitor(settings,device) as preparation_monitor:
            preparation_monitor.check()
            model=builder('fp32')
            reference=_signature(model,parity_batch,stage,condition,stats,config,device,'fp32')
            del model; model=None; cleanup()
            model=builder(settings['precision'])
            candidate=_signature(model,parity_batch,stage,condition,stats,config,device,settings['precision'])
            numerical=parity(reference,candidate,device,1 if stage=='L' else settings['micro_batch'])
            del reference,candidate
            preparation_monitor.check()
        preparation_resources=preparation_monitor.snapshot()
        model.zero_grad(set_to_none=True); model.train()
        initial_hash=state_dict_hash(model.projector if stage=='L' else model)
        frozen_hash=state_dict_hash(model.llm) if stage=='L' else None
        parameters=model.projector if stage=='L' else model
        groups=adamw_parameters(parameters,settings['weight_decay']) if stage=='L' else adamw_parameters(parameters)
        optimizer=torch.optim.AdamW(groups,lr=settings['learning_rate'])
        scaler=torch.amp.GradScaler('cuda',enabled=settings['precision'] in ('fp16_amp','fp16_amp_grad_scaler'))
        subset=Subset(dataset,range(bound))
        if stage!='L':
            generator=torch.Generator().manual_seed(settings.get('seed',42))
            loader=make_loader(subset,settings['micro_batch'],settings,device,collate_fn=physical._collate,generator=generator)
            iterator=iter(loader)
        elapsed=wait=samples=updates=overflows=warmup_overflows=0
        durations=[]
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        with _monitor(settings,device) as monitor:
            for step in range(warmup_steps+timed_steps):
                monitor.check(); sync(device); started=time.perf_counter()
                if stage=='L': logical=[subset[step*effective+i] for i in range(effective)]
                else: logical=[next(iterator) for _ in range(settings['accumulation'])]
                waited=time.perf_counter()-started
                if stage=='L':
                    _,_,retries=_projector_logical_step(model,logical,settings,optimizer,scaler,device)
                else:
                    result=physical._logical_training_step(model,stage,condition,logical,stats,config,device,optimizer,scaler)
                    if result is None: raise ValueError('no supervision in bounded train group')
                    retries=result[-1]
                sync(device); monitor.check(); duration=time.perf_counter()-started
                if step<warmup_steps: warmup_overflows+=retries
                else:
                    durations.append(duration); elapsed+=duration; wait+=waited; samples+=effective; updates+=1; overflows+=retries
        resources=monitor.snapshot()
        frozen_verified=stage!='L' or (state_dict_hash(model.llm)==frozen_hash and all(p.grad is None and not p.requires_grad for p in model.llm.parameters()))
        return {'status':'ok','samples':samples,'loaded_samples_bound':bound,'warmup_steps':warmup_steps,'timed_steps':timed_steps,
                'seconds':elapsed,'samples_per_second':samples/elapsed,'loader_wait_seconds':wait,'timed_logical_update_seconds':durations,
                'median_logical_update_seconds':float(np.median(durations)),'p95_logical_update_seconds':float(np.percentile(durations,95)),
                'successful_updates':updates,'overflow_skips':overflows,'warmup_overflow_skips':warmup_overflows,
                'numerical_parity':numerical,'initialization_hash':initial_hash,'frozen_weights_verified':frozen_verified,
                'resources':resources,'preparation_resources':preparation_resources,'workers_active':settings.get('workers',0) if stage!='L' else 0,
                'sustained_resource_profile':timed_steps>=100 and updates==timed_steps and not overflows and numerical['passed'] and frozen_verified,
                'scientific_gates_passed':False}
    finally:
        if iterator is not None and hasattr(iterator,'_shutdown_workers'): iterator._shutdown_workers()
        del model,optimizer,scaler,loader,iterator
        cleanup()


def profile_generation(dataset,builder,settings,device,batch_size,warmup_steps=1,timed_steps=3,max_new_tokens=64):
    configure_runtime(settings,device)
    if settings.get('workers',0):raise ValueError('L generation uses synchronous QA loading; workers must be zero')
    if type(batch_size)!=int or batch_size<1: raise ValueError('generation batch must be positive')
    if type(warmup_steps)!=int or warmup_steps<0 or type(timed_steps)!=int or timed_steps<1 or type(max_new_tokens)!=int or max_new_tokens<1:
        raise ValueError('invalid bounded generation settings')
    bound=batch_size*(warmup_steps+timed_steps)
    if len(dataset)<bound: raise ValueError(f'need {bound} train QA for bounded generation')
    model=None
    try:
        model=builder(settings['precision']); frozen=state_dict_hash(model.llm)
        durations=[]; tokens=0
        if torch.device(device).type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        with _monitor(settings,device) as monitor:
            for step in range(warmup_steps+timed_steps):
                monitor.check(); sync(device); started=time.perf_counter()
                samples=[dataset[step*batch_size+i] for i in range(batch_size)]
                with _context(settings,device): records,_=model.generate(samples,max_new_tokens=max_new_tokens)
                if len(records)!=batch_size: raise ValueError('generation batch count mismatch')
                sync(device); monitor.check()
                if step>=warmup_steps:
                    durations.append(time.perf_counter()-started); tokens+=sum(len(r['raw_ids']) for r in records)
        resources=monitor.snapshot()
        elapsed=sum(durations)
        return {'status':'ok','generation_batch_size':batch_size,'samples':timed_steps*batch_size,'generated_tokens':tokens,
                'seconds':elapsed,'samples_per_second':timed_steps*batch_size/elapsed,'tokens_per_second':tokens/elapsed,
                'median_batch_seconds':float(np.median(durations)),'p95_batch_seconds':float(np.percentile(durations,95)),
                'resources':resources,'frozen_weights_verified':state_dict_hash(model.llm)==frozen,
                'scope':'train_questions_only_generation_performance; no scientific evaluation'}
    finally:
        del model; cleanup()


def _physical_dataset(config,stage,condition):
    metadata=read_json(Path(config['source_cache'])/'metadata.json')
    # Profiling opens audited upstream input; precision parity is measured below.
    physical._scientific_gate(dict(config,precision='fp32'),metadata)
    if stage=='M':
        from eksperimen_model.datasets.m4human_state_dataset import EncodedStateWindowDataset
        source=EncodedStateWindowDataset(config['source_cache'],length=32,stride=config.get('train_stride',8),split='train',targets_dir=config['target_cache'],allow_debug=config['data_kind']=='synthetic')
    else: source=physical.WindowTensorCache(config['source_cache'],'train',config['lineage'],require_targets=True)
    dataset=physical._PhysicalWindows(source,config['max_gap_s'],stage!='M')
    if stage=='probe' and condition!='M':
        dataset=physical._TokenWindows(dataset,config['token_cache'],config['lineage'],condition)
        if dataset.tokens.metadata.get('K')!=config['K']:raise ValueError('probe token budget differs from config')
    stats=physical._fit_statistics(dataset,stage,config['pelvis_index'])
    if stage!='M':
        locked=read_json(config['normalizer_path'])
        if locked.get('hash')!=canonical_hash({k:v for k,v in locked.items() if k!='hash'}) or metadata['lineage'].get('normalizer_hash')!=locked['hash']:
            raise ValueError('foreign physical normalizer')
        for key in ('s_v_joint','s_v_root'): stats[key]=locked[key]
        stats['physical_normalizer_hash']=locked['hash']; stats['hash']=canonical_hash({k:v for k,v in stats.items() if k!='hash'})
    return dataset,stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=['M','C','probe','L']);p.add_argument('--condition')
    p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--selected-config')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--micro-batches',type=int,nargs='+',default=[1,2,4,8,16,32,64]);p.add_argument('--workers',type=int,nargs='+',default=[0,2,4])
    p.add_argument('--precisions',nargs='+');p.add_argument('--effective-batch',type=int)
    p.add_argument('--warmup-steps',type=int,default=3);p.add_argument('--timed-steps',type=int,default=3)
    p.add_argument('--qa');p.add_argument('--u-cache');p.add_argument('--initial');p.add_argument('--synthetic-smoke',action='store_true')
    p.add_argument('--generation-batches',type=int,nargs='+',default=[1,2,4,8,16,32]);p.add_argument('--max-new-tokens',type=int,help='Explicit generation length policy; defaults to config.generation.max_new_tokens')
    a=p.parse_args();config=yaml.safe_load(Path(a.config).read_text());stage=a.stage;settings=_settings(config,stage)
    if a.max_new_tokens is None:a.max_new_tokens=config.get('generation',{}).get('max_new_tokens',64)
    conditions={'M':(None,'M'),'C':('C_base','C_kin'),'probe':('M','U_base','U_kin'),'L':('C_base','C_kin')}
    if a.condition not in conditions[stage]: p.error('explicit valid condition required for this stage')
    if config.get('data_kind')=='synthetic' and not a.synthetic_smoke: p.error('synthetic profiling requires --synthetic-smoke')
    settings.setdefault('ram_reserve_mib',2048);settings.setdefault('vram_reserve_mib',1024);settings.setdefault('resource_poll_ms',250)
    original=settings['micro_batch']*settings['accumulation'];effective=a.effective_batch if a.effective_batch is not None else original
    candidates=fixed_batch_candidates(effective,a.micro_batches)
    if not candidates:p.error('no micro batch divides chosen effective batch')
    precision_values=a.precisions or (['fp32','fp16_amp_grad_scaler','bf16'] if stage=='L' else ['fp32','fp16_amp'])
    allowed={'fp32','fp16_amp_grad_scaler','bf16'} if stage=='L' else {'fp32','fp16_amp'}
    if not set(precision_values)<=allowed:p.error('invalid stage precision')
    preparation=time.perf_counter();stats=None
    with _monitor(dict(settings,precision='fp32'),a.device) as input_monitor:
        if stage=='L':
            if not all((a.qa,a.u_cache,a.initial)):p.error('L requires --qa --u-cache --initial')
            from eksperimen_model.datasets.m4human_qa_dataset import M4HumanQADataset
            from eksperimen_model.models.m4human_projector import load_frozen_qwen
            dataset=M4HumanQADataset(a.qa,a.u_cache,'train',a.condition,config['lineage'])
            config=dict(config,data_kind=dataset.metadata['data_kind'])
            if dataset.metadata.get('data_kind')!='m4human' or dataset.metadata.get('scientific_eligible') is not True:raise ValueError('L requires audited real exact-U cache')
            if dataset.metadata['K']!=config['primary_budget']:raise ValueError('L token budget differs')
            expected=dict(config['lineage'],llm_revision=config['llm']['revision'],seed=settings['seed'],role='paired_initial')
            def builder(precision):
                seed_everything(settings['seed']);model=load_frozen_qwen(config['llm'],a.device,precision)
                model.checkpoint_language=settings.get('activation_checkpointing',False)
                model.max_prefix_tokens=settings['max_prefix_tokens'];model.max_total_tokens=settings['max_total_tokens']
                load_checkpoint(a.initial,model.projector,expected);return model
        else:
            dataset,stats=_physical_dataset(config,stage,a.condition)
            def builder(precision):
                seed_everything(settings['seed']);return physical._models(stage,stats,config).to(a.device)
    input_resources=input_monitor.snapshot()
    preparation=time.perf_counter()-preparation;outcomes=[]
    worker_values=[0] if stage=='L' else a.workers
    for batch,workers,precision in itertools.product(candidates,worker_values,precision_values):
        chosen=dict(settings,**batch,effective_batch=effective,workers=workers,precision=precision,
                    persistent_workers=workers>0 and settings.get('persistent_workers',False))
        candidate=copy.deepcopy(config)
        if stage=='L':candidate['training']=chosen
        else:candidate.update(chosen)
        outcome={'settings':chosen}
        try:outcome.update(profile_stage(dataset,builder,stage,a.condition,candidate,stats,a.device,a.warmup_steps,a.timed_steps))
        except (ValueError,RuntimeError) as exc:
            outcome.update(status='oom' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'rejected',error=str(exc));exc.__traceback__=None
        cleanup();outcomes.append(outcome)
    eligible=[r for r in outcomes if r['status']=='ok' and r['numerical_parity']['passed'] and not r['overflow_skips'] and r['frozen_weights_verified']]
    chosen=max(eligible,key=lambda row:row['samples_per_second'])['settings'] if eligible else None
    generation=[]
    if stage=='L' and chosen:
        for batch_size in a.generation_batches:
            try:result=profile_generation(dataset,builder,chosen,a.device,batch_size,a.warmup_steps,a.timed_steps,a.max_new_tokens)
            except (ValueError,RuntimeError) as exc:
                result={'status':'oom' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'rejected','generation_batch_size':batch_size,'error':str(exc)};exc.__traceback__=None
            cleanup();generation.append(result)
    resolved=copy.deepcopy(config)
    if chosen:
        if stage=='L':resolved['training']=chosen
        else:resolved.update(chosen)
        if stage in ('C','probe') and chosen!=settings:
            paired=Path(config['paired_contract_path']);resolved['paired_contract_path']=str(paired.with_name(paired.stem+'_profile_'+canonical_hash(chosen)[:12]+paired.suffix))
    generation_eligible=[r for r in generation if r['status']=='ok' and r['frozen_weights_verified']]
    generation_batch=max(generation_eligible,key=lambda row:row['samples_per_second'])['generation_batch_size'] if generation_eligible else None
    if chosen:
        if generation_batch is not None:
            resolved.setdefault('generation',{}).update(batch_size=generation_batch,max_new_tokens=a.max_new_tokens)
        resolved['performance_profile_policy']={'status':'measured_screening' if a.timed_steps<100 else 'measured_sustained',
            'scientific_gates_passed':False,'original_effective_batch':original,'effective_batch':effective,
            'batch_policy_changed':effective!=original,'paired_contract_requires_regeneration':stage in ('C','probe') and chosen!=settings,
            'learning_rate_policy':'unchanged_no_automatic_scaling'}
    report={'kind':'bounded_train_only_stage_resource_profile','stage':stage,'condition':a.condition,'data_kind':config.get('data_kind'),
            'device':a.device,'preparation_seconds':preparation,'input_preparation_resources':input_resources,'normalizer_hash':stats['hash'] if stats else file_sha256(a.initial),
            'original_effective_batch':original,'effective_batch':effective,'batch_policy_changed':effective!=original,
            'paired_contract_requires_regeneration':stage in ('C','probe') and chosen is not None and chosen!=settings,
            'outcomes':outcomes,'recommended_settings':chosen,'recommended_config_hash':canonical_hash(resolved) if chosen else None,
            'generation_outcomes':generation,'recommended_generation_batch':generation_batch,'generation_max_new_tokens':a.max_new_tokens,
            'selection_status':'no_valid_generation_candidate' if stage=='L' and chosen and generation_batch is None else ('selected' if chosen else 'no_valid_training_candidate'),
            'scientific_gates_passed':False,'scientific_freeze_eligible':False,'screening_only':a.timed_steps<100}
    atomic_json(a.output,report)
    if a.selected_config and chosen and (stage!='L' or generation_batch is not None):
        target=Path(a.selected_config)
        with target.open('x',encoding='utf-8') as handle:yaml.safe_dump(resolved,handle,sort_keys=False)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
