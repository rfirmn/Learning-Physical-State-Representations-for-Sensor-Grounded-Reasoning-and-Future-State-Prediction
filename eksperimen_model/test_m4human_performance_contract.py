"""Regression checks for performance policy and recoverable scaled overflow."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import yaml
import torch
from torch.utils.data import TensorDataset
from eksperimen_model.utils.m4human_performance import make_loader,finish_optimizer_step,validate_performance
from eksperimen_model.profile_m4human_encoder import fixed_batch_candidates,profile_candidate


def main():
    expected_baseline = {'encoder':16,'encoder_run':16,'motion':32,'compressor':16,'probes':16,'projector':16}
    paths = sorted(Path('eksperimen_model/configs').glob('m4human*.yaml'))
    assert len(paths) == 11
    for path in paths:
        config = yaml.safe_load(path.read_text())
        settings = config.get('training',config)
        if '_candidate' not in path.stem:
            assert (settings['ram_reserve_mib'],settings['vram_reserve_mib'],settings['resource_poll_ms']) == (2048,1024,250)
            stage = path.stem.removeprefix('m4human_')
            assert settings['micro_batch']*settings['accumulation'] == expected_baseline[stage]
        else:
            assert (settings['ram_reserve_mib'],settings['vram_reserve_mib'],settings['resource_poll_ms']) == (128,128,250)
            assert config['performance_policy']['status'] in ('unmeasured_candidate', 'unmeasured_capacity_candidate')
            assert settings['micro_batch']*settings['accumulation'] == settings['effective_batch']
    # Matched C/probe/L protocols retain K16 and effective16 across candidates.
    for stage in ('compressor','probes','projector'):
        candidate = yaml.safe_load(Path(f'eksperimen_model/configs/m4human_{stage}_rtx3060_candidate.yaml').read_text())
        assert candidate.get('K',candidate.get('primary_budget')) == 16
    parameter=torch.nn.Parameter(torch.tensor(1.))
    optimizer=torch.optim.SGD([parameter],lr=.1)
    scaler=torch.amp.GradScaler('cpu',init_scale=8.)
    scaler.scale(parameter*torch.tensor(float('inf'))).backward()
    norm,did_step=finish_optimizer_step(optimizer,scaler,[parameter],1.)
    assert norm is None and not did_step and parameter.item()==1. and scaler.get_scale()==4.
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(parameter.square()).backward()
    norm,did_step=finish_optimizer_step(optimizer,scaler,[parameter],1.)
    assert did_step and norm==2. and parameter.item()<1.
    parameter.grad=torch.tensor(float('nan'))
    try: finish_optimizer_step(optimizer,torch.amp.GradScaler('cpu',enabled=False),[parameter],1.)
    except RuntimeError: pass
    else: raise AssertionError('fp32 nonfinite gradients accepted')
    a=torch.nn.Parameter(torch.tensor(0.)); b=torch.nn.Parameter(torch.tensor(0.))
    grouped_optimizer=torch.optim.SGD([a,b],lr=1.)
    a.grad=torch.tensor(3.); b.grad=torch.tensor(4.)
    norm,did_step=finish_optimizer_step(grouped_optimizer,torch.amp.GradScaler('cpu',enabled=False),[a,b],1.,clipping_groups=[([a],1.),([b],2.)])
    assert did_step and norm==4. and torch.isclose(a,torch.tensor(-1.)) and torch.isclose(b,torch.tensor(-2.))
    grouped_optimizer.zero_grad(set_to_none=True)
    grouped_scaler=torch.amp.GradScaler('cpu',init_scale=8.)
    grouped_scaler.scale(a*3.+b*torch.tensor(float('inf'))).backward()
    before=(a.detach().clone(),b.detach().clone())
    norm,did_step=finish_optimizer_step(grouped_optimizer,grouped_scaler,[a,b],1.,clipping_groups=[([a],1.),([b],2.)])
    assert not did_step and norm is None and torch.equal(a,before[0]) and torch.equal(b,before[1])
    assert a.grad.item()==3.  # Overflow check precedes every group clipping.
    for groups in ([],[([],1.)],[([a],float('nan'))],[([a],0.)]):
        try: finish_optimizer_step(grouped_optimizer,torch.amp.GradScaler('cpu',enabled=False),[a],1.,clipping_groups=groups)
        except ValueError: pass
        else: raise AssertionError('invalid clipping group accepted')
    loader=make_loader(TensorDataset(torch.arange(5)),2,{'workers':0},'cpu')
    assert loader.prefetch_factor is None and loader.multiprocessing_context is None
    for settings in ({'workers':True},{'persistent_workers':True},{'prefetch_factor':0},{'pin_memory':1},{'micro_batch':4,'accumulation':2,'effective_batch':16}):
        try: validate_performance(settings)
        except ValueError: pass
        else: raise AssertionError(settings)
    assert fixed_batch_candidates(16,[4,8,16,32])==[{'micro_batch':4,'accumulation':4},{'micro_batch':8,'accumulation':2},{'micro_batch':16,'accumulation':1}]
    # Masked component counts must produce the same effective-group loss across partitions.
    x=torch.tensor([1.,2.,3.,4.]); y=torch.zeros(4); mask=torch.tensor([True,False,True,True])
    gradients=[]
    for micro in (1,2,4):
        weight=torch.tensor(.5,requires_grad=True)
        for start in range(0,4,micro):
            local=mask[start:start+micro]
            if local.any():
                loss=torch.nn.functional.smooth_l1_loss(weight*x[start:start+micro][local],y[start:start+micro][local],reduction='sum')/mask.sum()
                loss.backward()
        gradients.append(weight.grad)
    assert all(torch.allclose(gradients[0],g) for g in gradients)
    sample={'sensor':{'rpc_m':torch.ones(4,512,4),'point_mask':torch.ones(4,512,dtype=torch.bool),'time_s':torch.arange(4).double()/12},
            'targets':{'annotation_joint_valid':torch.ones(22,dtype=torch.bool),'annotation_root_valid':torch.tensor(True),
                       'joint_relative_m':torch.zeros(22,3),'root_m':torch.zeros(3)},'provenance':{}}
    from eksperimen_model.train_m4human_encoder import collate_samples,encoder_group_denominators,_encoder_logical_step
    from eksperimen_model.models.m4human_encoder import M4HumanSetEncoderV2
    torch.set_num_threads(1)
    invalid_cloud=copy.deepcopy(sample); invalid_cloud['sensor']['point_mask'][1]=False
    invalid_time=copy.deepcopy(sample); invalid_time['sensor']['time_s'][2]=invalid_time['sensor']['time_s'][1]
    mixed=[sample,invalid_cloud,invalid_time]
    expected={'pose':63,'root':3}
    assert encoder_group_denominators([collate_samples(mixed)],0)==expected
    gradients=[]
    initial_model=M4HumanSetEncoderV2(0)
    for partition in ([collate_samples(mixed)],[collate_samples([row]) for row in mixed]):
        model=copy.deepcopy(initial_model); opt=torch.optim.SGD(model.parameters(),lr=0.)
        norm,sums,counts,exposure,retries=_encoder_logical_step(model,opt,torch.amp.GradScaler('cpu',enabled=False),partition,expected,
                          {'smooth_l1_beta_m':.05,'clip_grad_norm':100.},'cpu')
        assert counts==expected and exposure==3 and retries==0
        gradients.append(torch.cat([parameter.grad.flatten() for parameter in model.parameters() if parameter.grad is not None]))
    assert torch.allclose(gradients[0],gradients[1],atol=2e-6,rtol=2e-5)
    # Inject one finite-loss scaled gradient overflow. Replay preserves batch identity and RNG.
    model=copy.deepcopy(initial_model); opt=torch.optim.SGD(model.parameters(),lr=.01)
    cpu_scaler=torch.amp.GradScaler('cpu',init_scale=8.)
    calls=[]; rng_draws=[]; retries_logged=[]
    def forward_record(module,args): rng_draws.append(torch.rand(()).item())
    forward_handle=model.register_forward_pre_hook(forward_record)
    def first_overflow(gradient):
        calls.append(1)
        return torch.full_like(gradient,float('inf')) if len(calls)==1 else gradient
    gradient_handle=next(model.parameters()).register_hook(first_overflow)
    logger=SimpleNamespace(log=lambda event,**values:retries_logged.append((event,values)))
    valid_batch=collate_samples([dict(sample,provenance={'frame_uid':'same_anchor'})])
    norm,sums,counts,exposure,retries=_encoder_logical_step(model,opt,cpu_scaler,[valid_batch],expected,
                         {'smooth_l1_beta_m':.05,'clip_grad_norm':1.},'cpu',logger=logger)
    gradient_handle.remove(); forward_handle.remove()
    assert retries==1 and exposure==1 and counts==expected and cpu_scaler.get_scale()==4.
    assert rng_draws[0]==rng_draws[1] and len(rng_draws)==2
    assert retries_logged[0][0]=='overflow_retry' and retries_logged[0][1]['frame_uids']==['same_anchor']
    # Tiny CPU fixtures declare1MiB headroom; real hardware configs retain2GiB.
    result=profile_candidate([sample,sample],0,{'micro_batch':1,'accumulation':1,'precision':'fp32','workers':0,'cpu_threads':1,'ram_reserve_mib':1,'profile_normalizer':{'mean':[0.]*4,'std':[1.]*4}},'cpu',1,1)
    assert result['samples']==1 and result['loaded_samples_bound']==2 and result['successful_updates']==1
    assert result['numerical_parity']['passed'] and result['samples_per_second']>0
    assert len(result['timed_logical_update_seconds'])==1 and result['median_logical_update_seconds']==result['p95_logical_update_seconds']>0
    from eksperimen_model.check_m4human_encoder_overfit import overfit
    diagnostic_sample=copy.deepcopy(sample)
    diagnostic_sample['provenance']['frame_uid']='fixed_train_fixture'
    diagnostic=overfit([diagnostic_sample],0,{'mean':[0.]*4,'std':[1.]*4},
                      {'micro_batch':1,'accumulation':1,'precision':'fp32','workers':0},'cpu',1,12,.25)
    assert diagnostic['status']=='passed' and diagnostic['loss_ratio']<.25
    assert diagnostic['frame_uids']==['fixed_train_fixture'] and diagnostic['successful_updates']==12
    assert diagnostic['scientific_gates_passed'] is False and diagnostic['scientific_freeze_eligible'] is False
    # Generate a real CPU report through the public CLI and roundtrip its gate schema.
    import eksperimen_model.profile_m4human_encoder as profiler
    from eksperimen_model.utils.m4human_gates import _verify_readiness,encoder_readiness_config_hash
    from eksperimen_model.utils.m4human_runtime import atomic_json,file_sha256
    class ProfileDataset(list):
        sensor_dataset=SimpleNamespace(reader=SimpleNamespace(close=lambda:None))
    with tempfile.TemporaryDirectory() as temporary:
        folder=Path(temporary); config_path=folder/'config.yaml'; report_path=folder/'profile.json'
        config={'data_kind':'synthetic','normalizer':{'mean':[0.]*4,'std':[1.]*4},
                'training':{'micro_batch':1,'accumulation':1,'precision':'fp32','workers':0,'cpu_threads':1}}
        config_path.write_text(yaml.safe_dump(config))
        arguments=['profile','--config',str(config_path),'--output',str(report_path),'--device','cpu',
                   '--micro-batches','1','--workers','0','--precisions','fp32','--warmup-steps','1','--timed-steps','1','--ram-reserve-mib','1','--synthetic-smoke']
        with patch.object(profiler,'make_datasets',return_value=(ProfileDataset([sample,sample]),ProfileDataset(),{'pelvis_index':0},{})),patch('sys.argv',arguments),contextlib.redirect_stdout(io.StringIO()):
            profiler.main()
        report=json.loads(report_path.read_text())
        chosen=dict(config,training=report['recommended_settings'])
        assert report['data_kind']=='synthetic' and not report['scientific_gates_passed']
        assert report['readiness_config_hash']==encoder_readiness_config_hash(chosen)
        assert report['outcomes'][0]['settings']==report['recommended_settings']
        assert report['outcomes'][0]['numerical_parity']['cuda_measured'] is False
        chosen['resource_profile_passed']={'artifact_path':str(report_path),'artifact_sha256':file_sha256(report_path)}
        try: _verify_readiness(chosen,'resource_profile_passed')
        except ValueError: pass
        else: raise AssertionError('CPU synthetic profiling certified scientific readiness')
        safety = report['outcomes'][0]['resource_safety']
        assert safety['enabled'] is True and safety['cuda_measured'] is False
        assert safety['status'] == 'measured' and safety['sample_count'] > 0
        assert safety['min_ram_available_bytes'] >= safety['ram_reserve_bytes']
        # Broad batch policies are explicit and selected YAML preserves measured settings.
        broad_path, selected_path = folder/'broad_profile.json', folder/'selected.yaml'
        broad_config=dict(config,performance_policy={'status':'unmeasured_candidate','original_effective_batch':1,'effective_batch':1})
        config_path.write_text(yaml.safe_dump(broad_config))
        broad_arguments = ['profile','--config',str(config_path),'--output',str(broad_path),
            '--selected-config',str(selected_path),'--device','cpu','--effective-batches','1','2',
            '--micro-batches','1','2','--workers','0','--precisions','fp32',
            '--warmup-steps','1','--timed-steps','1','--ram-reserve-mib','1','--synthetic-smoke']
        with patch.object(profiler,'make_datasets',return_value=(ProfileDataset([sample]*4),ProfileDataset(),{'pelvis_index':0},{})),patch('sys.argv',broad_arguments),contextlib.redirect_stdout(io.StringIO()):
            profiler.main()
        broad = json.loads(broad_path.read_text())
        selected = yaml.safe_load(selected_path.read_text())
        from eksperimen_model.utils.m4human_runtime import canonical_hash
        assert broad['effective_batch_candidates'] == [1,2]
        assert {row['settings']['effective_batch'] for row in broad['outcomes']} == {1,2}
        assert broad['excluded_micro_batches'] == {'1':[2],'2':[]}
        assert selected['training'] == broad['recommended_settings']
        assert selected['training']['effective_batch'] == broad['effective_batch']
        assert selected['training']['micro_batch']*selected['training']['accumulation'] == broad['effective_batch']
        assert canonical_hash(selected) == broad['recommended_config_hash']
        assert encoder_readiness_config_hash(selected) == broad['readiness_config_hash']
        assert selected['data_kind'] == 'synthetic' and not broad['scientific_gates_passed']
        assert selected['performance_policy']['status']=='resource_profile_selected'
        assert selected['performance_policy']['effective_batch']==selected['training']['effective_batch']
        assert 'real_smoke_passed' not in selected and 'resource_profile_passed' not in selected
        assert all(row['resource_safety']['cuda_measured'] is False for row in broad['outcomes'])
        # Reprofile chosen settings: metadata must not silently invalidate readiness hash.
        reprofile_path=folder/'selected_profile.json'
        arguments=['profile','--config',str(selected_path),'--output',str(reprofile_path),'--device','cpu',
            '--micro-batches',str(selected['training']['micro_batch']),'--workers','0','--precisions','fp32',
            '--warmup-steps','1','--timed-steps','1','--ram-reserve-mib','1','--synthetic-smoke']
        with patch.object(profiler,'make_datasets',return_value=(ProfileDataset([sample]*4),ProfileDataset(),{'pelvis_index':0},{})),patch('sys.argv',arguments),contextlib.redirect_stdout(io.StringIO()):
            profiler.main()
        assert json.loads(reprofile_path.read_text())['readiness_config_hash']==encoder_readiness_config_hash(selected)
        # Simulated CUDA metadata tests schema only; this is no hardware measurement.
        simulated=copy.deepcopy(report); intended=copy.deepcopy(chosen); intended['data_kind']='m4human'
        simulated.update(data_kind='m4human',device='cuda:0',readiness_config_hash=encoder_readiness_config_hash(intended),
                         fixture_label='simulated CUDA fields for schema checks only; actual CPU measurement remains separate')
        simulated['outcomes'][0]['numerical_parity']['cuda_measured']=True
        simulated['outcomes'][0]['resource_safety']={'status':'measured','enabled':True,'cuda_measured':True,
            'sample_count':2,'ram_reserve_bytes':intended['training'].get('ram_reserve_mib',2048)*1024**2,
            'vram_reserve_bytes':intended['training'].get('vram_reserve_mib',1024)*1024**2,
            'min_ram_available_bytes':4096*1024**2,'min_cuda_headroom_bytes':2048*1024**2}
        simulated['outcomes'][0].update(peak_reserved_bytes=100,reserve_bytes=10,available_cuda_budget_bytes=200,
                                       timed_steps=100,successful_updates=100,samples=100)
        atomic_json(report_path,simulated,overwrite=True)
        intended['resource_profile_passed']['artifact_sha256']=file_sha256(report_path)
        assert _verify_readiness(intended,'resource_profile_passed')['status']=='verified'
        intended['lineage'] = {'normalizer_hash':canonical_hash(intended['normalizer'])}
        assert _verify_readiness(intended,'resource_profile_passed')['status']=='verified'
        for digest in (None,'0'*64):
            changed = copy.deepcopy(simulated)
            if digest is None:
                del changed['normalizer_hash']
            else:
                changed['normalizer_hash'] = digest
            atomic_json(report_path,changed,overwrite=True)
            intended['resource_profile_passed']['artifact_sha256'] = file_sha256(report_path)
            try: _verify_readiness(intended,'resource_profile_passed')
            except ValueError: pass
            else: raise AssertionError('wrong/missing profiled normalizer accepted despite exact artifact SHA')
        atomic_json(report_path,simulated,overwrite=True)
        intended['resource_profile_passed']['artifact_sha256'] = file_sha256(report_path)
        changed = copy.deepcopy(intended)
        changed['lineage']['normalizer_hash'] = '0'*64
        try: _verify_readiness(changed,'resource_profile_passed')
        except ValueError: pass
        else: raise AssertionError('config normalizer/lineage mismatch accepted')

    print('M4Human performance contract checks passed')

if __name__=='__main__': main()
