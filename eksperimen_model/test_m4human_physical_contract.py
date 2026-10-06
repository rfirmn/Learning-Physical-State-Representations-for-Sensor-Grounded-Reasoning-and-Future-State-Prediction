"""Runnable contract checks on temporary fabricated artifacts; no scientific claims."""
import copy
import json
from pathlib import Path
import random
import tempfile
from unittest.mock import patch

import numpy as np
import torch
import yaml

from eksperimen_model.evaluate_m4human_physical import (
    _caches, _probe_payload, _qa_rows, compare_physical, create_lock, verify_lock, main as evaluate_cli,
)
from eksperimen_model.m4human_training import _models
from eksperimen_model.utils.m4human_kinematics import DEFAULT_RECIPE
from eksperimen_model.utils.m4human_runtime import (
    CONTRACT_VERSION, atomic_json, canonical_hash, file_sha256, read_json,
    save_checkpoint, state_dict_hash, validate_run_artifacts,
)


def reject(action):
    try:
        action()
    except ValueError:
        return
    raise AssertionError('foreign/incomplete contract accepted')


def fixture(root):
    lineage={key:canonical_hash(key) for key in ('schema_hash','coordinate_hash','joint_map_hash','split_hash','encoder_hash','target_hash')}
    lineage['recipe_hash']=canonical_hash(DEFAULT_RECIPE)
    normalizer={'train_only':True,'s_v_joint':1.,'s_v_root':1.}
    normalizer['hash']=canonical_hash(normalizer)
    atomic_json(root/'normalizers.json',normalizer)
    lineage['normalizer_hash']=normalizer['hash']
    rows=[]; H=root/'H'; H.mkdir()
    for subject,split in enumerate(('train','val','test'),1):
        path=H/f'{subject}.pt'
        torch.save({'time_s':torch.arange(32,dtype=torch.float64)/12},path)
        row={'window_id':f'w{subject}','subject_id':subject,'recording_id':f'r{subject}','split':split,
             'tensor_path':path.name,'tensor_sha256':file_sha256(path),'source_keys':[f's{subject}:{k}' for k in range(32)],
             'frame_uids':[f'f{subject}:{k}' for k in range(32)],'window_support_hash':canonical_hash(subject),
             'target_path':'DO_NOT_READ.pt','target_sha256':'not_a_target_fixture'}
        rows.append(row)
    def cache(directory,metadata,entries):
        directory.mkdir(exist_ok=True)
        (directory/'index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in entries))
        atomic_json(directory/'metadata.json',{'contract_version':CONTRACT_VERSION,'complete':True,
                    'data_kind':'m4human','scientific_eligible':True,'count':len(entries),
                    'index_sha256':file_sha256(directory/'index.jsonl'),**metadata})
    cache(H,{'lineage':lineage,'representation':'full_H'},rows)
    hmeta=read_json(H/'metadata.json')
    for condition in ('C_base','C_kin'):
        cache(root/condition,{'lineage':{**lineage,'h_cache_hash':canonical_hash(hmeta)},
              'representation':'exact_U','condition':condition,'K':16},rows)
    config={'contract_version':CONTRACT_VERSION,'source_cache':str(H),'data_kind':'m4human',
            'scientific_gates_passed':True,'precision':'fp32','lineage':lineage,'pelvis_index':0,
            'normalizer_path':str(root/'normalizers.json'),'successful_updates':1,'K':16,
            'output_dir':str(root/'runs'),'paired_contract_path':str(root/'paired.json')}
    stats={**normalizer,'physical_normalizer_hash':normalizer['hash']}
    stats['hash']=canonical_hash({k:v for k,v in stats.items() if k!='hash'})
    model=_models('probe',stats,config)
    initialization=state_dict_hash(model)
    paired={**{k:v for k,v in config.items() if k not in ('output_dir','parent_run_id','token_cache')},
            'initialization_hash':initialization,'normalizer_hash':stats['hash'],
            'sample_order':'seeded_epoch_permutation','selection':'minimum_physical_objective_earliest',
            'source_cache_hash':canonical_hash(hmeta),'train_cohort_hash':canonical_hash(['w1']),
            'validation_cohort_hash':canonical_hash(['w2'])}
    atomic_json(root/'paired.json',paired)
    entries={}
    for condition in ('M','U_base','U_kin'):
        current=copy.deepcopy(config)
        if condition!='M':
            current['token_cache']=str(root/('C_base' if condition=='U_base' else 'C_kin'))
        path=root/f'{condition}.yaml'; path.write_text(yaml.safe_dump(current))
        run=root/'runs'/condition; run.mkdir(parents=True)
        (run/'config_snapshot.yaml').write_text(yaml.safe_dump(current))
        metadata={**lineage,'condition':condition,'source_cache_hash':canonical_hash(hmeta),
                  'stage_normalizer_hash':stats['hash'],'initialization_hash':initialization,
                  'training_config_hash':canonical_hash({k:v for k,v in current.items() if k not in ('output_dir','parent_run_id')})}
        if condition!='M':
            metadata['token_cache_hash']=canonical_hash(read_json(Path(current['token_cache'])/'metadata.json'))
        for name in ('best.pt','last.pt'):
            save_checkpoint(run/name,model,metadata,stats=stats,stage='probe',condition=condition,K=16)
        atomic_json(run/'lineage.json',{'contract_version':CONTRACT_VERSION,'config_hash':canonical_hash(current)})
        (run/'history.jsonl').write_text('{}\n'); (run/'events.jsonl').write_text(json.dumps({'event':'update','update':1,'source_ids':['w1'],'exposed_windows':1})+'\n')
        record={**rows[1],'metrics':{'p_root':{'sum':1.,'count':1,'mean':1.,'unit':'m'}}}
        (run/'predictions_val.jsonl').write_text(json.dumps(record)+'\n')
        atomic_json(run/'diagnostic_samples.json',{'checkpoint_hash':file_sha256(run/'best.pt'),'records':[{'window_id':'w2'}]})
        np.savez(run/'diagnostic_samples.npz',value=np.ones(1))
        atomic_json(run/'metric_recomputation.json',{'status':'verified','kind':'physical',
                    'predictions_sha256':file_sha256(run/'predictions_val.jsonl'),'metrics':record['metrics']})
        atomic_json(run/'metrics.json',{'status':'complete','analysis_ready':True,'successful_updates':1,'validation_physical':record['metrics']})
        atomic_json(run/'completion.json',{'status':'complete','metrics_hash':file_sha256(run/'metrics.json'),'history_hash':file_sha256(run/'history.jsonl')})
        assert not validate_run_artifacts(run), validate_run_artifacts(run)
        entries[condition]={'config':str(path),'checkpoint':str(run/'best.pt')}
    return config,entries,rows


def main():
    torch.set_num_threads(1)
    base=[];kin=[]
    for subject,(total,count) in enumerate(((1.,1),(4.,2),(27.,9)),1):
        row={'window_id':str(subject),'subject_id':subject,'recording_id':str(subject),
             'window_support_hash':str(subject),'split':'val','failure':False,
             'metrics':{'error':{'sum':total,'count':count,'unit':'m'}}}
        base.append(row); kin.append({**row,'metrics':{'error':{**row['metrics']['error'],'sum':0.}}})
    comparison=compare_physical(base,kin,draws=100,seed=7)
    metric=comparison['metrics']['error']; assert metric['delta_base_minus_kin']==32/12
    rng=random.Random(7); totals=[1.,4.,27.]; counts=[1,2,9]
    expected=[]
    for _ in range(100):
        sampled=rng.choices(range(3),k=3)
        expected.append(sum(totals[i] for i in sampled)/sum(counts[i] for i in sampled))
    assert metric['ci95']==np.quantile(expected,[.025,.975]).tolist()
    failed=[{**r,'failure':True,'metrics':{}} for r in base]
    assert compare_physical(failed,kin,draws=10)['status']=='unavailable'
    reject(lambda:compare_physical(base,kin,draws=True))
    with tempfile.TemporaryDirectory(prefix='m4physical_contract_', ignore_cleanup_errors=True) as temporary:
        root=Path(temporary); config,entries,rows=fixture(root)
        for condition,records in (('M',base),('U_base',base),('U_kin',kin)):
            atomic_json(root/f'{condition}.json',{'condition':condition,'split':'val',
                        'records':[{**r,'condition':condition} for r in records]})
        def cli(reference,candidate):
            with patch('sys.argv',['physical','compare','--base',str(root/f'{reference}.json'),
                      '--kin',str(root/f'{candidate}.json'),'--draws','10','--output',str(root/'comparison.json')]):
                evaluate_cli()
        for reference,candidate in (('M','U_base'),('M','U_kin'),('U_base','U_kin')):
            cli(reference,candidate)
            compared=read_json(root/'comparison.json')
            assert (compared['reference_condition'],compared['candidate_condition'])==(reference,candidate)
            assert 'delta_reference_minus_candidate' in compared['metrics']['error']
            (root/'comparison.json').unlink()
        for reference,candidate in (('U_kin','U_base'),('M','M'),('U_base','M')):
            reject(lambda:cli(reference,candidate))
        source,tokens=_caches(config,'M','val')
        row=rows[1]
        qa={**{k:row[k] for k in ('window_id','recording_id','split','source_keys','window_support_hash')},
            'subject_id':str(row['subject_id']),'frame_ids':row['frame_uids'],'time_s':(np.arange(32)/12).tolist(),
            **{k:config['lineage'][k] for k in ('recipe_hash','joint_map_hash','split_hash','coordinate_hash')},
            'h_cache_hash':canonical_hash(source.metadata),'gt_source_hash':config['lineage']['target_hash'],
            'qa_id':'q','task':'root_speed_trend','question':'speed?','target_status':'defined','validity':'valid','answer':'speeding_up'}
        qa_path=root/'qa.jsonl'
        def write_qa(value):
            qa_path.write_text(json.dumps(value)+'\n')
        write_qa(qa); assert len(_qa_rows(qa_path,source,config,'val'))==1
        for field in ('recipe_hash','joint_map_hash','split_hash','coordinate_hash','h_cache_hash','gt_source_hash','subject_id','window_id','recording_id','window_support_hash','source_keys','frame_ids','time_s','split'):
            changed=copy.deepcopy(qa)
            changed[field]=(['foreign']*32 if field in ('source_keys','frame_ids') else [float(i) for i in range(32)] if field=='time_s' else 'test' if field=='split' else 'foreign')
            write_qa(changed); reject(lambda:_qa_rows(qa_path,source,config,'val'))
        checkpoint=entries['M']['checkpoint']; payload=torch.load(checkpoint,weights_only=True)
        for field in ('source_cache_hash','stage_normalizer_hash'):
            damaged=copy.deepcopy(payload); damaged['metadata'][field]='foreign'
            torch.save(damaged,root/'tampered.pt')
            reject(lambda:_probe_payload(root/'tampered.pt',config,'M',source,tokens))
        uconfig=yaml.safe_load(Path(entries['U_base']['config']).read_text()); h,u=_caches(uconfig,'U_base','val')
        damaged=torch.load(entries['U_base']['checkpoint'],weights_only=True); damaged['metadata']['token_cache_hash']='foreign'
        torch.save(damaged,root/'tampered.pt'); reject(lambda:_probe_payload(root/'tampered.pt',uconfig,'U_base',h,u))
        spec={'entries':entries,'effect_tolerance':{'p_root':.01},'held_out_outcomes_accessed':False,
              'uncertainty':{'method':'paired_subject_cluster_row_weighted','draws':100,'seed':7}}
        spec_path=root/'spec.json'; atomic_json(spec_path,spec)
        original=torch.load
        def checkpoint_only(path,*args,**kwargs):
            assert Path(path).name in ('best.pt','last.pt'),'lock read a held-out sensor/target payload'
            return original(path,*args,**kwargs)
        with patch('torch.load',side_effect=checkpoint_only):
            create_lock(spec_path,root/'lock.json')
        verify_lock(root/'lock.json')
        for field,value in (('held_out_outcomes_accessed',True),('effect_tolerance',{'p_root':-1}),('uncertainty',{'method':'paired_subject_cluster_row_weighted','draws':True,'seed':7})):
            changed={**spec,field:value}; atomic_json(spec_path,changed,overwrite=True)
            reject(lambda:create_lock(spec_path,root/'rejected.json'))
        atomic_json(spec_path,spec,overwrite=True)
        run=Path(entries['M']['checkpoint']).parent
        (run/'predictions_test.json').write_text('{}')
        reject(lambda:create_lock(spec_path,root/'rejected.json'))
        (run/'predictions_test.json').unlink()
        (run/'completion.json').unlink(); reject(lambda:create_lock(spec_path,root/'rejected.json'))
        changed=read_json(root/'lock.json'); changed['dependency_hashes']['utils/m4human_kinematics.py']='changed'
        changed['lock_hash']=canonical_hash({k:v for k,v in changed.items() if k!='lock_hash'})
        atomic_json(root/'tampered_lock.json',changed); reject(lambda:verify_lock(root/'tampered_lock.json'))
    print('PASS physical QA/probe provenance, pre-outcome lock, paired row-weighted subject bootstrap')


if __name__=='__main__':
    main()
