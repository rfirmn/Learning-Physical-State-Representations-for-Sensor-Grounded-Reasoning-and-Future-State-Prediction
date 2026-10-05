"""Fresh-probe/direct-rule evaluation and paired physical error with subject CIs.

The test lock is distinct from the language lock: it binds physical checkpoints,
cache bytes, recipes and the analysis policy before producing held-out outcomes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from eksperimen_model.m4human_training import (
    WindowTensorCache, _PhysicalWindows, _TokenWindows, _models, _scientific_gate,
    _collate, _validate, _target_from_annotation,
)
from eksperimen_model.utils.m4human_kinematics import DEFAULT_RECIPE, build_physical_targets, build_evidence, physical_metrics
from eksperimen_model.utils.m4human_runtime import (
    CONTRACT_VERSION, atomic_json, canonical_hash, file_sha256, read_json, iter_jsonl,
    load_checkpoint, freeze, require_lineage, state_dict_hash, validate_run_artifacts,
)

CONDITIONS = ('M', 'U_base', 'U_kin', 'D_encoder_sg')


def dependency_hashes():
    root = Path(__file__).resolve().parent
    names = ('evaluate_m4human_physical.py', 'm4human_training.py',
             'datasets/m4human_qa_dataset.py', 'utils/m4human_runtime.py',
             'utils/m4human_kinematics.py', 'models/m4human_motion.py',
             'models/m4human_tokenizer.py', 'models/m4human_readout.py')
    return {name:file_sha256(root/name) for name in names}


def _policy(effect, uncertainty):
    if not isinstance(effect, dict) or not effect or any(not isinstance(k,str) or not k or
            type(v) not in (int,float) or not math.isfinite(v) or v < 0 for k,v in effect.items()):
        raise ValueError('Pin finite nonnegative physical metric effect tolerances')
    if not isinstance(uncertainty,dict) or uncertainty.get('method') != 'paired_subject_cluster_row_weighted' or type(uncertainty.get('draws')) is not int or uncertainty['draws'] < 1 or type(uncertainty.get('seed')) is not int:
        raise ValueError('Unsupported uncertainty policy')


def _caches(config, condition, split=None, require_targets=False):
    cache=WindowTensorCache(config['source_cache'],split,config['lineage'],require_targets)
    _scientific_gate(config,cache.metadata)
    if cache.metadata.get('representation') != 'full_H':
        raise ValueError('Physical probes require the exact full-H source cache')
    if canonical_hash({**DEFAULT_RECIPE, **(config.get('recipe') or {})}) != cache.metadata['lineage'].get('recipe_hash'):
        raise ValueError('Physical evidence recipe differs from audited lineage')
    if config.get('joint_map_path') and (file_sha256(config['joint_map_path']) != cache.metadata['lineage'].get('joint_map_hash') or read_json(config['joint_map_path']) != config.get('joint_map')):
        raise ValueError('Physical joint map differs from audited lineage')
    tokens=None
    if condition in ('U_base','U_kin'):
        tokens=WindowTensorCache(config['token_cache'],split,config['lineage'])
        _scientific_gate(config,tokens.metadata)
        if tokens.metadata.get('condition') != {'U_base':'C_base','U_kin':'C_kin'}[condition] or tokens.metadata.get('representation') != 'exact_U' or tokens.metadata.get('K')!=config.get('K') or tokens.metadata['lineage'].get('h_cache_hash') != canonical_hash(cache.metadata):
            raise ValueError('Foreign physical token cache condition or full-H source')
        support=lambda rows:{r['window_id']:(str(r['subject_id']),r['recording_id'],r['split'],r['window_support_hash'],r['source_keys'],r['frame_uids']) for r in rows}
        if support(cache.rows)!=support(tokens.rows):
            raise ValueError('Physical token/source exact-window cohort differs')
    return cache,tokens


def _probe_payload(checkpoint, config, condition, cache, tokens):
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if payload.get('stage')!='probe' or payload.get('condition')!=condition or payload.get('K')!=config.get('K'):
        raise ValueError('Auxiliary/training heads cannot substitute for a fresh probe')
    stats=payload.get('stats',{})
    if stats.get('train_only') is not True or stats.get('hash') != canonical_hash({k:v for k,v in stats.items() if k!='hash'}):
        raise ValueError('Fresh probe normalizer content/hash mismatch')
    physical=read_json(config['normalizer_path'])
    if physical.get('hash') != canonical_hash({k:v for k,v in physical.items() if k!='hash'}) or stats.get('physical_normalizer_hash') != physical.get('hash') or cache.metadata['lineage'].get('normalizer_hash') != physical.get('hash'):
        raise ValueError('Foreign physical probe normalizer')
    expected={**config['lineage'],'condition':condition,'source_cache_hash':canonical_hash(cache.metadata),
              'stage_normalizer_hash':stats['hash'],
              'training_config_hash':canonical_hash({k:v for k,v in config.items() if k not in ('output_dir','parent_run_id')})}
    if tokens is not None:
        expected['token_cache_hash']=canonical_hash(tokens.metadata)
    require_lineage(payload['metadata'],expected)
    if payload.get('state_hash')!=state_dict_hash(payload['state_dict']):
        raise ValueError('Fresh probe checkpoint state hash mismatch')
    return payload


def _qa_rows(qa, cache, config, split):
    from eksperimen_model.datasets.m4human_qa_dataset import read_qa
    rows=read_qa(qa); metadata=cache.metadata; lineage=metadata['lineage']
    # The index and sensor payloads establish provenance; annotation bytes are never needed.
    full=WindowTensorCache(cache.root,expected_lineage=config['lineage'])
    lookup={r['window_id']:i for i,r in enumerate(full.rows)}
    times={}
    for row in rows:
        expected={k:lineage.get(k) for k in ('recipe_hash','joint_map_hash','split_hash','coordinate_hash')}
        expected.update(h_cache_hash=canonical_hash(metadata),gt_source_hash=lineage.get('target_hash'))
        if any(not value or row.get(k)!=value for k,value in expected.items()):
            raise ValueError('QA/physical cache scientific lineage mismatch')
        index=lookup.get(row['window_id'])
        if index is None:
            raise ValueError('QA window absent from physical cache')
        cached=full.rows[index]
        if str(row['subject_id'])!=str(cached['subject_id']) or any(row.get(k)!=cached.get(k) for k in ('recording_id','split','window_support_hash','source_keys')) or row.get('frame_ids')!=cached.get('frame_uids'):
            raise ValueError('QA/physical cache exact-window identity mismatch')
        if row['split']!=split:
            continue
        if index not in times:
            times[index]=full[index]['sensor']['time_s'].tolist()
        if len(times[index])!=32 or not isinstance(row.get('time_s'),list) or len(row['time_s'])!=32 or any(not math.isfinite(float(a)) or not math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-9) for a,b in zip(row['time_s'],times[index])):
            raise ValueError('QA/physical cache exact-window timestamps mismatch')
    return [r for r in rows if r['split']==split and r['target_status']!='undefined']


def artifact_identity(config_path, checkpoint=None, qa=None):
    config = yaml.safe_load(Path(config_path).read_text())
    return {'config_hash':canonical_hash(config),
            'source_cache_hash':file_sha256(Path(config['source_cache'])/'metadata.json'),
            'token_cache_hash':file_sha256(Path(config['token_cache'])/'metadata.json') if config.get('token_cache') else None,
            'checkpoint_sha256':file_sha256(checkpoint) if checkpoint else None,
            'qa_sha256':file_sha256(qa) if qa else None}


def create_lock(spec_path, output):
    spec = read_json(spec_path)
    if not spec.get('entries') or spec.get('held_out_outcomes_accessed') is not False:
        raise ValueError('Pin effect tolerances, uncertainty and physical model entries before test')
    _policy(spec.get('effect_tolerance'),spec.get('uncertainty'))
    entries = {}; matched=[]; exposures=[]; test_support=None
    for condition, entry in spec['entries'].items():
        if condition not in CONDITIONS or condition != 'D_encoder_sg' and not entry.get('checkpoint'):
            raise ValueError('Unknown/missing independent probe checkpoint')
        config=yaml.safe_load(Path(entry['config']).read_text())
        cache,tokens=_caches(config,condition)
        if cache.metadata.get('scientific_eligible') is not True or config.get('data_kind')!='m4human':
            raise ValueError('Physical test lock requires scientifically eligible real-data caches')
        support={r['window_id']:r['window_support_hash'] for r in cache.rows if r['split']=='test'}
        if not support or test_support is not None and test_support!=support:
            raise ValueError('Physical lock requires identical nonempty held-out window support')
        test_support=support
        if condition!='D_encoder_sg':
            checkpoint=Path(entry['checkpoint']); run=checkpoint.parent
            if checkpoint.name!='best.pt' or validate_run_artifacts(run):
                raise ValueError('Physical lock requires a complete validated selected best.pt probe run')
            if canonical_hash(yaml.safe_load((run/'config_snapshot.yaml').read_text())) != canonical_hash(config):
                raise ValueError('Physical lock config differs from selected probe run')
            events=list(iter_jsonl(run/'events.jsonl'))
            if any('test' in path.name.lower() for path in run.glob('*') if path.is_file()) or any(r.get('split')=='test' or r.get('partition')=='test' or 'test' in r.get('event','').lower() or set(r.get('source_ids',[])) & support.keys() for r in events):
                raise ValueError('Physical lock cannot follow held-out test outcomes')
            payload=_probe_payload(checkpoint,config,condition,cache,tokens)
            load_checkpoint(checkpoint,_models('probe',payload['stats'],config),payload['metadata'])
            paired=read_json(config['paired_contract_path'])
            if not paired.get('initialization_hash') or paired.get('selection')!='minimum_physical_objective_earliest' or paired.get('sample_order')!='seeded_epoch_permutation' or paired.get('initialization_hash')!=payload['metadata'].get('initialization_hash') or paired.get('normalizer_hash')!=payload['stats']['hash'] or paired.get('source_cache_hash')!=canonical_hash(cache.metadata):
                raise ValueError('Selected probe differs from matched initialization/cache contract')
            for partition,key in (('train','train_cohort_hash'),('val','validation_cohort_hash')):
                if paired.get(key)!=canonical_hash([r['window_id'] for r in cache.rows if r['split']==partition]):
                    raise ValueError('Matched probe cohort differs')
            budget={k:v for k,v in config.items() if k not in ('output_dir','parent_run_id','token_cache')}
            metrics=read_json(run/'metrics.json')
            if metrics.get('status')!='complete' or metrics.get('analysis_ready') is not True or any(paired.get(k)!=v for k,v in budget.items()) or metrics.get('successful_updates')!=config['successful_updates']:
                raise ValueError('Selected probe differs from matched training budget')
            updates=[{k:r.get(k) for k in ('update','source_ids','exposed_windows')} for r in events if r.get('event')=='update']
            if len(updates)!=config['successful_updates'] or any(not r['source_ids'] or r['exposed_windows']!=len(r['source_ids']) for r in updates):
                raise ValueError('Missing matched successful probe exposure records')
            exposures.append(updates)
            matched.append(paired)
        entries[condition] = artifact_identity(entry['config'],entry.get('checkpoint'),entry.get('qa'))
    if not {'M','U_base','U_kin'} <= entries.keys():
        raise ValueError('Physical retention lock requires all three fresh probe sources')
    if any(item!=matched[0] for item in matched[1:]) or any(item!=exposures[0] for item in exposures[1:]):
        raise ValueError('Physical retention requires matched fresh probe initialization, budget and cohorts')
    payload={'contract_version':CONTRACT_VERSION,'kind':'physical_probe_test','entries':entries,
             'effect_tolerance':spec['effect_tolerance'],'uncertainty':spec['uncertainty'],
             'held_out_outcomes_accessed':False,'test_support_hash':canonical_hash(test_support),
             'dependency_hashes':dependency_hashes()}
    atomic_json(output,{**payload,'lock_hash':canonical_hash(payload)})


def verify_lock(path, condition=None, identity=None):
    lock=read_json(path)
    if lock.get('contract_version')!=CONTRACT_VERSION or lock.get('kind')!='physical_probe_test' or lock.get('dependency_hashes')!=dependency_hashes() or lock.get('held_out_outcomes_accessed') is not False:
        raise ValueError('Foreign physical test contract')
    if lock.get('lock_hash') != canonical_hash({k:v for k,v in lock.items() if k!='lock_hash'}):
        raise ValueError('Physical test contract checksum mismatch')
    _policy(lock.get('effect_tolerance'),lock.get('uncertainty'))
    if not {'M','U_base','U_kin'}<=lock.get('entries',{}).keys() or not lock.get('test_support_hash'):
        raise ValueError('Incomplete physical test protocol')
    if condition is not None and lock['entries'].get(condition)!=identity:
        raise ValueError('Physical test inputs differ from locked protocol')
    return lock


def aggregate_records(records):
    totals={}
    for row in records:
        for name,metric in row.get('metrics',{}).items():
            total=totals.setdefault(name,{'sum':0.,'count':0,'unit':metric['unit']})
            total['sum']+=metric['sum']; total['count']+=metric['count']
    for metric in totals.values():
        metric['mean']=metric['sum']/metric['count'] if metric['count'] else None
    return {'metrics':totals,'n_windows':len(records),'failed_windows':sum(bool(r.get('failure')) for r in records),
            'n_subjects':len({r['subject_id'] for r in records}),
            'n_recordings':len({r['recording_id'] for r in records})}


def compare_physical(base, kin, draws=2000, seed=42):
    """Positive delta means lower error for kin; paired finite cohort explicit."""
    if type(draws) is not int or draws<1 or type(seed) is not int:
        raise ValueError('Positive bootstrap draws required')
    left={r['window_id']:r for r in base}; right={r['window_id']:r for r in kin}
    if len(left)!=len(base) or len(right)!=len(kin) or left.keys()!=right.keys() or not left:
        raise ValueError('Paired physical records require identical unique window IDs')
    pairs=[]
    for uid,a in left.items():
        b=right[uid]
        if str(a.get('subject_id'))!=str(b.get('subject_id')) or any(a.get(k)!=b.get(k) for k in ('recording_id','window_support_hash','split')):
            raise ValueError('Paired support/group mismatch')
        if not a.get('failure') and not b.get('failure'):
            if a['metrics'].keys()!=b['metrics'].keys() or any(a['metrics'][k]['count']!=b['metrics'][k]['count'] or a['metrics'][k]['unit']!=b['metrics'][k]['unit'] for k in a['metrics']):
                raise ValueError('Paired physical denominators differ')
            for row in (a,b):
                for metric in row['metrics'].values():
                    if type(metric['count']) is not int or metric['count']<0 or type(metric['sum']) not in (int,float) or not math.isfinite(metric['sum']) or metric['sum']<0:
                        raise ValueError('Invalid physical metric sum/count')
            pairs.append((a,b))
    subjects=sorted({str(r['subject_id']) for r in base})
    names=sorted({k for a,_ in pairs for k in a['metrics']})
    if any(set(a['metrics'])!=set(names) for a,_ in pairs):
        raise ValueError('Paired physical metric registry differs between windows')
    def delta(name, weights):
        numerator=count=0.
        for a,b in pairs:
            w=weights.get(str(a['subject_id']),0)
            numerator+=w*(a['metrics'][name]['sum']-b['metrics'][name]['sum'])
            count+=w*a['metrics'][name]['count']
        return numerator/count if count else None
    rng=random.Random(seed)
    distributions={name:[] for name in names}; unsupported={name:0 for name in names}
    for _ in range(draws):
        weights=Counter(rng.choices(subjects,k=len(subjects)))
        for name in names:
            value=delta(name,weights)
            if value is None:
                unsupported[name]+=1
            else:
                distributions[name].append(value)
    results={}
    for name in names:
        values=distributions[name]
        results[name]={'delta_base_minus_kin':delta(name,Counter(subjects)),
                       'delta_reference_minus_candidate':delta(name,Counter(subjects)),
                       'unit':pairs[0][0]['metrics'][name]['unit'],
                       'ci95':np.quantile(values,[.025,.975]).tolist() if len(subjects)>=3 and len(values)==draws else None,
                       'unsupported_resamples':unsupported[name],
                       'per_subject_delta':{str(s):delta(name,{s:1}) for s in subjects}}
    return {'status':'available' if names and any(r['delta_base_minus_kin'] is not None for r in results.values()) else 'unavailable',
            'unavailable_reason':None if names and any(r['delta_base_minus_kin'] is not None for r in results.values()) else 'no_paired_finite_metric_support',
            'metrics':results,'cohort_windows':len(base),'paired_finite_windows':len(pairs),
            'base_failures':sum(bool(r.get('failure')) for r in base),'kin_failures':sum(bool(r.get('failure')) for r in kin),
            'n_subjects':len(subjects),'draws':draws,'seed':seed,
            'method':'paired_subject_cluster_row_weighted','conditional_on_fitted_checkpoints':True,
            'limitation':'No CI with fewer than three subjects or undefined resamples; seed variance not measured'}


def predict(config_path, condition, checkpoint, split, output, qa=None, test_contract=None):
    if condition not in CONDITIONS or split not in ('val','test'):
        raise ValueError('Unsupported physical condition or partition')
    config=yaml.safe_load(Path(config_path).read_text())
    identity=artifact_identity(config_path,checkpoint,qa)
    if split=='test':
        if not test_contract:
            raise ValueError('Held-out physical prediction requires a pre-locked test contract')
        verify_lock(test_contract,condition,identity)
    cache,tokens=_caches(config,condition,split,require_targets=True)
    qa_rows=_qa_rows(qa,cache,config,split) if qa else []
    dataset=_PhysicalWindows(cache,config['max_gap_s'],common=True)
    model=None
    if condition!='D_encoder_sg':
        if not checkpoint:
            raise ValueError('A fresh independent probe checkpoint is required')
        payload=_probe_payload(checkpoint,config,condition,cache,tokens)
        if condition!='M':
            dataset=_TokenWindows(dataset,config['token_cache'],config['lineage'],condition)
        model=_models('probe',payload['stats'],config)
        load_checkpoint(checkpoint,model,{**config['lineage'],'condition':condition})
        freeze(model.to(config.get('device','cpu')))
    records=[]
    for index in range(len(dataset)):
        sample=dataset[index]; provenance=sample['provenance']
        base={'window_id':provenance['window_id'],'subject_id':provenance['subject_id'],
              'recording_id':provenance['recording_id'],'condition':condition,'split':split,
              'window_support_hash':provenance['window_support_hash']}
        try:
            if model is not None:
                loader=DataLoader([sample],batch_size=1,collate_fn=_collate)
                _,rows=_validate(model,'probe',condition,loader,torch.device(config.get('device','cpu')),payload['stats'],config)
                record={**rows[0],**base,'failure':False}
            else:
                s=sample['sensor']; t=sample['targets']; times=s['time_s'].numpy()
                frame=s['common_frame_valid'].numpy(); joint=np.broadcast_to(frame[:,None],(32,22)).copy()
                predicted=build_physical_targets(s['P_enc_relative_m'].numpy(),s['r_enc_m'].numpy(),times,joint,frame,max_gap_s=config['max_gap_s'])
                tensor={k:torch.as_tensor(v) for k,v in predicted.items()}
                metrics=physical_metrics(tensor,t,pelvis_index=config['pelvis_index'])
                tasks=[]
                for task,body in [('root_speed_trend',None),('relative_limb_motion','arms'),('relative_limb_motion','legs')]:
                    if body and not config.get('joint_map'):
                        continue
                    tasks.append({'task':task,'body_part':body,'prediction':build_evidence(predicted,times,predicted,
                                  config.get('recipe'),task,body,config.get('joint_map'))})
                record={**base,'metrics':metrics,'tasks':tasks,'failure':False}
        except (ValueError,RuntimeError) as exc:
            record={**base,'metrics':{},'tasks':[],'failure':True,'error':str(exc)}
        records.append(record)
    predictions=[]
    if qa:
        from eksperimen_model.datasets.m4human_qa_dataset import canonical_answer
        by_window={r['window_id']:r for r in records}
        for row in qa_rows:
            record=by_window.get(row['window_id'])
            task=next((t for t in record['tasks'] if t['task']==row['task'] and t['body_part']==row.get('body_part')),None) if record else None
            answer=task['prediction']['diagnostic_answer'] if task else None
            predictions.append({'qa_id':row['qa_id'],'window_id':row['window_id'],
                                'raw_text':canonical_answer(row['task'],answer) if answer else '',
                                'raw_ids':[],'error':None if answer else 'missing_or_failed_physical_prediction'})
    result={'contract_version':CONTRACT_VERSION,'condition':condition,'split':split,'identity':identity,
            'data_kind':cache.metadata['data_kind'],'records':records,'predictions':predictions,
            'aggregate':aggregate_records(records),'test_lock_hash':read_json(test_contract)['lock_hash'] if test_contract else None}
    atomic_json(output,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    lock=sub.add_parser('lock'); lock.add_argument('--spec',required=True); lock.add_argument('--output',required=True)
    pred=sub.add_parser('predict')
    pred.add_argument('--config',required=True); pred.add_argument('--condition',choices=CONDITIONS,required=True)
    pred.add_argument('--checkpoint'); pred.add_argument('--qa'); pred.add_argument('--test-contract')
    pred.add_argument('--split',choices=('val','test'),default='val'); pred.add_argument('--output',required=True)
    compare=sub.add_parser('compare'); compare.add_argument('--base',required=True); compare.add_argument('--kin',required=True)
    compare.add_argument('--draws',type=int,default=2000); compare.add_argument('--seed',type=int,default=42)
    compare.add_argument('--test-contract'); compare.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='lock':
        create_lock(args.spec,args.output)
    elif args.command=='predict':
        predict(args.config,args.condition,args.checkpoint,args.split,args.output,args.qa,args.test_contract)
    else:
        base,kin=read_json(args.base),read_json(args.kin)
        if (base.get('condition'),kin.get('condition')) not in {('U_base','U_kin'),('M','U_base'),('M','U_kin')} or base['split']!=kin['split'] or base['split'] not in ('val','test'):
            raise ValueError('Comparison partition mismatch')
        if any(row.get('condition')!=result['condition'] or row.get('split')!=result['split'] for result in (base,kin) for row in result['records']):
            raise ValueError('Comparison records differ from declared condition/partition')
        draws,seed=args.draws,args.seed
        if base['split']=='test':
            if not args.test_contract:
                raise ValueError('Held-out comparison requires physical test contract')
            lock=verify_lock(args.test_contract)
            for result in (base,kin):
                verify_lock(args.test_contract,result['condition'],result['identity'])
                if result['test_lock_hash']!=lock['lock_hash']:
                    raise ValueError('Predictions use a different test lock')
            draws,seed=lock['uncertainty']['draws'],lock['uncertainty']['seed']
        result=compare_physical(base['records'],kin['records'],draws,seed)
        result.update(reference_condition=base['condition'],candidate_condition=kin['condition'],
                      delta_convention='reference_error_minus_candidate_error; positive means lower candidate error',
                      input_sha256={'reference':file_sha256(args.base),'candidate':file_sha256(args.kin)},
                      test_lock_hash=lock['lock_hash'] if base['split']=='test' else None,
                      effect_tolerance=lock['effect_tolerance'] if base['split']=='test' else None)
        atomic_json(args.output,result)


if __name__=='__main__':
    main()
