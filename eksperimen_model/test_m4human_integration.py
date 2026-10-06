"""Exercise real stage M/C/probe loops on explicitly synthetic derived artifacts.

No network/model/dataset downloads. Outputs live in a temporary directory.
"""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch
import yaml

from eksperimen_model.m4human_training import train, extract_motion, extract_tokens, WindowTensorCache
from eksperimen_model.evaluate_m4human_physical import predict, compare_physical
from eksperimen_model.datasets.generate_m4human_qa import windows_from_h_cache, generate_records
from eksperimen_model.datasets.m4human_qa_dataset import M4HumanQADataset
from eksperimen_model.utils.m4human_kinematics import DEFAULT_RECIPE
from eksperimen_model.utils.m4human_runtime import (
    CONTRACT_VERSION, RunLogger, atomic_json, canonical_hash, file_sha256, read_json, validate_run_artifacts,
)


def fixture(root):
    rng = np.random.default_rng(42)
    state, target = root/'states', root/'targets'
    state.mkdir(); target.mkdir()
    rows = []
    for subject, split, count in ((1,'train',35),(2,'val',33)):
        for k in range(count):
            frame = k + 3
            rows.append({'frame_uid':f'{subject}:{frame}', 'subject_id':subject, 'action_id':1,
                         'recording_id':f'{subject}:take', 'segment_id':f'{subject}:segment',
                         'split':split, 'source_key':str([subject,1,frame]), 'source_frame_id':frame,
                         'time_s':frame/12, 'time_source':'nominal_frame_index', 'sensor_frame_valid':True,
                         'context_source_frames':[f'{subject}:{j}' for j in range(frame-3,frame+1)]})
    n = len(rows)
    t = np.array([r['time_s'] for r in rows], np.float64)
    p = rng.normal(0,.2,(n,22,3)).astype(np.float32)
    p[:,0] = 0
    r = np.stack((.03*t*t, np.ones(n), .2*t), -1).astype(np.float32)
    labels = {'joint_relative_m':p, 'root_m':r, 'joint_global_m':p+r[:,None],
              'annotation_joint_valid':np.ones((n,22),bool), 'annotation_root_valid':np.ones(n,bool)}
    fields = {'P_enc':p+rng.normal(0,.01,p.shape).astype(np.float32),
              'r_enc':r+rng.normal(0,.01,r.shape).astype(np.float32),
              'f_enc':rng.normal(size=(n,256)).astype(np.float32), 'time_s':t,
              'sensor_valid':np.ones(n,bool)}
    fields['P_enc'][:,0]=0
    text = ''.join(json.dumps(row)+'\n' for row in rows)
    lineage = {k:canonical_hash(k) for k in ('schema_hash','coordinate_hash','joint_map_hash','encoder_hash')}
    joint_map={'audited':True,'task_groups':{'arms':{'left':[18,20],'right':[19,21]},
                                           'legs':{'left':[4,7],'right':[5,8]}}}
    atomic_json(root/'joint_map.json',joint_map)
    lineage['joint_map_hash']=file_sha256(root/'joint_map.json')
    lineage['recipe_hash']=canonical_hash(DEFAULT_RECIPE)
    for directory, arrays in ((target,labels),(state,fields)):
        (directory/'manifest.jsonl').write_text(text)
        for name,array in arrays.items():
            np.save(directory/(name+'.npy'),array,allow_pickle=False)
        lineage['frame_manifest_hash']=lineage['split_hash']=file_sha256(directory/'manifest.jsonl')
        if directory==state:
            lineage['target_hash']=file_sha256(target/'metadata.json')
        atomic_json(directory/'metadata.json',{'contract_version':CONTRACT_VERSION, 'data_kind':'synthetic',
                    'scientific_eligible':False, 'complete':True, 'count':n, 'max_gap_s':.12,
                    'arrays':{name:file_sha256(directory/(name+'.npy')) for name in arrays},
                    'manifest_hash':file_sha256(directory/'manifest.jsonl'), 'lineage':dict(lineage)})
    config = {'contract_version':CONTRACT_VERSION, 'data_kind':'synthetic', 'source_cache':str(state),
              'target_cache':str(target), 'scientific_gates_passed':False, 'device':'cpu', 'precision':'fp32',
              'seed':42, 'pelvis_index':0, 'max_gap_s':.12, 'micro_batch':1, 'accumulation':4,
              'successful_updates':1, 'learning_rate':.0003, 'train_stride':1, 'val_stride':1,
              'extraction_stride':1, 'diagnostic_windows':1, 'K':16, 'lambda_aux':1.,
              'lineage':lineage, 'output_dir':str(root/'M_run'), 'query_chunk':128,
              'joint_map':joint_map,'joint_map_path':str(root/'joint_map.json')}
    return config



def extract_with_batch_parity(extractor, config_path, checkpoint, output):
    """Same frozen checkpoint: exact masks/IDs and fp32 cache parity within3e-6."""
    config = yaml.safe_load(Path(config_path).read_text())
    output = Path(output)
    batched_output = output.with_name(output.name + '_batch4')
    for size, destination in ((1, output), (4, batched_output)):
        extraction_config = output.with_name(output.name + f'_batch{size}.yaml')
        extraction_config.write_text(yaml.safe_dump(dict(config, eval_batch_size=size)))
        extractor(extraction_config, checkpoint, destination)
    single, batched = WindowTensorCache(output), WindowTensorCache(batched_output)
    assert len(single) == len(batched)
    assert single.metadata['lineage'] == batched.metadata['lineage']
    for i in range(len(single)):
        left, right = single[i], batched[i]
        assert left['provenance']['window_id'] == right['provenance']['window_id']
        assert left['sensor'].keys() == right['sensor'].keys()
        for key, value in left['sensor'].items():
            actual = right['sensor'][key]
            assert value.shape == actual.shape and value.dtype == actual.dtype, key
            if value.is_floating_point():
                assert torch.allclose(value, actual, atol=3e-6, rtol=3e-6), key
            else:
                assert torch.equal(value, actual), key


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='m4human_integration_', ignore_cleanup_errors=True) as temporary:
        root=Path(temporary)
        config=fixture(root)
        config_path=root/'config.yaml'
        def save_config():
            config_path.write_text(yaml.safe_dump(config))
        def check(run):
            issues=validate_run_artifacts(run)
            assert not issues,(run,issues)
            assert read_json(run/'metrics.json')['data_kind']=='synthetic'
        save_config()
        train(config_path,'M')
        m_run=root/'M_run'/'M'
        check(m_run)
        original_metrics=(m_run/'metrics.json').read_bytes()
        original_completion=(m_run/'completion.json').read_bytes()
        dishonest=read_json(m_run/'metrics.json')
        dishonest['validation_physical']['root_error_m']['mean']+=1
        atomic_json(m_run/'metrics.json',dishonest,overwrite=True)
        marker=read_json(m_run/'completion.json')
        marker['metrics_hash']=file_sha256(m_run/'metrics.json')
        atomic_json(m_run/'completion.json',marker,overwrite=True)
        assert 'reported_mean_mismatch:root_error_m' in validate_run_artifacts(m_run)
        (m_run/'metrics.json').write_bytes(original_metrics)
        (m_run/'completion.json').write_bytes(original_completion)
        extract_with_batch_parity(extract_motion,config_path,m_run/'best.pt',root/'H')
        hmeta=read_json(root/'H'/'metadata.json')
        assert hmeta['representation']=='full_H' and not hmeta['scientific_eligible']
        qa_rows,_=generate_records(windows_from_h_cache(root/'H',hmeta['lineage']['joint_map_hash'],hmeta['lineage']['recipe_hash']),config['joint_map'],DEFAULT_RECIPE,['root_speed_trend','relative_limb_motion'])
        qa=root/'qa.jsonl'
        qa.write_text(''.join(json.dumps(row)+'\n' for row in qa_rows))
        config.update(source_cache=str(root/'H'), output_dir=str(root/'C_runs'),
                      normalizer_path=str(m_run/'normalizers.json'),
                      paired_contract_path=str(root/'C_matched.json'),lineage=hmeta['lineage'])
        save_config()
        for condition in ('C_base','C_kin'):
            train(config_path,'C',condition)
            check(root/'C_runs'/condition)
            extract_with_batch_parity(extract_tokens,config_path,root/'C_runs'/condition/'best.pt',root/condition)
            qa_dataset=M4HumanQADataset(qa,root/condition,'val',condition,config['lineage'])
            assert len(qa_dataset)==6 and qa_dataset[0]['U'].shape==(16,256)
        base=read_json(root/'C_runs'/'C_base'/'metrics.json')
        kin=read_json(root/'C_runs'/'C_kin'/'metrics.json')
        assert base['initialization_hash']==kin['initialization_hash']
        assert base['successful_updates']==kin['successful_updates']==1
        # Inference loading must have no annotation dependency whatsoever.
        hrows=list(WindowTensorCache(root/'H').rows)
        for row in hrows:
            (root/'H'/row['target_path']).rename(root/'H'/(row['target_path']+'.hidden'))
        assert WindowTensorCache(root/'H')[0]['targets']=={}
        for row in hrows:
            (root/'H'/(row['target_path']+'.hidden')).rename(root/'H'/row['target_path'])
        config.update(output_dir=str(root/'probe_runs'),paired_contract_path=str(root/'probe_matched.json'))
        evaluations={}
        for condition in ('M','U_base','U_kin'):
            if condition!='M':
                config['token_cache']=str(root/('C_base' if condition=='U_base' else 'C_kin'))
            save_config()
            train(config_path,'probe',condition)
            check(root/'probe_runs'/condition)
            evaluations[condition]=predict(config_path,condition,root/'probe_runs'/condition/'best.pt','val',root/(condition+'_physical.json'),qa=qa)
            assert evaluations[condition]['aggregate']['failed_windows']==0
            assert evaluations[condition]['aggregate']['n_windows']==2
            assert len(evaluations[condition]['predictions'])==6
        direct=predict(config_path,'D_encoder_sg',None,'val',root/'direct_physical.json')
        assert direct['aggregate']['failed_windows']==0
        comparison=compare_physical(evaluations['U_base']['records'],evaluations['U_kin']['records'],draws=20)
        assert comparison['paired_finite_windows']==2
        assert all(v['ci95'] is None for v in comparison['metrics'].values())
        try:
            predict(config_path,'D_encoder_sg',None,'test',root/'unlocked_test.json')
        except ValueError as error:
            assert 'pre-locked' in str(error)
        else:
            raise AssertionError('unlocked test accepted')
        # Resume the identical schedule from the last completed epoch after an
        # injected interruption, then compare actual final parameters bitwise.
        config.update(output_dir=str(root/'resume_reference'), successful_updates=2,
                      paired_contract_path=str(root/'resume_matched.json'))
        save_config()
        train(config_path,'C','C_base')
        config['output_dir']=str(root/'interrupted')
        save_config()
        original_log=RunLogger.log
        def interrupt(logger,event,**values):
            if event=='update' and values.get('epoch')==2:
                raise RuntimeError('injected interruption after saved epoch one')
            return original_log(logger,event,**values)
        with patch.object(RunLogger,'log',interrupt):
            try:
                train(config_path,'C','C_base')
            except RuntimeError as error:
                assert 'injected interruption' in str(error)
            else:
                raise AssertionError('injected interruption did not execute')
        config['output_dir']=str(root/'resumed')
        save_config()
        train(config_path,'C','C_base',resume=root/'interrupted'/'C_base'/'last.pt')
        check(root/'resumed'/'C_base')
        ref=torch.load(root/'resume_reference'/'C_base'/'last.pt',weights_only=True)
        resumed=torch.load(root/'resumed'/'C_base'/'last.pt',weights_only=True)
        assert ref['state_hash']==resumed['state_hash'], 'resume diverged from identical uninterrupted optimizer schedule'
        # Changed supervision cannot be accepted as the original frozen artifact.
        cache=WindowTensorCache(root/'H','train',require_targets=True)
        damaged=root/'H'/cache.rows[0]['target_path']
        with damaged.open('ab') as stream:
            stream.write(b'changed')
        try:
            cache[0]
        except ValueError:
            pass
        else:
            raise AssertionError('changed labels accepted')
    print('PASS synthetic stage M -> exact H -> paired C -> exact U -> fresh M/U probes; artifacts recomputed; extraction batch1/batch4 fp32 tolerance3e-6 passed; real-data gates not_run')


if __name__=='__main__':
    main()
