"""Synthetic contract checks, not evidence of real M4Human labels/performance."""
import hashlib
import json
import multiprocessing as mp
import pickle
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import struct
import tempfile
import numpy as np
import torch
from torch.utils.data import Dataset
from eksperimen_model.datasets.m4human_dataset import (RPCOnlyReader, M4HumanCausalSensorDataset, decode_rpc, decode_msgpack,
    parse_source_key,sample_rpc,clean_rpc,validate_frames,OfflineTargets,EncoderTrainingDataset)
from eksperimen_model.datasets.m4human_state_dataset import EncodedStateWindowDataset
from eksperimen_model.models.m4human_encoder import M4HumanSetEncoderV2,encoder_loss
from eksperimen_model.export_m4human_targets import target_from_joints,transform_to_radar,rotation
from eksperimen_model.train_m4human_encoder import evaluate_encoder,collate_samples,recompute_encoder_metrics,fit_input_normalizer
from eksperimen_model.utils.m4human_runtime import CONTRACT_VERSION,atomic_json,load_checkpoint,save_checkpoint,state_dict_hash,file_sha256
from eksperimen_model.audit_m4human import audit, inspect, draft_manifest
from eksperimen_model.export_m4human_targets import export_targets
from eksperimen_model.train_m4human_encoder import train
from eksperimen_model.extract_m4human_states import extract


def raises(call,kind=ValueError):
    try:
        call()
    except kind:
        return
    raise AssertionError('expected rejection')


def pack_rpc(arr,order='<'):
    return struct.pack(order+'III',2,len(arr),4)+arr.astype(order+'f4').tobytes()


def child_read(reader,queue):
    queue.put(reader.read('[1, 1, 0]').sum().item())
    reader.close()


def fixture(root):
    import lmdb
    schema = {'audited':True,'immutable_snapshot':True,'channels':['x','y','z','intensity'],'xyz_unit':'meter',
              'byte_order':'little','max_points':10000,'max_value_bytes':1024*1024,'sentinel_policy':'none','max_gap_s':.2}
    env = lmdb.open(str(root/'radar_pc.lmdb'),subdir=False,map_size=8*1024*1024)
    rows = []
    with env.begin(write=True) as tx:
        for subject,split in ((1,'train'),(2,'val'),(3,'test')):
            for i in range(40):
                cloud = np.array([[i/100.,1,2,.5],[i/100.+.1,1.1,2.1,1]],np.float32)
                key = str([subject,1,i])
                tx.put(key.encode(),pack_rpc(cloud))
                rows.append(dict(frame_uid=f'{subject}:takeA:{i}',source_key=key,subject_id=subject,action_id=1,
                                 recording_id=f'{subject}:takeA',segment_id=f'{subject}:segment0',source_frame_id=i,
                                 time_s=i/12.,time_source='nominal_frame_index',split=split,sensor_frame_valid=True))
    env.close()
    manifest = root/'frames.jsonl'
    manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return schema,rows,manifest


class FixtureTrainingDataset(Dataset):
    def __init__(self,sensor):
        self.sensor = sensor
    def __len__(self):
        return len(self.sensor)
    def __getitem__(self,i):
        sample = self.sensor[i]
        global_m = np.zeros((22,3),np.float32)
        global_m[:,0] = np.arange(22)*.01
        global_m += np.array([sample['provenance']['source_frame_id']/100.,1,2],np.float32)
        sample['targets'] = {k:torch.from_numpy(np.asarray(v)) for k,v in target_from_joints(global_m,0).items()}
        return sample


def worker_equivalence(dataset):
    from torch.utils.data import DataLoader
    expected = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_samples)))
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_samples, num_workers=2,
                        multiprocessing_context='spawn', persistent_workers=True)
    actual = next(iter(loader))
    for section in ('sensor', 'targets'):
        for key, value in expected.get(section, {}).items():
            assert torch.equal(value, actual[section][key]), (section, key)
    return loader


def synthetic_pipeline(root):
    import lmdb
    import msgpack
    import yaml
    source = root/'source'
    source.mkdir()
    schema,rows,manifest = fixture(source)
    # More than512 points makes epoch changes observable in worker tests.
    env = lmdb.open(str(source/'radar_pc.lmdb'),subdir=False,map_size=8*1024*1024)
    with env.begin(write=True) as tx:
        for row in rows[:40]:
            tx.put(row['source_key'].encode(),pack_rpc(np.arange(2800,dtype=np.float32).reshape(700,4)))
    env.close()
    schema['data_kind'] = 'synthetic'
    schema['source_provenance'] = {'package_id':'synthetic_contract_fixture','upstream_commit':'0'*40,
       'serializer_sha256':'synthetic_fixture','calibration_sha256':'synthetic_fixture','rpc_preprocessing':'synthetic_fixture'}
    for name in ('params','calib','indicator'):
        env = lmdb.open(str(source/(name+'.lmdb')),subdir=False,map_size=8*1024*1024)
        with env.begin(write=True) as tx:
            for row in rows:
                if name=='params':
                    global_m = np.zeros((22,3),np.float32)
                    global_m[:,0] = np.arange(22)*.01 + row['source_frame_id']/100.
                    global_m[:,1] = 1
                    global_m[:,2] = 2
                    value = {'joints':{'__nd__':True,'dtype':'float32','shape':[22,3],'data':global_m.tobytes()}}
                else:
                    value = {'fixture':True}
                tx.put(row['source_key'].encode(),msgpack.packb(value,use_bin_type=True))
        env.close()
    schema_path,joint_path,coord_path,export_path = (root/name for name in ('schema.json','joint_map.json','coordinate.json','target_recipe.json'))
    atomic_json(schema_path,schema)
    atomic_json(joint_path,{'audited':True,'ordered_names':['pelvis']+[f'joint_{i}' for i in range(1,22)],
                            'source_row_indices':list(range(22)),'pelvis_index':0})
    atomic_json(coord_path,{'audited':True,'output_unit':'meter','source_frame':'radar','radar_frame':'synthetic_radar',
                            'evidence':'synthetic fixture only'})
    atomic_json(export_path,{'audited':True,'immutable_snapshot':True,'method':'direct_joints_equivalent',
                             'equivalence_evidence':'synthetic fixture only','direct_joint_unit':'meter','data_kind':'synthetic'})
    inspect_result = inspect(source,root/'inspect',2)
    assert inspect_result['status']=='not_verified' and inspect_result['entry_counts']['radar_pc.lmdb']==120
    policy = root/'draft_policy.json'
    atomic_json(policy,{'package_id':'synthetic_contract_fixture','single_take_per_subject_action_verified':True,
                        'nominal_grid_verified':True,'evidence':'synthetic fixture only','nominal_hz':12,
                        'subject_splits':{'1':'train','2':'val','3':'test'}})
    draft = draft_manifest(source,root/'draft',policy)
    assert draft['status']=='not_verified' and draft['count']==120
    audited = audit(source,root/'audit',schema_path,manifest,joint_path,coord_path)
    assert audited['status']=='verified' and audited['data_kind']=='synthetic'
    assert audited['annotation_semantics_verified'] is False
    phantom = [dict(r) for r in rows]
    phantom[-1].update(source_key='[3, 1, 40]', source_frame_id=40, time_s=40/12.)
    phantom_path = root/'phantom.jsonl'
    phantom_path.write_text(''.join(json.dumps(r)+'\n' for r in phantom))
    raises(lambda:audit(source,root/'audit_phantom',schema_path,phantom_path,joint_path,coord_path))
    for name in ('params', 'calib', 'indicator'):
        env = lmdb.open(str(source/(name+'.lmdb')),subdir=False,map_size=8*1024*1024)
        with env.begin(write=True) as tx:
            key = rows[0]['source_key'].encode()
            value = tx.get(key)
            tx.delete(key)
            tx.put(b'[99, 1, 0]',value)
        env.close()
        raises(lambda:audit(source,root/('audit_missing_'+name),schema_path,manifest,joint_path,coord_path))
        env = lmdb.open(str(source/(name+'.lmdb')),subdir=False,map_size=8*1024*1024)
        with env.begin(write=True) as tx:
            tx.delete(b'[99, 1, 0]')
            tx.put(key,value)
        env.close()

    env = lmdb.open(str(source/'indicator.lmdb'),subdir=False,map_size=8*1024*1024)
    with env.begin(write=True) as tx:
        saved = list(tx.cursor())
        for key, _ in saved:
            tx.delete(key)
    env.close()
    raises(lambda:audit(source,root/'audit_empty',schema_path,manifest,joint_path,coord_path))
    env = lmdb.open(str(source/'indicator.lmdb'),subdir=False,map_size=8*1024*1024)
    with env.begin(write=True) as tx:
        for key, value in saved:
            tx.put(key,value)
    env.close()
    audited_manifest = root/'audit'/'frames.jsonl'
    target_dir = root/'targets'
    target_info = export_targets(source,audited_manifest,target_dir,export_path,joint_path,coord_path)
    assert target_info['data_kind']=='synthetic' and target_info['invalid_target_frames']==0
    sensor_dataset = M4HumanCausalSensorDataset(source,audited_manifest,schema,'train',training=True)
    training_dataset = EncoderTrainingDataset(sensor_dataset,target_dir)
    assert not sensor_dataset._epoch.is_shared()
    loader = worker_equivalence(training_dataset)
    assert sensor_dataset._epoch.is_shared()
    before = next(iter(loader))['sensor']['rpc_m']
    sensor_dataset.set_epoch(1)
    actual = next(iter(loader))['sensor']['rpc_m']
    expected = collate_samples([training_dataset[0],training_dataset[1]])['sensor']['rpc_m']
    assert torch.equal(actual,expected) and not torch.equal(actual,before)
    assert training_dataset.targets.__getstate__()['arrays'] == {}
    del loader
    sensor_dataset.reader.close()

    config = yaml.safe_load(Path('eksperimen_model/configs/m4human_encoder.yaml').read_text())
    config['data_kind']='synthetic'
    config['paths'].update(dataset_root=str(source),schema=str(schema_path),frame_manifest=str(audited_manifest),
                           target_dir=str(target_dir),joint_map=str(joint_path),coordinate_audit=str(coord_path),
                           audit_report=str(root/'audit'/'audit_report.json'))
    config['training'].update(precision='fp32',micro_batch=8,accumulation=1,effective_batch=8,epochs_max=1,diagnostic_samples=2,ram_reserve_mib=1)
    trained = train(config,root/'encoder_run',device='cpu',mode='smoke')
    assert trained['scientific_freeze_eligible'] is False
    state_dir = root/'extracted_states'
    target_dir.rename(root/'targets_offline')
    states = extract(config,root/'encoder_run'/'best.pt',state_dir,'train','cpu',allow_debug=True,batch_size=1)
    batched_state_dir = root/'extracted_states_batch4'
    batched_states = extract(config,root/'encoder_run'/'best.pt',batched_state_dir,'train','cpu',allow_debug=True,batch_size=4)
    # Both cache writes run with offline targets absent; fp32 batch tolerance3e-6.
    assert states['lineage'] == batched_states['lineage']
    assert states['count'] == batched_states['count']
    assert (state_dir/'manifest.jsonl').read_bytes() == (batched_state_dir/'manifest.jsonl').read_bytes()
    for name in ('P_enc','r_enc','f_enc','time_s','sensor_valid'):
        single = np.load(state_dir/(name+'.npy'),allow_pickle=False)
        batched = np.load(batched_state_dir/(name+'.npy'),allow_pickle=False)
        assert single.shape == batched.shape and single.dtype == batched.dtype, name
        if name in ('time_s','sensor_valid'):
            assert np.array_equal(single,batched), name
        else:
            assert np.allclose(single,batched,atol=3e-6,rtol=3e-6), name
    (root/'targets_offline').rename(target_dir)
    assert states['data_kind']=='synthetic' and states['scientific_eligible'] is False
    raises(lambda:EncodedStateWindowDataset(state_dir))
    windows = EncodedStateWindowDataset(state_dir,targets_dir=target_dir,allow_debug=True)
    assert len(windows)==1 and windows[0]['targets']['joint_global_m'].shape==(32,22,3)
    state_loader = worker_equivalence(windows)
    assert windows.__getstate__()['arrays'] == {}
    assert isinstance(pickle.loads(pickle.dumps(windows))[0]['sensor']['f_enc'],torch.Tensor)
    del state_loader
    with (state_dir/'f_enc.npy').open('r+b') as handle:
        handle.seek(-1,2)
        byte=handle.read(1)
        handle.seek(-1,2)
        handle.write(bytes([byte[0]^1]))
    raises(lambda:EncodedStateWindowDataset(state_dir,allow_debug=True))
    return {'audit_frames':audited['frame_count'],'target_frames':target_info['count'],
            'state_anchors':states['count'],'windows':len(windows)}


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    with tempfile.TemporaryDirectory(prefix='m4human_e_contract_') as directory:
        root = Path(directory)
        schema,rows,manifest = fixture(root)
        cloud = np.arange(20,dtype=np.float32).reshape(5,4)
        assert np.array_equal(decode_rpc(pack_rpc(cloud),schema),cloud)
        big = dict(schema,byte_order='big')
        assert np.array_equal(decode_rpc(pack_rpc(cloud,'>'),big),cloud)
        for payload in (b'',pack_rpc(cloud)[:-1],pack_rpc(cloud)+b'0',struct.pack('<III',3,5,4)+cloud.tobytes(),pack_rpc(cloud,'>')):
            raises(lambda payload=payload:decode_rpc(payload,schema))
        import msgpack
        tagged = {'__nd__':True,'dtype':'float32','shape':[2,3],'data':np.arange(6,dtype=np.float32).tobytes()}
        assert decode_msgpack(msgpack.packb(tagged)) .shape==(2,3)
        raises(lambda:decode_msgpack(msgpack.packb(dict(tagged,dtype='object'))))
        raises(lambda:decode_msgpack(msgpack.packb(dict(tagged,shape=[2,4]))))
        raises(lambda:decode_msgpack(msgpack.packb(msgpack.ExtType(1,b'x'))))
        raises(lambda:parse_source_key('__import__("os").system("echo unsafe")'))
        raises(lambda:parse_source_key('[True,1,2]'))
        assert parse_source_key('[1, 2, 3]')==(1,2,3)
        many = np.random.randn(700,4).astype(np.float32)
        a,mask = sample_rpc(many,'uid')
        b,other = sample_rpc(many[::-1],'uid')
        assert np.array_equal(a,b) and mask.sum()==512
        a,mask = sample_rpc(many[:3],'uid')
        assert mask.sum()==3 and not a[~mask].any()
        clean,count = clean_rpc(np.array([[np.nan,0,0,0],[0,0,0,0],[1,2,3,4]],np.float32),dict(schema,sentinel_policy='exact_row',sentinel_row=[0,0,0,0]))
        assert len(clean)==1 and count['nonfinite']==1 and count['sentinel']==1
        leaked = [dict(rows[0]),dict(rows[40],subject_id=1,source_key='[1, 1, 0]')]
        raises(lambda:validate_frames(leaked))
        interleaved = [rows[0],rows[40],rows[1],rows[41]]
        raises(lambda:validate_frames(interleaved))
        raises(lambda:RPCOnlyReader(root/'absent',schema),FileNotFoundError)
        before = hashlib.sha256((root/'radar_pc.lmdb').read_bytes()).hexdigest()
        reader = RPCOnlyReader(root,schema)
        expected = reader.read('[1, 1, 0]').sum().item()
        ctx = mp.get_context('spawn')
        queue = ctx.Queue()
        process = ctx.Process(target=child_read,args=(reader,queue))
        process.start()
        actual = queue.get(timeout=30)
        process.join(30)
        assert process.exitcode==0 and actual==expected
        reader.close()
        assert hashlib.sha256((root/'radar_pc.lmdb').read_bytes()).hexdigest()==before
        ds = M4HumanCausalSensorDataset(root,manifest,schema,'train')
        assert len(ds)==37 and ds[0]['provenance']['context_source_frames']==[r['frame_uid'] for r in rows[:4]]
        normalizer = fit_input_normalizer(ds)
        model = M4HumanSetEncoderV2(normalizer=normalizer)
        sample = ds[0]
        sensor = {k:v[None] for k,v in sample['sensor'].items()}
        pred = model(**sensor)
        assert pred['P_enc_relative_m'].shape==(1,22,3) and pred['f_enc'].shape==(1,256)
        assert pred['P_enc_relative_m'][:,0].eq(0).all()
        permutation = torch.randperm(512)
        swapped = dict(sensor,rpc_m=sensor['rpc_m'][:,:,permutation],point_mask=sensor['point_mask'][:,:,permutation])
        for key in ('P_enc_relative_m','r_enc_m','f_enc'):
            assert torch.allclose(pred[key],model(**swapped)[key],atol=2e-6)
        padded = dict(sensor,rpc_m=sensor['rpc_m'].clone())
        padded['rpc_m'][~sensor['point_mask']] = float('nan')
        assert torch.allclose(pred['f_enc'],model(**padded)['f_enc'],atol=2e-6)
        invalid = dict(sensor,point_mask=torch.zeros_like(sensor['point_mask']))
        output = model(**invalid)
        assert not output['sensor_valid'].any() and output['f_enc'].eq(0).all() and output['r_enc_m'].eq(0).all()
        nonmonotonic = dict(sensor,time_s=sensor['time_s'].flip(-1))
        assert not model(**nonmonotonic)['sensor_valid'].any()
        bad = dict(sensor,rpc_m=sensor['rpc_m'].clone())
        bad['rpc_m'][0,0,0,0] = float('nan')
        raises(lambda:model(**bad))
        fixture_data = FixtureTrainingDataset(ds)
        batch = collate_samples([fixture_data[0],fixture_data[1]])
        optimizer = torch.optim.AdamW(model.parameters(),lr=.0001)
        for term in ('pose','root'):
            optimizer.zero_grad()
            loss = encoder_loss(model(**batch['sensor']),batch['targets'])
            loss[term].backward()
            for component in (model.point_stem,model.temporal):
                assert any(p.grad is not None and p.grad.abs().sum()>0 for p in component.parameters())
        model.zero_grad()
        loss = encoder_loss(model(**batch['sensor']),batch['targets'])
        assert loss['pose_count']==2*21*3 and loss['root_count']==6
        loss['total'].backward()
        optimizer.step()
        initial = float(encoder_loss(model(**batch['sensor']),batch['targets'])['total'].detach())
        for _ in range(15):
            optimizer.zero_grad()
            loss = encoder_loss(model(**batch['sensor']),batch['targets'])
            loss['total'].backward()
            optimizer.step()
        final = float(encoder_loss(model(**batch['sensor']),batch['targets'])['total'].detach())
        assert final<initial
        invalid_target = {k:v.clone() for k,v in batch['targets'].items()}
        invalid_target['annotation_joint_valid'].zero_()
        invalid_target['annotation_root_valid'].zero_()
        invalid_target['joint_relative_m'].fill_(float('nan'))
        invalid_target['root_m'].fill_(float('nan'))
        loss = encoder_loss(model(**batch['sensor']),invalid_target)
        assert loss['total'].item()==0 and loss['pose_count']==0 and loss['root_count']==0
        r = rotation(np.eye(3))
        calibration = {'vicon_to_cam_rotmatrix':r,'radar_to_cam_rotmatrix':r,'vicon_to_cam_tvec':np.array([1000,0,0]),'radar_to_cam_tvec':np.array([0,1,0])}
        coord = {'source_frame':'vicon','vicon_to_cam_translation_unit':'millimeter','radar_to_cam_translation_unit':'meter'}
        result = transform_to_radar(np.array([[1.,2,3]]),calibration,coord)
        assert np.allclose(result,[[2,1,3]])
        raises(lambda:rotation(np.eye(3)*2))
        target = target_from_joints(np.arange(66,dtype=np.float32).reshape(22,3),0)
        assert np.allclose(target['joint_relative_m']+target['root_m'],target['joint_global_m'])
        checkpoint = root/'encoder.pt'
        save_checkpoint(checkpoint,model,{'contract_version':CONTRACT_VERSION,'lineage':{'fixture_hash':'fixture'},'data_kind':'synthetic'})
        restored = M4HumanSetEncoderV2()
        load_checkpoint(checkpoint,restored,{'fixture_hash':'fixture'})
        assert state_dict_hash(restored)==state_dict_hash(model)
        raises(lambda:load_checkpoint(checkpoint,restored,{'fixture_hash':'foreign'}))
        metrics,diagnostics,ids = evaluate_encoder(model,fixture_data,'cpu',4,root/'predictions.jsonl')
        recomputed = recompute_encoder_metrics(root/'predictions.jsonl')
        assert recomputed['global']['count']==37*22
        assert np.isclose(recomputed['global']['error_sum'],metrics['global']['error_sum'])
        state = root/'state'
        state.mkdir()
        state_rows=[]
        for i in range(len(ds)):
            state_rows.append(ds[i]['provenance'])
        for name,array in [('P_enc',np.zeros((37,22,3),np.float32)),('r_enc',np.zeros((37,3),np.float32)),('f_enc',np.zeros((37,256),np.float32)),('time_s',np.array([r['time_s'] for r in state_rows],np.float64)),('sensor_valid',np.ones(37,bool))]:
            np.save(state/(name+'.npy'),array,allow_pickle=False)
        (state/'manifest.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in state_rows))
        atomic_json(state/'metadata.json',{'contract_version':CONTRACT_VERSION,'data_kind':'synthetic','complete':True,'count':37,
                    'scientific_eligible':False,'max_gap_s':.2,'lineage':{'fixture_hash':'fixture'},
                    'manifest_hash':file_sha256(state/'manifest.jsonl'),
                    'arrays':{name:file_sha256(state/(name+'.npy')) for name in ('P_enc','r_enc','f_enc','time_s','sensor_valid')}})
        windows = EncodedStateWindowDataset(state,allow_debug=True)
        assert len(windows)==1 and windows[0]['sensor']['P_enc_relative_m'].shape==(32,22,3)
        assert len(windows[0]['provenance']['context_source_frames'])==32
        original_manifest = (state/'manifest.jsonl').read_text()
        state_rows[1]['segment_id'] = 'interleaved_segment'
        (state/'manifest.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in state_rows))
        metadata = json.loads((state/'metadata.json').read_text())
        metadata['manifest_hash'] = file_sha256(state/'manifest.jsonl')
        atomic_json(state/'metadata.json',metadata,overwrite=True)
        raises(lambda:EncodedStateWindowDataset(state,allow_debug=True))
        (state/'manifest.jsonl').write_text(original_manifest)

        ds.reader.close()
        pipeline = synthetic_pipeline(root)
        print(json.dumps({'data_kind':'synthetic','status':'passed','checks':['bounded_serializers','endianness','safe_keys','immutable_flat_lmdb_spawn','split_context','sampling_permutation','masked_pooling','head_gradients','tiny_overfit','coordinate_units','strict_checkpoint','metric_recomputation','state_windows','exact_source_metadata_keys','interleaved_segments_rejected','spawn_targets_states','persistent_worker_epoch','target_absent_sensor_extraction','encoder_extraction_batch1_batch4_fp32_parity_3e-6'],
                          'parameter_count':sum(p.numel() for p in model.parameters()),'tiny_overfit_loss_initial':initial,'tiny_overfit_loss_final':final,
                          'synthetic_pipeline':pipeline,'real_data_gates':'not_run'},indent=2))

if __name__=='__main__':
    main()
