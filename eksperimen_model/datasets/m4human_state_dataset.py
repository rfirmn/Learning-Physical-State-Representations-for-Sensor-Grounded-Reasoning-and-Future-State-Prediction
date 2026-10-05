"""Lazy read-only encoder-state windows; annotation arrays are joined afterwards."""
from pathlib import Path
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from .m4human_dataset import CONTRACT_VERSION, OfflineTargets, read_json, read_manifest, parse_source_key
from eksperimen_model.utils.m4human_runtime import file_sha256, require_lineage


class EncodedStateWindowDataset(Dataset):
    def __init__(self, cache_dir, length=32, stride=8, split=None, targets_dir=None, expected_lineage=None, allow_debug=False):
        self.root = Path(cache_dir)
        self.metadata = read_json(self.root / 'metadata.json')
        if self.metadata.get('complete') is not True or self.metadata.get('contract_version') != CONTRACT_VERSION:
            raise ValueError('state cache incomplete/incompatible')
        if self.metadata.get('data_kind') != 'm4human' or self.metadata.get('scientific_eligible') is not True:
            if not allow_debug:
                raise ValueError('state cache is debug/non-scientific')
        require_lineage(self.metadata, expected_lineage or {})
        if self.metadata.get('manifest_hash') != file_sha256(self.root / 'manifest.jsonl'):
            raise ValueError('state manifest hash mismatch')
        gap = self.metadata.get('max_gap_s')
        if not isinstance(gap,(float,int)) or not 0 < gap < 60:
            raise ValueError('state cache max_gap_s missing')
        if length != 32 or stride <= 0:
            raise ValueError('reference motion length32 and positive stride required')
        self.rows = read_manifest(self.root / 'manifest.jsonl')
        if self.metadata.get('count') != len(self.rows):
            raise ValueError('cache manifest count mismatch')
        shapes = {'P_enc': ((len(self.rows),22,3),np.float32), 'r_enc':((len(self.rows),3),np.float32), 'f_enc':((len(self.rows),256),np.float32), 'time_s':((len(self.rows),),np.float64), 'sensor_valid':((len(self.rows),),np.bool_)}
        self.arrays = {}
        for name, (shape,dtype) in shapes.items():
            if self.metadata.get('arrays',{}).get(name) != file_sha256(self.root / (name+'.npy')):
                raise ValueError(f'state cache array hash mismatch:{name}')
            arr = np.load(self.root / (name + '.npy'), mmap_mode='r', allow_pickle=False)
            if arr.shape != shape or arr.dtype != dtype:
                raise ValueError(f'state cache shape/dtype mismatch:{name}')
            self.arrays[name] = arr
        self.targets = OfflineTargets(targets_dir) if targets_dir else None
        if self.targets:
            if self.metadata.get('lineage',{}).get('target_hash') != file_sha256(self.targets.root/'metadata.json'):
                raise ValueError('state/target export mismatch')
            for key in ('frame_manifest_hash','joint_map_hash','coordinate_hash'):
                if self.targets.metadata.get('lineage',{}).get(key) != self.metadata.get('lineage',{}).get(key):
                    raise ValueError(f'state/target lineage mismatch:{key}')
        self.length = length
        self.starts = []
        group_start = 0
        subjects = {}
        recordings = {}
        previous = None
        for i, row in enumerate(self.rows):
            if row.get('split') not in ('train','val','test') or row.get('time_source') not in ('measured','nominal_frame_index') or type(row.get('time_s')) not in (float,int) or not math.isfinite(row['time_s']) or not math.isclose(float(self.arrays['time_s'][i]),row['time_s'],rel_tol=0,abs_tol=1e-9):
                raise ValueError('state timestamp/manifest mismatch')
            if row.get('sensor_frame_valid') is not True or self.arrays['sensor_valid'][i] != True:
                raise ValueError('state extraction contains invalid sensor anchor')
            if parse_source_key(row['source_key']) != (row['subject_id'],row['action_id'],row['source_frame_id']):
                raise ValueError('state source identity mismatch')
            if row['subject_id'] in subjects and subjects[row['subject_id']] != row['split']:
                raise ValueError('state subject split leakage')
            subjects[row['subject_id']] = row['split']
            identity = (row['subject_id'],row['action_id'],row['split'],row['time_source'])
            if row['recording_id'] in recordings and recordings[row['recording_id']] != identity:
                raise ValueError('state recording identity mismatch')
            recordings[row['recording_id']] = identity
            if len(row['context_source_frames']) != 4 or row['context_source_frames'][-1] != row['frame_uid']:
                raise ValueError('context provenance incomplete')
            group = (row['recording_id'], row['segment_id'], row['split'],row['subject_id'],row['action_id'])
            if previous is None or group != previous:
                group_start = i
            else:
                prior = self.rows[i-1]
                if row['time_s'] <= prior['time_s'] or row['time_s']-prior['time_s'] > gap or (row['time_source']=='nominal_frame_index' and row['source_frame_id'] != prior['source_frame_id']+1):
                    group_start = i
                elif row['context_source_frames'][:-1] != prior['context_source_frames'][1:]:
                    group_start = i
            previous = group
            start = i - length + 1
            if start >= group_start and (start - group_start) % stride == 0 and (split is None or row['split'] == split) and self.arrays['sensor_valid'][start:i+1].all():
                self.starts.append(start)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = self.starts[index]
        stop = start + self.length
        rows = self.rows[start:stop]
        sensor = {name: torch.from_numpy(np.array(self.arrays[source][start:stop], copy=True)) for name, source in
                  [('P_enc_relative_m','P_enc'), ('r_enc_m','r_enc'), ('f_enc','f_enc'), ('time_s','time_s'), ('sensor_valid','sensor_valid')]}
        sensor['f_enc'] = sensor['f_enc'].float()
        mask = sensor['sensor_valid'].bool()
        sensor.update(root_sensor_valid=mask, feature_valid=mask, context_time_valid=mask,
                      joint_sensor_valid=mask[:,None].expand(-1,22))
        targets = {}
        if self.targets:
            values = [self.targets.read(r['frame_uid']) for r in rows]
            targets = {name:torch.stack([v[name] for v in values]) for name in values[0]}
        provenance = {name: rows[-1][name] for name in ('recording_id','segment_id','subject_id','action_id','split')}
        provenance.update(frame_uids=[r['frame_uid'] for r in rows],
                          source_keys=[r['source_key'] for r in rows],
                          context_source_frames=[r['context_source_frames'] for r in rows],
                          window_id=f'{rows[0]["frame_uid"]}..{rows[-1]["frame_uid"]}')
        return {'sensor':sensor, 'targets':targets, 'provenance':provenance}
