"""Read-only RPC/target adapters. No GT, body model, RGB or activity enters sensor forward."""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
from collections.abc import Sequence, Mapping
import io
import struct
import numpy as np
import torch
from torch.utils.data import Dataset

CONTRACT_VERSION = 'm4human_kinetok_v3'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def decode_json_row(line, strict=False):
    if not strict:
        return json.loads(line)
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate JSON field: {key}')
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError(f'nonfinite JSON: {value}')
    return json.loads(line, object_pairs_hook=unique_keys, parse_constant=nonfinite)


def share_integer_arrays(state, names):
    for name in names:
        value = state[name]
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        if not value.is_shared():
            value.share_memory_()
        state[name] = value


class UIDIndex(Mapping):
    """16 bytes/row; hash collisions resolve against the exact immutable source ID."""
    def __init__(self, rows, identity_field='frame_uid', rows_as_values=False):
        self.rows, self.identity_field, self.rows_as_values = rows, identity_field, rows_as_values
        hashes = np.fromiter((self.hash_uid(row[identity_field]) for row in rows), dtype=np.int64, count=len(rows))
        self.indices = np.argsort(hashes, kind='stable').astype(np.int64)
        self.hashes = hashes[self.indices]
        # Only collision groups need decoded IDs in a set; never trust hash uniqueness.
        start = 0
        while start < len(self.hashes):
            stop = int(np.searchsorted(self.hashes, self.hashes[start], side='right'))
            if stop - start > 1:
                ids = set()
                for i in self.indices[start:stop]:
                    uid = rows[int(i)][identity_field]
                    if uid in ids:
                        raise ValueError(f'duplicate {identity_field}')
                    ids.add(uid)
            start = stop

    @staticmethod
    def hash_uid(uid):
        return int.from_bytes(hashlib.sha256(uid.encode('utf-8')).digest()[:8], 'little', signed=True)

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return (row[self.identity_field] for row in self.rows)

    def __getitem__(self, uid):
        hashed = self.hash_uid(uid)
        # Shared torch storage is exposed to NumPy without copying after spawn.
        hashes = np.asarray(self.hashes)
        start, stop = np.searchsorted(hashes, hashed, side='left'), np.searchsorted(hashes, hashed, side='right')
        for position in range(int(start), int(stop)):
            index = int(self.indices[position])
            row = self.rows[index]
            if row[self.identity_field] == uid:
                return row if self.rows_as_values else index
        raise KeyError(uid)

    def __getstate__(self):
        state = self.__dict__.copy()
        share_integer_arrays(state, ('hashes', 'indices'))
        self.hashes, self.indices = state['hashes'], state['indices']
        return state


def checked_tensor_load(path, expected_hash, max_bytes=64 * 1024 * 1024):
    """Hash the same bounded window bytes that are deserialized; no double disk read."""
    with Path(path).open('rb') as handle:
        if Path(path).stat().st_size > max_bytes:
            raise ValueError('window tensor exceeds64MiB bound')
        payload = handle.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError('window tensor exceeds64MiB bound')
    if not expected_hash or hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError(f'cache payload changed: {Path(path).name}')
    return torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)


class ManifestSequence(Sequence):
    """Only line offsets in RAM; decode a row on demand, including Windows spawn."""
    def __init__(self, path, identity_field='frame_uid', strict_json=False):
        self.path = Path(path)
        self.identity_field, self.strict_json = identity_field, strict_json
        self._handle = None
        self._handle_pid = None
        offsets = []
        seen = set()
        with self.path.open('rb') as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = decode_json_row(line, self.strict_json)
                uid = row[self.identity_field]
                if uid in seen:
                    raise ValueError(f'duplicate {self.identity_field}')
                seen.add(uid)
                offsets.append(offset)
        self.offsets = np.asarray(offsets,dtype=np.int64)
        self._handle = None
        self._handle_pid = None

    def __eq__(self, other):
        return isinstance(other, Sequence) and len(self) == len(other) and all(a == b for a,b in zip(self,other))

    def _get_handle(self):
        pid = os.getpid()
        if self._handle is None or self._handle_pid != pid or self._handle.closed:
            self._handle = self.path.open('rb')
            self._handle_pid = pid
        return self._handle

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self,index):
        if isinstance(index,slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        handle = self._get_handle()
        handle.seek(int(self.offsets[index]))
        return decode_json_row(handle.readline(), self.strict_json)

    def close(self):
        if self._handle is not None and not self._handle.closed:
            self._handle.close()
            self._handle = None

    def __del__(self):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_handle'] = None
        state['_handle_pid'] = None
        share_integer_arrays(state, ('offsets',))
        self.offsets = state['offsets']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._handle = None
        self._handle_pid = None


class ManifestView(Sequence):
    def __init__(self, rows, indices):
        self.rows,self.indices = rows,np.asarray(indices,dtype=np.int64)
    def __getstate__(self):
        state = self.__dict__.copy()
        share_integer_arrays(state, ('indices',))
        self.indices = state['indices']
        return state

    def __len__(self):
        return len(self.indices)
    def __getitem__(self,index):
        if isinstance(index,slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        return self.rows[int(self.indices[index])]


def read_manifest(path):
    return ManifestSequence(path)


def validate_schema(schema):
    required = {'audited': True, 'immutable_snapshot': True, 'channels': ['x', 'y', 'z', 'intensity'], 'xyz_unit': 'meter'}
    for key, value in required.items():
        if schema.get(key) != value:
            raise ValueError(f'schema gate unresolved: {key}')
    if schema.get('byte_order') not in ('little', 'big'):
        raise ValueError('producer byte_order must be audited')
    for key in ('max_points', 'max_value_bytes'):
        if type(schema.get(key)) is not int or schema[key] <= 0:
            raise ValueError(f'bounded {key} required')
    if not isinstance(schema.get('max_gap_s'), (int, float)) or not 0 < schema['max_gap_s'] < 60:
        raise ValueError('audited positive max_gap_s required')
    if schema.get('sentinel_policy') not in ('none', 'exact_row'):
        raise ValueError('sentinel convention must be audited')
    if schema.get('sentinel_policy') == 'exact_row' and np.asarray(schema.get('sentinel_row')).shape != (4,):
        raise ValueError('sentinel_row shape must be4')
    return schema


def decode_rpc(value, schema):
    """Official native (=I shape) payload, with explicit producer endian and bounds."""
    if value is None or not 12 <= len(value) <= schema['max_value_bytes']:
        raise ValueError('missing/truncated/oversized RPC')
    endian = '<' if schema['byte_order'] == 'little' else '>'
    ndim, n, channels = struct.unpack_from(endian + 'III', value)
    if ndim != 2 or channels != 4 or n > schema['max_points'] or len(value) != 12 + n * 16:
        raise ValueError('RPC must be exact bounded Nx4 float32')
    return np.frombuffer(value, dtype=endian + 'f4', offset=12).astype(np.float32, copy=True).reshape(n, 4)


def decode_msgpack(value, max_bytes=16 * 1024 * 1024, max_depth=12):
    if value is None or len(value) > max_bytes:
        raise ValueError('missing/oversized MessagePack')
    import msgpack
    def reject_extension(code, data):
        raise ValueError('MessagePack extension forbidden')
    obj = msgpack.unpackb(value, raw=False, strict_map_key=True, ext_hook=reject_extension,
                         max_str_len=max_bytes, max_bin_len=max_bytes, max_array_len=100000,
                         max_map_len=10000, max_ext_len=0)
    def checked(x, depth=0):
        if depth > max_depth:
            raise ValueError('MessagePack nesting limit')
        if isinstance(x, dict):
            if '__nd__' in x:
                if x.get('__nd__') is not True or set(x) != {'__nd__', 'dtype', 'shape', 'data'}:
                    raise ValueError('invalid numeric array tag')
                try:
                    dtype = np.dtype(x['dtype'])
                except (TypeError, ValueError) as exc:
                    raise ValueError('invalid array dtype') from exc
                shape = x['shape']
                if dtype.kind not in 'biuf' or dtype.itemsize > 8 or not isinstance(shape, list) or len(shape) > 5 or any(type(s) is not int or s < 0 for s in shape):
                    raise ValueError('unsafe array dtype/shape')
                size = math.prod(shape) * dtype.itemsize
                if size > max_bytes or not isinstance(x['data'], bytes) or len(x['data']) != size:
                    raise ValueError('array payload length mismatch')
                return np.frombuffer(x['data'], dtype=dtype).copy().reshape(shape)
            if any(not isinstance(k, str) for k in x):
                raise ValueError('metadata keys must be strings')
            return {k: checked(v, depth + 1) for k, v in x.items()}
        if isinstance(x, list):
            return [checked(v, depth + 1) for v in x]
        if x is None or isinstance(x, (str, bytes, int, float, bool)):
            return x
        raise ValueError('unknown MessagePack value')
    return checked(obj)


def parse_source_key(key):
    if isinstance(key, bytes):
        key = key.decode('utf-8', errors='strict')
    if not isinstance(key, str) or len(key) > 128:
        raise ValueError('oversized source key')
    try:
        parts = ast.literal_eval(key)
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ValueError('source key must be integer triple') from exc
    if not isinstance(parts, (list, tuple)) or len(parts) != 3 or any(type(v) is not int or abs(v) > 2**31 - 1 for v in parts):
        raise ValueError('source key must be bounded integer triple')
    return tuple(parts)


class RPCOnlyReader:
    _environments = {}
    def __init__(self, root, schema):
        self.root = Path(root)
        self.schema = validate_schema(schema)
        self.path = self.root / 'radar_pc.lmdb'
        if not self.path.exists():
            raise FileNotFoundError(f'Existing dataset required: {self.path}; no automatic preprocessing')
        self._env = None
        self._pid = None

    def environment(self):
        if self._env is None or self._pid != os.getpid():
            self.close()
            import lmdb
            key = (os.getpid(),str(self.path.resolve()))
            if key not in self._environments:
                self._environments[key] = [lmdb.open(str(self.path), subdir=self.path.is_dir(), readonly=True,
                                  create=False, lock=False, writemap=False, readahead=False,
                                  max_readers=256),0]
            self._environments[key][1] += 1
            self._env = self._environments[key][0]
            self._pid = os.getpid()
        return self._env

    def read(self, source_key):
        parse_source_key(source_key)
        with self.environment().begin(write=False, buffers=False) as tx:
            value = tx.get(source_key.encode('utf-8'))
        return decode_rpc(value, self.schema)

    def close(self):
        if self._env is not None:
            key = (self._pid,str(self.path.resolve()))
            entry = self._environments.get(key)
            if entry and entry[0] is self._env:
                entry[1] -= 1
                if entry[1] == 0:
                    entry[0].close()
                    del self._environments[key]
        self._env = self._pid = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_env'] = state['_pid'] = None
        return state


def clean_rpc(rpc, schema):
    finite = np.isfinite(rpc).all(axis=1)
    keep = finite.copy()
    if schema['sentinel_policy'] == 'exact_row':
        keep &= ~(rpc == np.asarray(schema['sentinel_row'], dtype=np.float32)).all(axis=1)
    cleaned = rpc[keep].copy()
    return cleaned, {'n_raw': len(rpc), 'n_clean': len(cleaned), 'nonfinite': int((~finite).sum()), 'sentinel': int((finite & ~keep).sum())}


def sample_rpc(rpc, frame_uid, points=512, seed=42, epoch=None, version='canonical_sha256_v1'):
    """Order-independent selection, no repetition; epoch absent means stable cache/val."""
    if rpc.ndim != 2 or rpc.shape[1] != 4 or not np.isfinite(rpc).all():
        raise ValueError('sample_rpc requires clean Nx4 array')
    order = np.lexsort(tuple(rpc[:, c] for c in (3, 2, 1, 0)))
    selected = rpc[order]
    if len(selected) > points:
        digest = hashlib.sha256(f'{version}:{seed}:{epoch}:{frame_uid}'.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
        selected = selected[np.sort(rng.choice(len(selected), points, replace=False))]
    cloud = np.zeros((points, 4), dtype=np.float32)
    mask = np.zeros(points, dtype=bool)
    cloud[:len(selected)] = selected
    mask[:len(selected)] = True
    return cloud, mask


def validate_frames(rows, split=None, max_gap_s=None):
    required = ('frame_uid', 'source_key', 'subject_id', 'action_id', 'recording_id', 'segment_id', 'source_frame_id', 'time_s', 'time_source', 'split', 'sensor_frame_valid')
    subjects = {}
    recordings = {}
    previous = {}
    seen = set()
    last_group = None
    for row in rows:
        if any(k not in row for k in required):
            raise ValueError('frame manifest misses audited grouping/time fields')
        source_tuple = parse_source_key(row['source_key'])
        if source_tuple != (row['subject_id'], row['action_id'], row['source_frame_id']):
            raise ValueError('manifest/source key identity mismatch')
        if row['split'] not in ('train', 'val', 'test') or row['time_source'] not in ('measured', 'nominal_frame_index') or type(row['time_s']) not in (int,float) or not math.isfinite(row['time_s']) or type(row['sensor_frame_valid']) is not bool:
            raise ValueError('invalid split/time semantics')
        if row['time_source'] == 'measured' and (type(row.get('source_time_s')) not in (int,float) or row['source_time_s'] != row['time_s']):
            raise ValueError('measured time must carry matching source_time_s')
        if row['time_source'] == 'nominal_frame_index' and row.get('source_time_s') is not None:
            raise ValueError('nominal time cannot masquerade as measured time')
        subject = row['subject_id']
        if subject in subjects and subjects[subject] != row['split']:
            raise ValueError('subject leakage across splits')
        subjects[subject] = row['split']
        recording = row['recording_id']
        identity_group = (subject,row['action_id'],row['split'],row['time_source'])
        if recording in recordings and recordings[recording] != identity_group:
            raise ValueError('recording spans subject/action/split/time policy')
        recordings[recording] = identity_group
        identity = (row['recording_id'], row['source_frame_id'])
        if identity in seen:
            raise ValueError('duplicate recording frame')
        seen.add(identity)
        group = (row['recording_id'], row['segment_id'])
        if group != last_group and group in previous:
            raise ValueError('noncontiguous recording/segment group; sort manifest into contiguous segments')
        last_group = group
        if group in previous:
            p = previous[group]
            if row['time_s'] <= p['time_s'] or row['subject_id'] != p['subject_id'] or row['split'] != p['split'] or (row['time_source']=='nominal_frame_index' and row['source_frame_id'] != p['source_frame_id'] + 1) or (max_gap_s is not None and row['time_s']-p['time_s'] > max_gap_s):
                raise ValueError('segment ordering/gap/split violation')
        previous[group] = row
    return rows if split is None else ManifestView(rows,[i for i,r in enumerate(rows) if r['split'] == split])


class M4HumanCausalSensorDataset(Dataset):
    def __init__(self, root, manifest, schema, split=None, training=False, seed=42):
        self.reader = RPCOnlyReader(root, schema)
        rows = read_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest
        self.rows = validate_frames(rows, split, schema['max_gap_s'])
        self.contexts = []
        self.training, self.seed = training, seed
        self._epoch = torch.zeros((), dtype=torch.int64)
        for i in range(3, len(self.rows)):
            context = self.rows[i-3:i+1]
            group = {(r['recording_id'], r['segment_id'], r['split']) for r in context}
            if len(group) == 1 and all(r['sensor_frame_valid'] for r in context) and all(context[k+1]['time_s'] > context[k]['time_s'] and context[k+1]['time_s']-context[k]['time_s'] <= schema['max_gap_s'] and (context[k]['time_source']=='measured' or context[k+1]['source_frame_id'] == context[k]['source_frame_id']+1) for k in range(3)):
                self.contexts.append(i)
        self.contexts = np.asarray(self.contexts,dtype=np.int64)

    def __getstate__(self):
        # Spawn shares only this scalar; worker0 needs no OS shared-memory helper.
        self._epoch.share_memory_()
        state = self.__dict__.copy()
        share_integer_arrays(state, ('contexts',))
        self.contexts = state['contexts']
        return state

    @property
    def epoch(self):
        return int(self._epoch.item())

    def set_epoch(self, epoch):
        self._epoch.fill_(int(epoch))

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, index):
        anchor = int(self.contexts[index])
        context = self.rows[anchor-3:anchor+1]
        clouds, masks, counts = [], [], []
        for row in context:
            clean, count = clean_rpc(self.reader.read(row['source_key']), self.reader.schema)
            cloud, mask = sample_rpc(clean, row['frame_uid'], seed=self.seed, epoch=self.epoch if self.training else None)
            if not mask.any():
                raise ValueError(f'empty context cloud changed after audit: {row["frame_uid"]}')
            clouds.append(cloud)
            masks.append(mask)
            counts.append(count)
        sensor = {'rpc_m': torch.from_numpy(np.stack(clouds)), 'point_mask': torch.from_numpy(np.stack(masks)),
                  'time_s': torch.tensor([r['time_s'] for r in context], dtype=torch.float64)}
        provenance = dict(context[-1])
        provenance['context_source_frames'] = [r['frame_uid'] for r in context]
        provenance['counts'] = counts
        return {'sensor': sensor, 'provenance': provenance}


class OfflineTargets:
    """Labels remain outside sensor readers; array masks cannot alter contexts."""
    FIELDS = ('joint_global_m', 'root_m', 'joint_relative_m', 'annotation_joint_valid', 'annotation_root_valid')
    def __init__(self, root):
        self.root = Path(root)
        self.metadata = read_json(self.root / 'metadata.json')
        if self.metadata.get('contract_version') != CONTRACT_VERSION or self.metadata.get('complete') is not True:
            raise ValueError('incomplete or incompatible offline targets')
        from eksperimen_model.utils.m4human_runtime import file_sha256
        if self.metadata.get('manifest_hash') != file_sha256(self.root / 'manifest.jsonl'):
            raise ValueError('target manifest hash mismatch')
        self.rows = read_manifest(self.root / 'manifest.jsonl')
        self.index = UIDIndex(self.rows)
        if self.metadata.get('count') != len(self.rows):
            raise ValueError('target count mismatch')
        for name in self.FIELDS:
            if self.metadata.get('arrays',{}).get(name) != file_sha256(self.root / (name+'.npy')):
                raise ValueError(f'target array hash mismatch:{name}')
        self.arrays = {k: np.load(self.root / (k + '.npy'), mmap_mode='r', allow_pickle=False) for k in self.FIELDS}
        self._arrays_pid = os.getpid()
        shapes = {'joint_global_m': (len(self.rows),22,3), 'root_m': (len(self.rows),3), 'joint_relative_m':(len(self.rows),22,3), 'annotation_joint_valid':(len(self.rows),22), 'annotation_root_valid':(len(self.rows),)}
        for name, arr in self.arrays.items():
            if arr.shape != shapes[name] or (name.startswith('annotation') and arr.dtype != np.bool_) or (not name.startswith('annotation') and arr.dtype != np.float32):
                raise ValueError(f'invalid target array: {name}')

    def __getstate__(self):
        state = self.__dict__.copy()
        state['arrays'] = {}
        state['_arrays_pid'] = None
        return state

    def _ensure_arrays(self):
        if self._arrays_pid != os.getpid():
            self.arrays = {k: np.load(self.root / (k + '.npy'), mmap_mode='r', allow_pickle=False) for k in self.FIELDS}
            self._arrays_pid = os.getpid()

    def read(self, frame_uid):
        self._ensure_arrays()
        i = self.index.get(frame_uid)
        if i is None:
            return {'joint_global_m': torch.zeros(22,3), 'root_m': torch.zeros(3), 'joint_relative_m': torch.zeros(22,3),
                    'annotation_joint_valid': torch.zeros(22,dtype=torch.bool), 'annotation_root_valid':torch.tensor(False),
                    'annotation_position_valid_joint':torch.zeros(22,dtype=torch.bool), 'annotation_position_valid_root':torch.tensor(False)}
        target = {k: torch.from_numpy(np.array(v[i], copy=True)) for k, v in self.arrays.items()}
        target['annotation_position_valid_joint'] = target['annotation_joint_valid'] & target['annotation_root_valid']
        target['annotation_position_valid_root'] = target['annotation_root_valid']
        return target


class EncoderTrainingDataset(Dataset):
    def __init__(self, sensor_dataset, targets):
        self.sensor_dataset = sensor_dataset
        self.targets = targets if isinstance(targets, OfflineTargets) else OfflineTargets(targets)
    def __len__(self):
        return len(self.sensor_dataset)
    def __getitem__(self, index):
        sample = self.sensor_dataset[index]
        sample['targets'] = self.targets.read(sample['provenance']['frame_uid'])
        return sample
