"""Inventory or verify an immutable M4Human snapshot; never write inside source root."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eksperimen_model.datasets.m4human_dataset import RPCOnlyReader, clean_rpc, decode_msgpack, parse_source_key, read_json, read_manifest, validate_frames, validate_schema
from eksperimen_model.utils.m4human_runtime import CONTRACT_VERSION, atomic_json, canonical_hash, file_sha256


def inventory(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'M4Human dataset directory unavailable: {root}')
    entries = []
    for path in sorted(root.iterdir()):
        if path.name.endswith('.lmdb'):
            entries.append({'name':path.name,'kind':'directory' if path.is_dir() else 'file',
                            'bytes':sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) if path.is_dir() else path.stat().st_size})
    return {'root':str(root), 'databases':entries, 'lock_files':[p.name for p in sorted(root.glob('*.lmdb-lock'))],
            'source_modified':False, 'lock_files_are_labels':False}


def inspect(root, output, sample_limit=8):
    """Read bounded key examples only; no schema or scientific semantics inferred."""
    import lmdb
    source, output = Path(root).resolve(), Path(output).resolve()
    if source == output or source in output.parents or not 1 <= sample_limit <= 100:
        raise ValueError('outside-source output and sample limit 1..100 required')
    output.mkdir(parents=True,exist_ok=False)
    result = inventory(source)
    result.update(status='not_verified',samples={},entry_counts={},
                  required_evidence=['immutable snapshot','source hashes/commit','RPC channel/unit/sentinel',
                                     'recording and time policy','subject split','body joint map','coordinate transform'])
    for entry in result['databases']:
        path = source/entry['name']
        env = lmdb.open(str(path),subdir=path.is_dir(),readonly=True,create=False,lock=False,writemap=False,readahead=False)
        examples = []
        with env.begin(write=False) as tx:
            result['entry_counts'][entry['name']] = tx.stat()['entries']
            for key,value in tx.cursor():
                if len(examples) >= sample_limit:
                    break
                item = {'key_utf8':key.decode('utf-8','replace'),'value_bytes':len(value)}
                try:
                    item['source_identity'] = parse_source_key(key)
                except (ValueError,UnicodeDecodeError):
                    item['source_identity'] = None
                if entry['name']=='radar_pc.lmdb' and len(value)>=12:
                    import struct
                    item['header_little_endian'] = struct.unpack_from('<III',value)
                    item['header_big_endian'] = struct.unpack_from('>III',value)
                if entry['name']=='indicator.lmdb':
                    try:
                        decoded = decode_msgpack(value,max_bytes=1024*1024)
                        item['indicator_type'] = type(decoded).__name__
                        item['indicator_keys'] = list(decoded)[:20] if isinstance(decoded,dict) else None
                    except ValueError as exc:
                        item['indicator_error'] = str(exc)
                examples.append(item)
        env.close()
        result['samples'][entry['name']] = examples
    atomic_json(output/'inspection.json',result)
    return result


def draft_manifest(root, output, policy_path):
    """Build an unverified index only from explicit split/take/time assertions."""
    import lmdb
    source, output = Path(root).resolve(), Path(output).resolve()
    if source == output or source in output.parents:
        raise ValueError('draft output must be outside source')
    if not policy_path:
        raise ValueError('--policy is required for draft mode')
    policy = read_json(policy_path)
    if not isinstance(policy.get('package_id'),str) or not policy['package_id']:
        raise ValueError('explicit package_id required')
    if policy.get('single_take_per_subject_action_verified') is not True or policy.get('nominal_grid_verified') is not True or not policy.get('evidence'):
        raise ValueError('recording and nominal time evidence must be supplied')
    hz = policy.get('nominal_hz')
    if not isinstance(hz,(int,float)) or hz <= 0 or hz > 1000:
        raise ValueError('audited nominal_hz required')
    splits = policy.get('subject_splits',{})
    if not isinstance(splits,dict) or not splits or any(v not in ('train','val','test') for v in splits.values()):
        raise ValueError('explicit subject_splits required')
    path = source/'radar_pc.lmdb'
    if not path.exists():
        raise FileNotFoundError(path)
    output.mkdir(parents=True,exist_ok=False)
    env = lmdb.open(str(path),subdir=path.is_dir(),readonly=True,create=False,lock=False,writemap=False,readahead=False)
    keys = []
    with env.begin(write=False) as tx:
        for key in tx.cursor().iternext(keys=True,values=False):
            subject,action,frame = parse_source_key(key)
            if str(subject) not in splits:
                raise ValueError(f'subject {subject} has no explicit split')
            keys.append((subject,action,frame,key.decode('utf-8')))
    env.close()
    keys.sort()
    previous,segment,count = None,0,0
    with (output/'draft_frames.jsonl').open('w',encoding='utf-8') as handle:
        for subject,action,frame,key in keys:
            group = (subject,action)
            if previous is None or group != previous[:2] or frame != previous[2]+1:
                segment += 1
            row = {'frame_uid':f'{policy["package_id"]}:{subject}:{action}:{frame}',
                   'source_key':key,'subject_id':subject,'action_id':action,
                   'recording_id':f'{policy["package_id"]}:{subject}:{action}',
                   'segment_id':f'{policy["package_id"]}:{subject}:{action}:segment{segment}',
                   'source_frame_id':frame,'source_time_s':None,'time_s':frame/hz,
                   'time_source':'nominal_frame_index','split':splits[str(subject)],
                   'sensor_frame_valid':False,'source_manifest_version':'unverified_draft_v1'}
            handle.write(json.dumps(row,allow_nan=False)+'\n')
            previous=(subject,action,frame)
            count += 1
    report = {'status':'not_verified','data_kind':'m4human','count':count,'policy_hash':file_sha256(policy_path),
              'manifest_hash':file_sha256(output/'draft_frames.jsonl'),'next_step':'audit --mode verify with schema, joint map, coordinate evidence'}
    atomic_json(output/'draft_report.json',report)
    return report


def audit(root, output, schema_path=None, manifest_path=None, joint_map_path=None, coordinate_path=None):
    output = Path(output).resolve()
    source = Path(root).resolve()
    if output == source or source in output.parents:
        raise ValueError('audit output must be outside immutable dataset root')
    output.mkdir(parents=True, exist_ok=False)
    inv = inventory(root)
    atomic_json(output / 'inventory.json', inv)
    missing = [name for name, path in [('schema',schema_path),('frame_manifest',manifest_path),('joint_map',joint_map_path),('coordinate_audit',coordinate_path)] if path is None]
    report = {'contract_version':CONTRACT_VERSION,'data_kind':'m4human','status':'not_verified','missing_gates':missing,'source_modified':False}
    if missing:
        atomic_json(output / 'audit_report.json',report)
        return report
    schema = validate_schema(read_json(schema_path))
    if schema.get('data_kind') == 'synthetic':
        report['data_kind'] = 'synthetic'
    provenance = schema.get('source_provenance',{})
    if any(not provenance.get(k) for k in ('package_id','upstream_commit','serializer_sha256','calibration_sha256','rpc_preprocessing')) or len(provenance['upstream_commit']) != 40:
        raise ValueError('source package/full upstream commit/hash/preprocessing provenance required')
    rows = validate_frames(read_manifest(manifest_path),max_gap_s=schema['max_gap_s'])
    joint_map = read_json(joint_map_path)
    if joint_map.get('audited') is not True or len(joint_map.get('ordered_names',[])) != 22 or len(joint_map.get('source_row_indices',[])) != 22 or joint_map['ordered_names'][joint_map.get('pelvis_index',-1)] != 'pelvis':
        raise ValueError('audited ordered body joint map22/pelvis required')
    coordinate = read_json(coordinate_path)
    if coordinate.get('audited') is not True or coordinate.get('output_unit') != 'meter' or not coordinate.get('radar_frame') or not coordinate.get('evidence'):
        raise ValueError('coordinate/unit/origin evidence gate unresolved')
    import lmdb
    counts = {}
    for entry in inv['databases']:
        path = source / entry['name']
        env = lmdb.open(str(path), subdir=path.is_dir(), readonly=True,create=False,lock=False,writemap=False,readahead=False)
        with env.begin(write=False) as tx:
            counts[entry['name']] = tx.stat()['entries']
        env.close()
    if 'radar_pc.lmdb' not in counts or 'params.lmdb' not in counts or 'calib.lmdb' not in counts or 'indicator.lmdb' not in counts:
        raise ValueError('required RPC/target/audit LMDB source missing')
    reader = RPCOnlyReader(source,schema)
    rejected, cloud_counts, partition_counts = Counter(),Counter(),Counter()
    unique = set()
    with (output/'frames.jsonl').open('w',encoding='utf-8') as good, (output/'rejections.jsonl').open('w',encoding='utf-8') as bad:
        for row in rows:
            unique.add(row['source_key'])
            try:
                clean, count = clean_rpc(reader.read(row['source_key']),schema)
                valid = bool(len(clean))
                row = dict(row, schema_valid=True,finite_point_count=len(clean),sensor_frame_valid=valid,**count)
                if not valid:
                    rejected['empty_rpc'] += 1
                cloud_counts.update({k:v for k,v in count.items()})
                partition_counts[row['split']] += 1
                good.write(json.dumps(row,allow_nan=False)+'\n')
            except (ValueError,KeyError) as exc:
                rejected[type(exc).__name__] += 1
                bad.write(json.dumps({'frame_uid':row['frame_uid'],'source_key':row['source_key'],'phase':'schema','reason':str(exc)})+'\n')
                # Keep invalid source frame in the manifest so context cannot bridge it.
                row = dict(row,schema_valid=False,sensor_frame_valid=False,finite_point_count=0)
                good.write(json.dumps(row,allow_nan=False)+'\n')
    reader.close()
    if len(unique) != counts['radar_pc.lmdb']:
        raise ValueError('source manifest must cover full RPC inventory; unseen entries cannot be silently excluded')
    report.update(status='verified',missing_gates=[],schema_hash=file_sha256(schema_path),split_manifest_hash=file_sha256(manifest_path),
                  joint_map_hash=file_sha256(joint_map_path),coordinate_hash=file_sha256(coordinate_path),
                  source_provenance=provenance,entry_counts=counts,frame_count=len(rows),partition_counts=dict(partition_counts),
                  rejection_counts=dict(rejected),point_counts=dict(cloud_counts),frame_manifest_hash=file_sha256(output/'frames.jsonl'),
                  audit_scope='structural_and_supplied_semantics; supplied scientific evidence must be inspected separately')
    atomic_json(output/'audit_report.json',report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='datasets/m4human')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--mode',choices=('inspect','draft','verify'),default='verify')
    p.add_argument('--sample-limit',type=int,default=8)
    p.add_argument('--policy')
    for name in ('schema','frame-manifest','joint-map','coordinate-audit'):
        p.add_argument('--'+name)
    a = p.parse_args()
    result = (inspect(a.root,a.output_dir,a.sample_limit) if a.mode=='inspect' else
              draft_manifest(a.root,a.output_dir,a.policy) if a.mode=='draft' else
              audit(a.root,a.output_dir,a.schema,a.frame_manifest,a.joint_map,a.coordinate_audit))
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__ == '__main__':
    main()
