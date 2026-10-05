"""Frozen sensor-only E extraction with strict lineage and atomic cache publication."""
import argparse
import json
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import yaml
from eksperimen_model.datasets.m4human_dataset import M4HumanCausalSensorDataset, read_json
from eksperimen_model.models.m4human_encoder import M4HumanSetEncoderV2
from eksperimen_model.utils.m4human_runtime import (CONTRACT_VERSION, atomic_json, canonical_hash, file_sha256, freeze,
                                     load_checkpoint, state_dict_hash)
from eksperimen_model.utils.m4human_kinematics import DEFAULT_RECIPE


def extract(config,checkpoint,output_dir,split=None,device='cpu',allow_debug=False):
    paths = config['paths']
    kind = config.get('data_kind')
    if config.get('contract_version')!=CONTRACT_VERSION or kind not in (('m4human','synthetic') if allow_debug else ('m4human',)):
        raise ValueError('real extraction requires audited m4human v3 config; synthetic needs --allow-debug')
    for key in ('schema','frame_manifest','joint_map','coordinate_audit','audit_report'):
        if not paths.get(key):
            raise ValueError(f'audit path unresolved:{key}')
    audit = read_json(paths['audit_report'])
    if audit.get('status')!='verified' or audit.get('data_kind')!=kind:
        raise ValueError('dataset audit not verified')
    lineage = {k:file_sha256(paths[path]) for k,path in [('schema_hash','schema'),('frame_manifest_hash','frame_manifest'),('joint_map_hash','joint_map'),('coordinate_hash','coordinate_audit')]}
    for key in lineage:
        if audit.get(key)!=lineage[key]:
            raise ValueError(f'foreign audit lineage:{key}')
    lineage['source_hash'] = canonical_hash(audit['source_provenance'])
    lineage['split_hash'] = lineage['frame_manifest_hash']
    lineage['recipe_hash'] = canonical_hash(DEFAULT_RECIPE)
    # Target provenance stays in the frozen checkpoint, but inference never
    # opens annotation arrays or requires their files to remain mounted.
    joint_map = read_json(paths['joint_map'])
    model = M4HumanSetEncoderV2(joint_map['pelvis_index'])
    payload = load_checkpoint(checkpoint,model,lineage)
    if payload['metadata'].get('data_kind')!=kind:
        raise ValueError('encoder checkpoint data kind mismatch')
    expected_normalizer_hash = payload['metadata']['lineage'].get('normalizer_hash')
    if canonical_hash(payload['metadata']['normalizer'])!=expected_normalizer_hash:
        raise ValueError('normalizer metadata mismatch')
    norm = payload['metadata']['normalizer']
    if not torch.allclose(model.input_mean,torch.tensor(norm['mean'])) or not torch.allclose(model.input_std,torch.tensor(norm['std'])):
        raise ValueError('checkpoint normalizer buffers mismatch')
    root_gate = config['training']['root_quality_gate_m']
    root_error = payload['metadata'].get('validation_metrics',{}).get('root',{}).get('mean_m')
    eligible = kind=='m4human' and root_gate is not None and root_error is not None and root_error<=root_gate
    if not eligible and not allow_debug:
        raise ValueError('root gate unresolved/failed; --allow-debug marks non-scientific cache explicitly')
    freeze(model.to(device))
    before = state_dict_hash(model)
    dataset = M4HumanCausalSensorDataset(paths['dataset_root'],paths['frame_manifest'],read_json(paths['schema']),split,False,config['training']['seed'])
    if not len(dataset):
        raise ValueError('no eligible anchors')
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    if output.resolve()==Path(paths['dataset_root']).resolve() or Path(paths['dataset_root']).resolve() in output.resolve().parents:
        raise ValueError('derived cache cannot write source')
    temporary = output.with_name(output.name+'.incomplete')
    temporary.mkdir(parents=True,exist_ok=False)
    fields = {'P_enc':((len(dataset),22,3),np.float32),'r_enc':((len(dataset),3),np.float32),'f_enc':((len(dataset),256),np.float32),
              'time_s':((len(dataset),),np.float64),'sensor_valid':((len(dataset),),bool)}
    arrays = {name:np.lib.format.open_memmap(temporary/(name+'.npy'),mode='w+',dtype=dtype,shape=shape) for name,(shape,dtype) in fields.items()}
    try:
        with (temporary/'manifest.jsonl').open('w',encoding='utf-8') as handle, torch.no_grad():
            for i in range(len(dataset)):
                sample = dataset[i]
                sensor = {k:v.unsqueeze(0).to(device) for k,v in sample['sensor'].items()}
                prediction = model(**sensor)
                if not prediction['sensor_valid'].all() or any(not torch.isfinite(prediction[k]).all() for k in ('P_enc_relative_m','r_enc_m','f_enc')):
                    raise ValueError('invalid scientific extraction anchor')
                for name,key in [('P_enc','P_enc_relative_m'),('r_enc','r_enc_m'),('f_enc','f_enc'),('sensor_valid','sensor_valid')]:
                    arrays[name][i] = prediction[key][0].float().cpu().numpy() if name!='sensor_valid' else bool(prediction[key][0])
                arrays['time_s'][i] = float(sample['sensor']['time_s'][-1])
                handle.write(json.dumps(sample['provenance'],allow_nan=False)+'\n')
        for arr in arrays.values():
            arr.flush()
        after = state_dict_hash(model)
        if before!=after:
            raise ValueError('frozen encoder mutated during extraction')
        state_lineage = dict(payload['metadata']['lineage'],encoder_hash=before,encoder_state_hash=before,encoder_checkpoint_hash=file_sha256(checkpoint))
        metadata = {'contract_version':CONTRACT_VERSION,'complete':True,'data_kind':kind,'count':len(dataset),
                    'scientific_eligible':eligible,'feature_dtype':'float32','precision_status':'reference_no_fp16_quantization',
                    'lineage':state_lineage,'max_gap_s':read_json(paths['schema'])['max_gap_s'],
                    'encoder_hash_before':before,'encoder_hash_after':after,'split':split,
                    'arrays':{name:file_sha256(temporary/(name+'.npy')) for name in fields},'manifest_hash':file_sha256(temporary/'manifest.jsonl')}
        atomic_json(temporary/'metadata.json',metadata)
        del arrays
        temporary.rename(output)
        return metadata
    finally:
        dataset.reader.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='eksperimen_model/configs/m4human_encoder.yaml')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--split',choices=('train','val','test'))
    p.add_argument('--device',default='cpu')
    p.add_argument('--allow-debug',action='store_true')
    a = p.parse_args()
    with open(a.config,encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    print(json.dumps(extract(config,a.checkpoint,a.output_dir,a.split,a.device,a.allow_debug),indent=2))

if __name__=='__main__':
    main()
