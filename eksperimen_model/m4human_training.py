"""Bounded-cache M/C/probe training and extraction, sharing one stage implementation.

Scientific runs require audited lineage. Synthetic fixtures test code only.
"""
import argparse
import copy
import json
import math
import shutil
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import Dataset, DataLoader

from eksperimen_model.models.m4human_motion import CausalDSTformerLiteV1, make_rich_memory, build_rich_attention_memory
from eksperimen_model.models.m4human_tokenizer import KinematicTokenLearnerV1
from eksperimen_model.models.m4human_readout import KinematicReadoutV2, FullHFidelityDecoderV1, TokenKinematicReadoutV1, fidelity_loss, clip_compression_gradients
from eksperimen_model.utils.m4human_runtime import CONTRACT_VERSION, canonical_hash, file_sha256, state_dict_hash, atomic_json, read_json, iter_jsonl, require_lineage, seed_everything, freeze, save_checkpoint, load_checkpoint, RunLogger, adamw_parameters, rng_state, restore_rng_state
from eksperimen_model.utils.m4human_kinematics import build_physical_targets, derivative_support_mask, kinematic_loss, physical_metrics, build_evidence


class WindowTensorCache(Dataset):
    """Loads one exact-window payload at a time; annotation targets are separate."""
    def __init__(self, root, split=None, expected_lineage=None, require_targets=False):
        self.root = Path(root)
        self.require_targets = require_targets
        self.metadata = read_json(self.root / 'metadata.json')
        if self.metadata.get('contract_version') != CONTRACT_VERSION or not self.metadata.get('complete'):
            raise ValueError('foreign or incomplete window cache')
        require_lineage(self.metadata, expected_lineage or {})
        if not self.metadata.get('index_sha256') or file_sha256(self.root / 'index.jsonl') != self.metadata['index_sha256']:
            raise ValueError('window index missing integrity hash or changed')
        all_rows = list(iter_jsonl(self.root / 'index.jsonl'))
        if len(all_rows) != self.metadata.get('count') or len({r['window_id'] for r in all_rows}) != len(all_rows):
            raise ValueError('window cache index count or identity mismatch')
        subject_splits, recordings = {}, {}
        for row in all_rows:
            if row['split'] not in ('train','val','test'):
                raise ValueError('invalid cache partition')
            if subject_splits.setdefault(row['subject_id'],row['split']) != row['split']:
                raise ValueError('cache subject leakage')
            identity = (row['subject_id'],row['split'])
            if recordings.setdefault(row['recording_id'],identity) != identity:
                raise ValueError('cache recording identity leakage')
        self.rows = [r for r in all_rows if split is None or r['split'] == split]
        if not self.rows or len({r['window_id'] for r in self.rows}) != len(self.rows):
            raise ValueError('empty cache partition or duplicated window IDs')
        for row in self.rows:
            if require_targets and not row.get('target_path'):
                raise ValueError('offline target namespace required for training')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = (self.root / row['tensor_path']).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError('cache tensor path escapes artifact root')
        if not row.get('tensor_sha256') or file_sha256(path) != row['tensor_sha256']:
            raise ValueError(f'cache payload changed: {row["window_id"]}')
        tensor = torch.load(path, map_location='cpu', weights_only=True)
        target = {}
        if self.require_targets:
            target_path = (self.root / row['target_path']).resolve()
            if self.root.resolve() not in target_path.parents or file_sha256(target_path) != row.get('target_sha256'):
                raise ValueError('offline target path or checksum changed')
            target = torch.load(target_path, map_location='cpu', weights_only=True)
        return {'sensor': tensor, 'targets': target, 'provenance': row}


def _collate(samples):
    return {'sensor': {k: torch.stack([s['sensor'][k] for s in samples]) for k in samples[0]['sensor'] if isinstance(samples[0]['sensor'][k], torch.Tensor)},
            'targets': {k: torch.stack([s['targets'][k] for s in samples]) for k in samples[0]['targets'] if isinstance(samples[0]['targets'][k], torch.Tensor)},
            'provenance': [s['provenance'] for s in samples]}


def _device_batch(sample, device):
    return {group: {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sample[group].items()} for group in ('sensor', 'targets')}


def _target_from_annotation(sensor, annotation, max_gap_s, common=False):
    if 'v_relative_mps' in annotation:
        target = {k: v.clone() for k, v in annotation.items()}
    else:
        target = build_physical_targets(annotation['joint_relative_m'].cpu().numpy(), annotation['root_m'].cpu().numpy(),
                                        sensor['time_s'].cpu().numpy(), annotation['annotation_position_valid_joint'].cpu().numpy(),
                                        annotation['annotation_position_valid_root'].cpu().numpy(), max_gap_s=max_gap_s)
        target = {k: torch.as_tensor(v) for k, v in target.items() if isinstance(v, np.ndarray)}
    if common:
        frame = sensor['common_frame_valid']
        joint_valid, root_valid = frame[:, None].expand(-1, 22), frame
    else:
        joint_valid = sensor['joint_sensor_valid'] & sensor['context_time_valid'][:, None]
        root_valid = sensor['root_sensor_valid'] & sensor['feature_valid'] & sensor['context_time_valid']
    target['position_valid_joint'] &= joint_valid
    target['position_valid_root'] &= root_valid
    target['velocity_valid_joint'] &= derivative_support_mask(joint_valid, time_dim=0)
    target['velocity_valid_root'] &= derivative_support_mask(root_valid, time_dim=0)
    return target


class _PhysicalWindows(Dataset):
    def __init__(self, source, max_gap_s, common=False):
        self.source, self.max_gap_s, self.common = source, max_gap_s, common
    def __len__(self):
        return len(self.source)
    def __getitem__(self, index):
        sample = self.source[index]
        sample['targets'] = _target_from_annotation(sample['sensor'], sample['targets'], self.max_gap_s, self.common)
        return sample


class _TokenWindows(Dataset):
    def __init__(self, source, token_cache, expected_lineage, condition):
        self.source = source
        self.tokens = WindowTensorCache(token_cache, expected_lineage=expected_lineage)
        expected = {'U_base':'C_base', 'U_kin':'C_kin'}[condition]
        if self.tokens.metadata.get('condition') != expected or self.tokens.metadata.get('representation') != 'exact_U':
            raise ValueError('foreign probe token condition or non-final representation')
        self.lookup = {row['window_id']:i for i,row in enumerate(self.tokens.rows)}
    def __len__(self):
        return len(self.source)
    def __getitem__(self, index):
        sample = self.source[index]
        token = self.tokens[self.lookup[sample['provenance']['window_id']]]
        if token['provenance']['window_support_hash'] != sample['provenance']['window_support_hash']:
            raise ValueError('probe source/target exact-window support mismatch')
        if not torch.equal(token['sensor']['time_s'],sample['sensor']['time_s']):
            raise ValueError('probe source timestamp mismatch')
        sample['sensor'].update(token['sensor'])
        return sample


def _datasets(config, stage):
    if stage == 'M':
        from eksperimen_model.datasets.m4human_state_dataset import EncodedStateWindowDataset
        train = EncodedStateWindowDataset(config['source_cache'], length=32, stride=config.get('train_stride', 8), split='train', targets_dir=config['target_cache'], allow_debug=config['data_kind']=='synthetic')
        val = EncodedStateWindowDataset(config['source_cache'], length=32, stride=config.get('val_stride', 16), split='val', targets_dir=config['target_cache'], allow_debug=config['data_kind']=='synthetic')
    else:
        train = WindowTensorCache(config['source_cache'], 'train', config['lineage'], require_targets=True)
        val = WindowTensorCache(config['source_cache'], 'val', config['lineage'], require_targets=True)
    return (_PhysicalWindows(train, config['max_gap_s'], stage != 'M'), _PhysicalWindows(val, config['max_gap_s'], stage != 'M'))


def _fit_statistics(dataset, stage, pelvis_index):
    """Train-only fp64 population moments and component RMS, never test support."""
    sums = {}; squares = {}; counts = {}
    for sample in dataset:
        s, target = sample['sensor'], sample['targets']
        fields = {}
        if stage == 'M':
            fields['root'] = (s['r_enc_m'], s['root_sensor_valid'] & s['feature_valid'] & s['context_time_valid'])
        else:
            fields['H'] = (s['H'], s['latent_valid'] & s['common_frame_valid'][:, None])
        joint_mask = target['position_valid_joint'].clone(); joint_mask[:, pelvis_index] = False
        velocity_mask = target['velocity_valid_joint'].clone(); velocity_mask[:, pelvis_index] = False
        fields['s_v_joint'] = (target['v_relative_mps'], velocity_mask)
        fields['s_v_root'] = (target['v_root_mps'], target['velocity_valid_root'])
        if stage == 'M':
            fields['s_delta_joint'] = (target['P_relative_m'] - s['P_enc_relative_m'], joint_mask)
            fields['s_delta_root'] = (target['r_m'] - s['r_enc_m'], target['position_valid_root'])
        for name, (value, mask) in fields.items():
            values = value[mask].double()
            if not torch.isfinite(values).all():
                raise ValueError(f'non-finite training statistic: {name}')
            if values.numel() == 0:
                continue
            sums[name] = sums.get(name, 0.) + values.sum(0)
            squares[name] = squares.get(name, 0.) + values.square().sum(0)
            counts[name] = counts.get(name, 0) + values.shape[0]
    required = ['s_v_joint', 's_v_root'] + (['root', 's_delta_joint', 's_delta_root'] if stage == 'M' else ['H'])
    if any(not counts.get(name) for name in required):
        raise ValueError('train support empty for required normalizer')
    stats = {'counts': counts, 'variance': 'population_ddof0', 'train_only': True}
    for name in required:
        if name in ('H', 'root'):
            mean = sums[name] / counts[name]
            std = (squares[name] / counts[name] - mean.square()).clamp_min(0).sqrt().clamp_min(1e-3)
            stats[name + '_mean'], stats[name + '_std'] = mean.float().tolist(), std.float().tolist()
        else:
            floor = 0.01 if name.startswith('s_delta') else 0.001
            stats[name] = max(float((squares[name].sum() / (counts[name] * 3)).sqrt()), floor)
    stats['hash'] = canonical_hash(stats)
    return stats


def _scientific_gate(config, metadata):
    if config.get('contract_version') != CONTRACT_VERSION:
        raise ValueError('contract mismatch')
    if config.get('precision') not in ('fp32', 'fp16_amp'):
        raise ValueError('precision must be fp32 or gated fp16_amp')
    if config.get('data_kind') == metadata.get('data_kind') == 'synthetic':
        if config.get('scientific_gates_passed'):
            raise ValueError('synthetic development cannot close scientific gates')
        require_lineage(metadata,config['lineage'])
        return
    if config.get('scientific_gates_passed') is not True:
        raise ValueError('real-data audit/smoke/tiny-overfit gates must pass before training')
    lineage = config.get('lineage', {})
    required = ('schema_hash', 'coordinate_hash', 'joint_map_hash', 'split_hash', 'recipe_hash', 'encoder_hash')
    if any(not isinstance(lineage.get(k), str) or not lineage[k] for k in required):
        raise ValueError('unresolved mandatory audit lineage')
    if metadata.get('data_kind') != 'm4human' or not metadata.get('complete'):
        raise ValueError('scientific training requires complete real-data cache')
    if metadata.get('scientific_eligible') is not True:
        raise ValueError('upstream cache failed scientific eligibility')
    require_lineage(metadata, {k: lineage[k] for k in required})
    if config.get('precision') not in ('fp32', 'fp16_amp'):
        raise ValueError('precision must be fp32 or gated fp16_amp')


def _scheduler(optimizer, total):
    warmup = max(1, math.ceil(total * 0.05))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (step + 1) / warmup if step < warmup else
                                            0.5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, total - warmup)))))


def _models(stage, stats, config):
    if stage == 'M':
        return nn.ModuleDict({'motion': CausalDSTformerLiteV1(stats['root_mean'], stats['root_std']),
                              'readout': KinematicReadoutV2(stats, config['pelvis_index'])})
    if stage == 'C':
        return nn.ModuleDict({'compressor': KinematicTokenLearnerV1(), 'fidelity': FullHFidelityDecoderV1(),
                              'auxiliary': TokenKinematicReadoutV1(stats, config['pelvis_index'])})
    return nn.ModuleDict({'probe': TokenKinematicReadoutV1(stats, config['pelvis_index'])})


def _forward(model, stage, condition, sensor, target, stats, config):
    if stage == 'M':
        out = model['motion'](sensor, sensor, sensor['time_s'])
        prediction = model['readout'](out['H'], sensor['P_enc_relative_m'], sensor['r_enc_m'], out['latent_valid'])
        terms = kinematic_loss(prediction, target, stats, config['pelvis_index'])
        return terms['loss'], terms, prediction
    valid = sensor['latent_valid'] & sensor['common_frame_valid'][..., None]
    if stage == 'C' or condition == 'M':
        memory = build_rich_attention_memory(make_rich_memory(sensor['H'].detach()), sensor['time_s'], valid)
        memory_valid = valid.flatten(1, 2)
    if stage == 'C':
        tokens = model['compressor'].from_attention_memory(memory, sensor['time_s'], valid, config['K'])
        h_hat = model['fidelity'](tokens['U'], tokens['token_mask'], sensor['time_s'], config.get('query_chunk', 0))
        fidelity = fidelity_loss(h_hat, sensor['H'], valid, torch.tensor(stats['H_mean'], device=h_hat.device), torch.tensor(stats['H_std'], device=h_hat.device))
        source, source_valid = (memory.flatten(1, 2), memory_valid) if condition == 'C_base' else (tokens['U'], tokens['token_mask'])
        prediction = model['auxiliary'](source, source_valid, sensor['time_s'], config.get('query_chunk', 0))
        terms = kinematic_loss(prediction, target, stats, config['pelvis_index'])
        terms.update(L_fidelity=fidelity['loss'], fidelity_count=fidelity['count'], fidelity_sum=fidelity['sum'])
        return fidelity['loss'] + config['lambda_aux'] * terms['loss'], terms, prediction
    source, source_valid = (memory.flatten(1, 2), memory_valid) if condition == 'M' else (sensor['U'], sensor['token_mask'])
    prediction = model['probe'](source, source_valid, sensor['time_s'], config.get('query_chunk', 0))
    terms = kinematic_loss(prediction, target, stats, config['pelvis_index'])
    return terms['loss'], terms, prediction


def _numbers(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: _numbers(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _validate(model, stage, condition, loader, device, stats, config):
    model.eval(); records = []; numerator = {}; denominator = {}
    with torch.no_grad():
        for batch in loader:
            values = _device_batch(batch, device)
            _, terms, prediction = _forward(model, stage, condition, values['sensor'], values['targets'], stats, config)
            for name in ('L_p_joint', 'L_p_root', 'L_v_joint', 'L_v_root', 'L_fidelity'):
                if name not in terms:
                    continue
                count = terms['fidelity_count'] if name == 'L_fidelity' else terms['counts'][name[2:]]
                numerator[name] = numerator.get(name, 0.) + float(terms[name]) * int(count)
                denominator[name] = denominator.get(name, 0) + int(count)
            for i, row in enumerate(batch['provenance']):
                metrics = physical_metrics({k:v[i] for k,v in prediction.items()}, {k:v[i] for k,v in values['targets'].items()}, pelvis_index=config['pelvis_index'])
                pred = {k:v[i].detach().cpu().numpy() for k,v in prediction.items()}
                reference = {k:v[i].detach().cpu().numpy() for k,v in values['targets'].items()}
                times = values['sensor']['time_s'][i].cpu().numpy()
                tasks = []
                sensor = {k:v[i] for k,v in values['sensor'].items()}
                common = sensor.get('common_frame_valid')
                joint_mask = sensor['joint_sensor_valid'] & sensor['context_time_valid'][..., None] if common is None else common[:,None].expand(-1,22)
                root_mask = sensor['root_sensor_valid'] & sensor['feature_valid'] & sensor['context_time_valid'] if common is None else common
                sensor_support = {'velocity_valid_joint': derivative_support_mask(joint_mask,time_dim=0).cpu().numpy(),
                                  'velocity_valid_root': derivative_support_mask(root_mask,time_dim=0).cpu().numpy()}
                for task,body in [('root_speed_trend',None),('relative_limb_motion','arms'),('relative_limb_motion','legs')]:
                    if body is not None and not config.get('joint_map'):
                        continue
                    kwargs = dict(recipe=config.get('recipe'),task=task,body_part=body,joint_map=config.get('joint_map'))
                    tasks.append({'task':task,'body_part':body,'prediction':build_evidence(pred,times,sensor_support,**kwargs),
                                  'reference':build_evidence(reference,times,reference,source='gt_reference',**kwargs)})
                records.append({'window_id':row['window_id'], 'subject_id':row['subject_id'], 'recording_id':row['recording_id'], 'condition':condition, 'metrics':_numbers(metrics),'tasks':tasks,'sensor_support_hash':row.get('window_support_hash',canonical_hash(row))})
    means = {k: numerator[k] / denominator[k] if denominator[k] else None for k in numerator}
    if stage == 'C':
        selection = means['L_fidelity']
    else:
        if not sum(denominator.values()):
            raise ValueError('validation has no eligible physical objective')
        selection = sum((means.get(k) or 0.) * weight for k,weight in [('L_p_joint',1.),('L_p_root',1.),('L_v_joint',.25),('L_v_root',.25)])
    if selection is None or not math.isfinite(selection):
        raise ValueError('invalid validation selection objective')
    return {'selection':selection, 'losses':means, 'numerators':numerator, 'counts':denominator}, records


def _logical_batches(loader, accumulation):
    batches = []
    for batch in loader:
        batches.append(batch)
        if len(batches) == accumulation:
            yield batches
            batches = []
    if batches:
        yield batches


def _component_counts(batch, stage, pelvis):
    target = batch['targets']
    counts = {}
    for key,mask_key in [('p_joint','position_valid_joint'),('p_root','position_valid_root'),('v_joint','velocity_valid_joint'),('v_root','velocity_valid_root')]:
        mask = target[mask_key].clone()
        if key.endswith('joint'):
            mask[...,pelvis] = False
        counts[key] = int(mask.sum()) * 3
    if stage == 'C':
        s = batch['sensor']
        counts['fidelity'] = int((s['latent_valid'] & s['common_frame_valid'][...,None]).sum()) * 128
    return counts


def _diagnostic_panel(model,stage,condition,datasets,device,stats,config,output):
    arrays = {}; ids = []; model.eval()
    with torch.no_grad():
        for partition,dataset in zip(('train','val'),datasets):
            count = min(len(dataset),config.get('diagnostic_windows',2))
            samples = [dataset[i] for i in range(count)]
            batch = _collate(samples); values = _device_batch(batch,device)
            _,_,prediction = _forward(model,stage,condition,values['sensor'],values['targets'],stats,config)
            for key,value in prediction.items():
                arrays[f'{partition}_pred_{key}'] = value.cpu().numpy()
            for key,value in values['targets'].items():
                arrays[f'{partition}_ref_{key}'] = value.cpu().numpy()
            arrays[f'{partition}_time_s'] = values['sensor']['time_s'].cpu().numpy()
            ids.extend({'split':partition,'window_id':s['provenance']['window_id']} for s in samples)
    np.savez(output/'diagnostic_samples.npz',**arrays)
    atomic_json(output/'diagnostic_samples.json',{'policy':'first_manifest_windows_pinned_before_training','records':ids,'condition':condition,'checkpoint':'best.pt','checkpoint_hash':file_sha256(output/'best.pt')},overwrite=True)


def _recompute_metrics(records,output):
    aggregate = {}
    for row in records:
        for name,entry in row['metrics'].items():
            if not isinstance(entry,dict) or 'sum' not in entry:
                continue
            item = aggregate.setdefault(name,{'sum':0.,'count':0,'unit':entry['unit']})
            item['sum'] += entry['sum']; item['count'] += entry['count']
    for item in aggregate.values():
        item['mean'] = item['sum']/item['count'] if item['count'] else None
    atomic_json(output/'metric_recomputation.json',{'predictions_sha256':file_sha256(output/'predictions_val.jsonl'),'metrics':aggregate,'status':'verified','kind':'physical'},overwrite=True)
    return aggregate


def train(config_path, stage, condition=None, resume=None):
    config = yaml.safe_load(Path(config_path).read_text())
    metadata = read_json(Path(config['source_cache']) / 'metadata.json')
    _scientific_gate(config, metadata)
    if stage == 'C' and condition not in ('C_base','C_kin'):
        raise ValueError('C condition must be explicit')
    if stage == 'probe' and condition not in ('M','U_base','U_kin'):
        raise ValueError('fresh probe source must be explicit')
    seed_everything(config['seed'])
    train_set, val_set = _datasets(config, stage)
    if stage == 'probe' and condition != 'M':
        train_set = _TokenWindows(train_set,config['token_cache'],config['lineage'],condition)
        val_set = _TokenWindows(val_set,config['token_cache'],config['lineage'],condition)
        if any(dataset.tokens.metadata.get('K') != config['K'] for dataset in (train_set,val_set)):
            raise ValueError('probe token budget differs from locked config')
    stats = _fit_statistics(train_set, stage, config['pelvis_index'])
    if stage != 'M':
        physical_stats = read_json(config['normalizer_path'])
        if physical_stats.get('hash') != canonical_hash({k:v for k,v in physical_stats.items() if k!='hash'}):
            raise ValueError('physical normalizer content/hash mismatch')
        if metadata['lineage'].get('normalizer_hash') != physical_stats['hash']:
            raise ValueError('foreign physical scales for common motion cache')
        for key in ('s_v_joint', 's_v_root'):
            stats[key] = physical_stats[key]
        stats['physical_normalizer_hash'] = physical_stats['hash']
        stats['hash'] = canonical_hash({k:v for k,v in stats.items() if k != 'hash'})
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    amp = config['precision'] == 'fp16_amp'
    if amp and (device.type != 'cuda' or not config.get('precision_gate_passed')):
        raise ValueError('fp16 AMP requires CUDA and passed numeric gate')
    model = _models(stage, stats, config).to(device)
    initialization_hash = state_dict_hash(model)
    if stage in ('C', 'probe'):
        paired = {k:v for k,v in config.items() if k not in ('output_dir','parent_run_id','token_cache')}
        paired.update(initialization_hash=initialization_hash,normalizer_hash=stats['hash'],sample_order='seeded_epoch_permutation',
                      selection='minimum_validation_full_H_fidelity_earliest' if stage == 'C' else 'minimum_physical_objective_earliest',
                      source_cache_hash=canonical_hash(metadata),
                      train_cohort_hash=canonical_hash([train_set[i]['provenance']['window_id'] for i in range(len(train_set))]),
                      validation_cohort_hash=canonical_hash([val_set[i]['provenance']['window_id'] for i in range(len(val_set))]))
        paired_path = Path(config['paired_contract_path'])
        if paired_path.exists():
            if read_json(paired_path) != paired:
                raise ValueError('paired initialization, support, or training budget mismatch')
        else:
            atomic_json(paired_path,paired)
    optimizer = torch.optim.AdamW(adamw_parameters(model), lr=config['learning_rate'])
    total = int(config['successful_updates']); accumulation = int(config['accumulation'])
    if total <= 0 or accumulation <= 0:
        raise ValueError('positive update and accumulation budgets required')
    scheduler = _scheduler(optimizer, total)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    output = Path(config['output_dir']) / (condition or 'M')
    logger = RunLogger(output, config, stage)
    lineage = {**config['lineage'], 'source_cache_hash': canonical_hash(metadata), 'stage_normalizer_hash':stats['hash'], 'initialization_hash':initialization_hash, 'condition':condition or 'M', 'K':config.get('K'),
               'training_config_hash':canonical_hash({k:v for k,v in config.items() if k not in ('output_dir','parent_run_id')})}
    lineage = {k:v for k,v in lineage.items() if v is not None}
    if stage == 'probe' and condition != 'M':
        lineage['token_cache_hash'] = canonical_hash(read_json(Path(config['token_cache'])/'metadata.json'))
    atomic_json(output / 'normalizers.json', stats)
    logged_lineage = read_json(output / 'lineage.json')
    logged_lineage['lineage'] = lineage
    atomic_json(output / 'lineage.json', logged_lineage, overwrite=True)
    successful = 0; skipped = 0; epoch = 0; best = float('inf')
    generator = torch.Generator().manual_seed(config['seed'])
    if resume:
        saved = load_checkpoint(resume, model, lineage)
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler']); scaler.load_state_dict(saved['scaler'])
        successful, skipped, epoch, best = (saved[k] for k in ('successful_updates','skipped_updates','epoch','best'))
        if not saved.get('epoch_complete') and successful < total:
            raise ValueError('Only epoch-boundary resume is supported')
        generator.set_state(saved['sampler_rng']); restore_rng_state(saved['rng_state'])
        for name in ('best.pt','last.pt','predictions_val.jsonl','diagnostic_samples.npz','diagnostic_samples.json','metric_recomputation.json'):
            shutil.copyfile(Path(resume).parent/name, output/name)
        logger.log('history',parent_checkpoint=str(resume),epoch=epoch,successful_updates=successful,event_detail='resume')
    train_loader = DataLoader(train_set, batch_size=config['micro_batch'], shuffle=True, generator=generator, num_workers=0, collate_fn=_collate)
    val_loader = DataLoader(val_set, batch_size=config['micro_batch'], shuffle=False, num_workers=0, collate_fn=_collate)
    start = time.monotonic()
    epoch_complete = True
    def checkpoint(name):
        save_checkpoint(output / name, model, lineage, stats=stats, stage=stage, condition=condition, K=config.get('K'), config=config,
                        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                        successful_updates=successful, skipped_updates=skipped, epoch=epoch, best=best,
                        sampler_rng=generator.get_state(),rng_state=rng_state(),epoch_complete=epoch_complete)
    try:
        while successful < total:
            model.train(); optimizer.zero_grad(set_to_none=True); pending = 0; epoch += 1
            epoch_start_updates = successful
            logical_count = math.ceil(len(train_loader)/accumulation)
            for logical_index, logical in enumerate(_logical_batches(train_loader,accumulation)):
                if successful >= total:
                    break
                per_micro = [_component_counts(batch,stage,config['pelvis_index']) for batch in logical]
                totals = {name:sum(c[name] for c in per_micro) for name in per_micro[0]}
                if not sum(totals.values()):
                    logger.log('empty_objective',epoch=epoch,source_ids=[r['window_id'] for batch in logical for r in batch['provenance']])
                    continue
                objective_value = 0.; group_numerators = {name:0. for name in totals}
                for batch,counts in zip(logical,per_micro):
                    values = _device_batch(batch,device)
                    with torch.autocast(device.type,dtype=torch.float16,enabled=amp):
                        _,terms,_ = _forward(model,stage,condition,values['sensor'],values['targets'],stats,config)
                    physical = sum(terms['L_'+name] * counts[name] / max(1,totals[name]) * weight
                                   for name,weight in [('p_joint',1.),('p_root',1.),('v_joint',.25),('v_root',.25)])
                    loss = physical if stage!='C' else config['lambda_aux']*physical + terms['L_fidelity']*counts['fidelity']/max(1,totals['fidelity'])
                    if not torch.isfinite(loss):
                        raise ValueError('non-finite objective; scientific run halted')
                    scaler.scale(loss).backward(); objective_value += float(loss.detach())
                    for name in totals:
                        group_numerators[name] += float(terms['L_fidelity' if name=='fidelity' else 'L_'+name].detach())*counts[name]
                scaler.unscale_(optimizer)
                norms = clip_compression_gradients(model['compressor'],model['fidelity'],model['auxiliary']) if stage=='C' else {'model':float(nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))}
                old_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
                if scaler.get_scale() < old_scale:
                    skipped += 1
                    raise ValueError('AMP overflow breaks fixed successful exposure; rerun pair after numeric gate')
                successful += 1; scheduler.step(); optimizer.zero_grad(set_to_none=True)
                epoch_complete = logical_index + 1 == logical_count
                logger.log('update',epoch=epoch,update=successful,skipped_updates=skipped,loss=objective_value,
                           terms={name:group_numerators[name]/totals[name] if totals[name] else None for name in totals},
                           numerators=group_numerators,counts=totals,reduction='component_weighted_logical_batch',
                           gradient_norm_before=norms,gradient_norm_after={k:min(v,1.) for k,v in norms.items()},
                           learning_rate=scheduler.get_last_lr(),source_ids=[r['window_id'] for batch in logical for r in batch['provenance']],
                           exposed_windows=sum(len(batch['provenance']) for batch in logical),elapsed_s=time.monotonic()-start)
            if successful == epoch_start_updates:
                raise ValueError('entire training epoch has no eligible objective')
            result, records = _validate(model,stage,condition,val_loader,device,stats,config)
            logger.log('validation',epoch=epoch,update=successful,**result)
            train_panel_loader = DataLoader(torch.utils.data.Subset(train_set,range(min(len(train_set),config.get('diagnostic_windows',2)))),batch_size=config['micro_batch'],collate_fn=_collate)
            train_result,_ = _validate(model,stage,condition,train_panel_loader,device,stats,config)
            logger.log('validation',partition='fixed_train_diagnostic',epoch=epoch,update=successful,**train_result)
            if result['selection'] < best:
                best = result['selection']; checkpoint('best.pt')
                with (output/'predictions_val.jsonl').open('w') as stream:
                    for row in records:
                        row.update(split='val',checkpoint_hash=file_sha256(output/'best.pt'),source_hash=canonical_hash(metadata),window_support_hash=row.pop('sensor_support_hash'))
                        stream.write(json.dumps(row,allow_nan=False)+'\n')
                _diagnostic_panel(model,stage,condition,(train_set,val_set),device,stats,config,output)
                _recompute_metrics(records,output)
                logger.log('checkpoint_selection',metric='minimum_normalized_full_H_fidelity' if stage=='C' else 'physical_validation_objective',tie_break='earliest',value=best,checkpoint='best.pt')
            checkpoint('last.pt')
        return logger.finish({'successful_updates':successful,'skipped_updates':skipped,'best_validation':best,
                       'validation_physical':read_json(output/'metric_recomputation.json')['metrics'],
                       'data_kind':config['data_kind'],'initialization_hash':initialization_hash,'condition':condition,'K':config.get('K')})
    except Exception as exc:
        logger.fail(exc); raise


def train_cli(stage):
    parser = argparse.ArgumentParser(description=f'M4Human {stage} training; audit-gated real data only')
    parser.add_argument('--config', required=True)
    if stage != 'M':
        parser.add_argument('--condition', required=True, choices=('C_base','C_kin') if stage=='C' else ('M','U_base','U_kin'))
    parser.add_argument('--resume')
    args = parser.parse_args()
    train(args.config, stage, getattr(args,'condition',None), args.resume)


def extract_motion(config_path, checkpoint_path, output_dir, split=None):
    config = yaml.safe_load(Path(config_path).read_text())
    metadata = read_json(Path(config['source_cache'])/'metadata.json')
    _scientific_gate(config, metadata)
    payload = torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if payload.get('stage') != 'M':
        raise ValueError('motion extraction requires a selected common M checkpoint')
    model = _models('M',payload['stats'],config)
    load_checkpoint(checkpoint_path,model,config['lineage']); freeze(model)
    from eksperimen_model.datasets.m4human_state_dataset import EncodedStateWindowDataset
    source = EncodedStateWindowDataset(config['source_cache'],length=32,stride=config.get('extraction_stride',8),split=split,targets_dir=config.get('target_cache'),allow_debug=config['data_kind']=='synthetic')
    lineage = {**config['lineage'], 'motion_hash':state_dict_hash(model['motion']), 'motion_checkpoint_hash':file_sha256(checkpoint_path),
               'physical_readout_hash':state_dict_hash(model['readout']), 'state_cache_hash':canonical_hash(metadata),
               'normalizer_hash':payload['stats']['hash'], 'window_policy':'exact32_local_reset', 'time_encoding':'actual_relative_nominal_dt_1_over_12'}
    root = Path(output_dir); root.mkdir(parents=True,exist_ok=False)
    (root/'tensors').mkdir(); (root/'targets').mkdir()
    atomic_json(root/'metadata.json',{'contract_version':CONTRACT_VERSION,'complete':False,'data_kind':metadata['data_kind'],'lineage':lineage,'representation':'full_H'})
    count = 0
    with (root/'index.jsonl').open('w') as index:
        for sample in source:
            sensor = sample['sensor']
            with torch.no_grad():
                result = model['motion']({k:v.unsqueeze(0) for k,v in sensor.items()}, {k:v.unsqueeze(0) for k,v in sensor.items()}, sensor['time_s'].unsqueeze(0))
            cached = {k:v.detach().cpu().clone() for k,v in sensor.items()}
            cached.update({k:v[0].detach().cpu().clone() for k,v in result.items()})
            path = root/'tensors'/f'{count:08d}.pt'; torch.save(cached,path)
            row = {**sample['provenance'],'tensor_path':str(path.relative_to(root)),'tensor_sha256':file_sha256(path),
                   'window_support_hash':canonical_hash(sample['provenance']),'motion_hash':lineage['motion_hash']}
            if sample['targets']:
                target = _target_from_annotation(cached,sample['targets'],config['max_gap_s'],common=True)
                target_path = root/'targets'/f'{count:08d}.pt'; torch.save(target,target_path)
                row.update(target_path=str(target_path.relative_to(root)),target_sha256=file_sha256(target_path))
            index.write(json.dumps(row,allow_nan=False)+'\n'); index.flush(); count += 1
    if not count:
        raise ValueError('no eligible exact windows for motion extraction')
    atomic_json(root/'metadata.json',{'contract_version':CONTRACT_VERSION,'complete':True,'data_kind':metadata['data_kind'],'scientific_eligible':metadata.get('scientific_eligible',False),'lineage':lineage,'representation':'full_H',
                                    'count':count,'index_sha256':file_sha256(root/'index.jsonl')},overwrite=True)


def extract_tokens(config_path, checkpoint_path, output_dir, split=None):
    config = yaml.safe_load(Path(config_path).read_text())
    source = WindowTensorCache(config['source_cache'],split,config['lineage'])
    _scientific_gate(config,source.metadata)
    payload = torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if payload.get('stage') != 'C' or payload.get('condition') not in ('C_base','C_kin') or payload.get('K') != config['K']:
        raise ValueError('foreign stage/condition/budget for exact U extraction')
    model = _models('C',payload['stats'],config)
    load_checkpoint(checkpoint_path,model,config['lineage']); freeze(model)
    tokenizer_hash = state_dict_hash(model['compressor'])
    lineage = {**source.metadata['lineage'],'tokenizer_hash':tokenizer_hash,'compressor_checkpoint_hash':file_sha256(checkpoint_path),
               'h_cache_hash':canonical_hash(source.metadata),'bin_policy':'floor_k32_K','time_encoding':'actual_relative_nominal_dt_1_over_12'}
    root = Path(output_dir); root.mkdir(parents=True,exist_ok=False); (root/'tensors').mkdir()
    info = {'contract_version':CONTRACT_VERSION,'complete':False,'data_kind':source.metadata['data_kind'],'lineage':lineage,
            'representation':'exact_U','condition':payload['condition'],'K':config['K'],'scientific_eligible':source.metadata.get('scientific_eligible',False)}
    atomic_json(root/'metadata.json',info)
    with (root/'index.jsonl').open('w') as index:
        for i in range(len(source)):
            sample = source[i]; s = sample['sensor']; valid = s['latent_valid'] & s['common_frame_valid'][:,None]
            with torch.no_grad():
                out = model['compressor'](make_rich_memory(s['H'].unsqueeze(0)),s['time_s'].unsqueeze(0),valid.unsqueeze(0),config['K'])
            tensor = {k:v[0].detach().cpu().clone() if v.ndim>1 else v.detach().cpu().clone() for k,v in out.items()}
            tensor['time_s'] = s['time_s'].clone()
            path = root/'tensors'/f'{i:08d}.pt'; torch.save(tensor,path)
            row = {k:v for k,v in sample['provenance'].items() if k not in ('tensor_path','tensor_sha256','target_path','target_sha256')}
            row.update(tensor_path=str(path.relative_to(root)),tensor_sha256=file_sha256(path),tokenizer_hash=tokenizer_hash,
                       condition=payload['condition'],K=config['K'],h_cache_hash=lineage['h_cache_hash'])
            index.write(json.dumps(row,allow_nan=False)+'\n'); index.flush()
    info.update(complete=True,count=len(source),index_sha256=file_sha256(root/'index.jsonl'))
    atomic_json(root/'metadata.json',info,overwrite=True)


def extract_cli(stage):
    parser = argparse.ArgumentParser(description=f'Exact-window frozen M4Human {stage} extraction')
    parser.add_argument('--config',required=True); parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True); parser.add_argument('--split',choices=('train','val','test'))
    args = parser.parse_args()
    (extract_motion if stage=='H' else extract_tokens)(args.config,args.checkpoint,args.output,args.split)
