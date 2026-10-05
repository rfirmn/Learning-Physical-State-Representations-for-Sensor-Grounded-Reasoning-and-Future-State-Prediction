"""Train E with explicit audit gates; validation selection never opens test samples."""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml
from eksperimen_model.datasets.m4human_dataset import M4HumanCausalSensorDataset, EncoderTrainingDataset, OfflineTargets, clean_rpc, read_json
from eksperimen_model.models.m4human_encoder import M4HumanSetEncoderV2, encoder_loss
from eksperimen_model.utils.m4human_runtime import (CONTRACT_VERSION, RunLogger, adamw_parameters, atomic_json, canonical_hash,
    file_sha256, load_checkpoint, resource_snapshot, restore_rng_state, rng_state, save_checkpoint, seed_everything)
from eksperimen_model.utils.m4human_kinematics import DEFAULT_RECIPE


def collate_samples(samples):
    return {'sensor':{k:torch.stack([s['sensor'][k] for s in samples]) for k in samples[0]['sensor']},
            'targets':{k:torch.stack([s['targets'][k] for s in samples]) for k in samples[0]['targets']},
            'provenance':[s['provenance'] for s in samples]}


def fit_input_normalizer(dataset):
    total = 0
    mean = np.zeros(4,dtype=np.float64)
    m2 = np.zeros(4,dtype=np.float64)
    for row in dataset.rows:
        if not row['sensor_frame_valid']:
            continue
        cloud,_ = clean_rpc(dataset.reader.read(row['source_key']),dataset.reader.schema)
        if not len(cloud):
            continue
        values = cloud.astype(np.float64)
        values[:,3] = np.sign(values[:,3])*np.log1p(np.abs(values[:,3]))
        count = len(values)
        local_mean = values.mean(0)
        delta = local_mean-mean
        m2 += ((values-local_mean)**2).sum(0) + delta**2 * total * count / (total+count)
        mean += delta*count/(total+count)
        total += count
    if not total:
        raise ValueError('no valid train returns for normalizer')
    std = np.maximum(np.sqrt(m2/total),np.array([.001,.001,.001,1e-5]))
    return {'mean':mean.tolist(),'std':std.tolist(),'count':total,'scope':'train_only_all_clean_returns',
            'intensity_transform':'signed_log1p','preprocessing_version':'canonical_sha256_v1'}


def make_datasets(config, allow_synthetic=False):
    kind = config.get('data_kind')
    if config.get('contract_version')!=CONTRACT_VERSION or kind not in (('m4human','synthetic') if allow_synthetic else ('m4human',)):
        raise ValueError('real E command requires m4human v3 config; synthetic only in explicit smoke mode')
    paths = config['paths']
    required = ('schema','frame_manifest','target_dir','joint_map','coordinate_audit','audit_report')
    if any(not paths.get(k) for k in required):
        raise ValueError('real-data gates unresolved: provide schema/frame_manifest/targets/joint_map/coordinate/audit_report')
    schema,joint_map,coordinate,audit = map(read_json,(paths['schema'],paths['joint_map'],paths['coordinate_audit'],paths['audit_report']))
    if audit.get('status')!='verified' or audit.get('data_kind') != kind or coordinate.get('audited') is not True or joint_map.get('audited') is not True:
        raise ValueError('audit not verified')
    lineage = {'schema_hash':file_sha256(paths['schema']),'joint_map_hash':file_sha256(paths['joint_map']),
               'coordinate_hash':file_sha256(paths['coordinate_audit']),'frame_manifest_hash':file_sha256(paths['frame_manifest']),
               'source_hash':canonical_hash(audit['source_provenance'])}
    lineage['split_hash'] = lineage['frame_manifest_hash']
    lineage['recipe_hash'] = canonical_hash(DEFAULT_RECIPE)
    for k in ('schema_hash','joint_map_hash','coordinate_hash','frame_manifest_hash'):
        if audit.get(k)!=lineage[k]:
            raise ValueError(f'audit lineage mismatch:{k}')
    targets = OfflineTargets(paths['target_dir'])
    if targets.metadata.get('data_kind') != kind:
        raise ValueError('target data kind mismatch')
    for k in ('joint_map_hash','coordinate_hash','frame_manifest_hash'):
        if targets.metadata['lineage'].get(k)!=lineage[k]:
            raise ValueError(f'target/audit lineage mismatch:{k}')
    lineage['target_hash'] = file_sha256(Path(paths['target_dir'])/'metadata.json')
    train_sensor = M4HumanCausalSensorDataset(paths['dataset_root'],paths['frame_manifest'],schema,'train',True,config['training']['seed'])
    val_sensor = M4HumanCausalSensorDataset(paths['dataset_root'],paths['frame_manifest'],schema,'val',False,config['training']['seed'])
    if not len(train_sensor) or not len(val_sensor):
        raise ValueError('train/val need eligible complete contexts')
    return EncoderTrainingDataset(train_sensor,targets),EncoderTrainingDataset(val_sensor,targets),joint_map,lineage


def to_device(values,device):
    return {k:v.to(device) for k,v in values.items()}


@torch.no_grad()
def evaluate_encoder(model,dataset,device,batch_size=4,records_path=None,panel_count=8):
    model.eval()
    sums = {'relative':0.,'global':0.,'root':0.}
    counts = {'relative':0,'global':0,'root':0}
    joint_sums = np.zeros(22)
    joint_counts = np.zeros(22,dtype=np.int64)
    diagnostics, diagnostic_ids, records = [],[],[]
    handle = open(records_path,'w',encoding='utf-8') if records_path else None
    try:
        for batch in DataLoader(dataset,batch_size=batch_size,shuffle=False,collate_fn=collate_samples,num_workers=0):
            sensor = to_device(batch['sensor'],device)
            pred = model(**sensor)
            target = to_device(batch['targets'],device)
            p,r = pred['P_enc_relative_m'],pred['r_enc_m']
            root_mask = target['annotation_root_valid'] & pred['sensor_valid']
            relative_mask = target['annotation_joint_valid'] & root_mask[:,None]
            relative_mask[:,model.pelvis_index] = False
            global_mask = target['annotation_joint_valid'] & pred['sensor_valid'][:,None]
            errors = {}
            for name,pv,tv,mask in [('relative',p,target['joint_relative_m'],relative_mask),('global',p+r[:,None],target['joint_global_m'],global_mask),('root',r,target['root_m'],root_mask)]:
                if not torch.isfinite(tv[mask]).all():
                    raise ValueError('nonfinite valid evaluation target')
                result = torch.zeros(mask.shape,device=device,dtype=torch.float32)
                result[mask] = (pv.float()[mask]-tv.float()[mask]).norm(dim=-1)
                errors[name] = (result,mask)
                sums[name] += result.sum().item()
                counts[name] += mask.sum().item()
            global_error,gm = errors['global']
            joint_sums += global_error.sum(0).cpu().numpy()
            joint_counts += gm.sum(0).cpu().numpy()
            for i,prov in enumerate(batch['provenance']):
                row = {k:prov[k] for k in ('frame_uid','subject_id','recording_id','split','context_source_frames')}
                row.update(unit='meter',failure=False,metric_reduction='euclidean_per_valid_joint_then_sum',
                           metrics={name:{'error_sum':e[i].sum().item(),'count':int(m[i].sum())} for name,(e,m) in errors.items()},
                           global_per_joint_error=global_error[i].cpu().tolist(),global_per_joint_valid=gm[i].cpu().tolist())
                if handle:
                    handle.write(json.dumps(row,allow_nan=False)+'\n')
                records.append(row) if records_path is None else None
                if len(diagnostics)<panel_count:
                    diagnostics.append({'P_pred':p[i].cpu().numpy(),'r_pred':r[i].cpu().numpy(),
                                        'P_reference':target['joint_relative_m'][i].cpu().numpy(),'r_reference':target['root_m'][i].cpu().numpy(),
                                        'annotation_joint_valid':target['annotation_joint_valid'][i].cpu().numpy(),
                                        'annotation_root_valid':target['annotation_root_valid'][i].cpu().numpy(),
                                        'time_s':sensor['time_s'][i].cpu().numpy()})
                    diagnostic_ids.append(prov)
    finally:
        if handle:
            handle.close()
    metrics = {name:{'error_sum':sums[name],'count':counts[name],'mean_m':sums[name]/counts[name] if counts[name] else None} for name in sums}
    metrics['global_per_joint'] = {'error_sum':joint_sums.tolist(),'count':joint_counts.tolist()}
    return metrics,diagnostics,diagnostic_ids


def recompute_encoder_metrics(path):
    sums,counts = {k:0. for k in ('relative','global','root')},{k:0 for k in ('relative','global','root')}
    with open(path,encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            for key in sums:
                sums[key] += row['metrics'][key]['error_sum']
                counts[key] += row['metrics'][key]['count']
    return {k:{'error_sum':sums[k],'count':counts[k],'mean_m':sums[k]/counts[k] if counts[k] else None} for k in sums}


def train(config,output_dir,device='cpu',resume=None,mode='pilot'):
    train_data,val_data,joint_map,lineage = make_datasets(config,allow_synthetic=mode=='smoke')
    settings = config['training']
    if settings['lambda_bone']!=0 or settings['lambda_root']!=1. or settings['optimizer']!='AdamW' or settings['selection_metric']!='validation_mpjpe_global_m':
        raise ValueError('unsupported scientific override; reference E loss/selection locked')
    if mode=='final' and (settings['root_quality_gate_m'] is None or not config.get('real_smoke_passed') or not config.get('resource_profile_passed')):
        raise ValueError('final needs real smoke/resource evidence and train/val locked root gate')
    if settings['workers']!=0:
        raise ValueError('reference starts workers0; worker spawn gate not closed by this trainer')
    amp = settings['precision']=='fp16_amp_with_grad_scaler'
    if settings['precision'] not in ('fp32','fp16_amp_with_grad_scaler') or (amp and not str(device).startswith('cuda')):
        raise ValueError('requested precision unavailable; resolve explicitly to fp32 for CPU')
    seed_everything(settings['seed'])
    normalizer = fit_input_normalizer(train_data.sensor_dataset)
    config = dict(config,lineage=dict(lineage,normalizer_hash=canonical_hash(normalizer)),normalizer=normalizer)
    logger = RunLogger(output_dir,config,'E')
    model = M4HumanSetEncoderV2(joint_map['pelvis_index'],normalizer).to(device)
    optimizer = torch.optim.AdamW(adamw_parameters(model,settings['weight_decay']),lr=settings['learning_rate'],betas=(.9,.999))
    scaler = torch.amp.GradScaler('cuda',enabled=amp)
    epochs = settings['epochs_max'] if mode=='final' else min(settings['epochs_max'],1 if mode=='smoke' else 5)
    groups_per_epoch = math.ceil(math.ceil(len(train_data)/settings['micro_batch'])/settings['accumulation'])
    total_budget = epochs*groups_per_epoch
    steps,start_epoch,best,patience = 0,0,None,0
    metadata = {'contract_version':CONTRACT_VERSION,'lineage':config['lineage'],'model_id':'m4human_set_v2','pelvis_index':joint_map['pelvis_index'],
                'normalizer':normalizer,'data_kind':config['data_kind'],'config_hash':canonical_hash(config),'selection':'min_val_global_then_root_then_earliest_epoch'}
    try:
        if resume:
            payload = load_checkpoint(resume,model,config['lineage'])
            if payload['metadata']['config_hash']!=metadata['config_hash'] or payload['resume']['total_budget']!=total_budget:
                raise ValueError('resume config/budget changed')
            state = payload['resume']
            optimizer.load_state_dict(state['optimizer'])
            scaler.load_state_dict(state['scaler'])
            restore_rng_state(state['rng'])
            start_epoch,steps,best,patience = state['next_epoch'],state['successful_updates'],state['best'],state['patience']
            logger.log('resume',checkpoint=str(resume),next_epoch=start_epoch,successful_updates=steps)
            best_source = Path(resume).parent/'best.pt'
            if not best_source.exists():
                raise ValueError('resume requires selected best checkpoint beside last')
            import shutil
            shutil.copyfile(best_source,Path(output_dir)/'best.pt')
        panel = Subset(EncoderTrainingDataset(M4HumanCausalSensorDataset(train_data.sensor_dataset.reader.root,train_data.sensor_dataset.rows,
                            train_data.sensor_dataset.reader.schema,'train',False,settings['seed']),train_data.targets),range(min(settings['diagnostic_samples'],len(train_data))))
        for epoch in range(start_epoch,epochs):
            model.train()
            train_data.sensor_dataset.set_epoch(epoch)
            generator = torch.Generator().manual_seed(settings['seed']+epoch)
            loader = DataLoader(train_data,batch_size=settings['micro_batch'],shuffle=True,collate_fn=collate_samples,num_workers=0,generator=generator)
            iterator = iter(loader)
            epoch_sums = {'pose':0.,'root':0.}
            epoch_counts = {'pose':0,'root':0}
            skipped,exposure = 0,0
            started = time.monotonic()
            norm = torch.tensor(0.)
            for _ in range(groups_per_epoch):
                batches = []
                for _ in range(settings['accumulation']):
                    try:
                        batches.append(next(iterator))
                    except StopIteration:
                        break
                if not batches:
                    break
                denominator = {'pose':0,'root':0}
                for batch in batches:
                    mask = batch['targets']['annotation_joint_valid'] & batch['targets']['annotation_root_valid'][:,None]
                    mask = mask.clone()
                    mask[:,model.pelvis_index] = False
                    denominator['pose'] += int(mask.sum())*3
                    denominator['root'] += int(batch['targets']['annotation_root_valid'].sum())*3
                if not sum(denominator.values()):
                    logger.log('skip_no_supervision',epoch=epoch,frame_uids=[p['frame_uid'] for b in batches for p in b['provenance']])
                    skipped += 1
                    continue
                warmup = max(1,math.ceil(total_budget*settings['warmup_optimizer_step_fraction']))
                ratio = (steps+1)/warmup if steps<warmup else .5*(1+math.cos(math.pi*(steps-warmup)/max(1,total_budget-warmup)))
                lr = settings['learning_rate_min'] + (settings['learning_rate']-settings['learning_rate_min'])*ratio
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                for batch in batches:
                    sensor,target = to_device(batch['sensor'],device),to_device(batch['targets'],device)
                    with torch.autocast('cuda',dtype=torch.float16) if amp else nullcontext():
                        pred = model(**sensor)
                    loss = encoder_loss(pred,target,model.pelvis_index,settings['smooth_l1_beta_m'])
                    weighted = sum(loss[k]*(loss[k+'_count']/denominator[k] if denominator[k] else 0) for k in denominator)
                    if not torch.isfinite(weighted):
                        raise ValueError('nonfinite E loss')
                    scaler.scale(weighted).backward()
                    for key in epoch_sums:
                        epoch_sums[key] += float(loss[key].detach())*loss[key+'_count']
                        epoch_counts[key] += loss[key+'_count']
                    exposure += len(batch['provenance'])
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(),settings['clip_grad_norm'],error_if_nonfinite=True)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale()<old_scale:
                    skipped += 1
                    logger.log('overflow_skip',epoch=epoch,successful_updates=steps)
                else:
                    steps += 1
            metrics,_,_ = evaluate_encoder(model,val_data,device,settings['micro_batch'],panel_count=0)
            train_panel_metrics,_,_ = evaluate_encoder(model,panel,device,settings['micro_batch'],panel_count=0)
            if metrics['global']['mean_m'] is None or metrics['root']['mean_m'] is None:
                raise ValueError('validation has no global/root targets')
            key = [metrics['global']['mean_m'],metrics['root']['mean_m'],epoch]
            selected = best is None or tuple(key)<tuple(best)
            if selected:
                best,patience = key,0
                save_checkpoint(Path(output_dir)/'best.pt',model,dict(metadata,selected_epoch=epoch,validation_metrics=metrics))
                logger.log('checkpoint_selection',epoch=epoch,metric=key,checkpoint='best.pt')
            else:
                patience += 1
            logger.log('epoch',epoch=epoch,successful_updates=steps,samples_exposed=exposure,loss={k:epoch_sums[k]/epoch_counts[k] if epoch_counts[k] else None for k in epoch_sums},
                       loss_counts=epoch_counts,loss_reduction='valid_component_weighted_per_accumulation_group',validation=metrics,train_fixed_panel=train_panel_metrics,
                       learning_rate=optimizer.param_groups[0]['lr'],gradient_norm_before_clip=float(norm) if exposure else None,
                       gradient_norm_after_clip=min(float(norm),settings['clip_grad_norm']) if exposure else None,scaler_scale=scaler.get_scale(),skipped_updates=skipped,
                       epoch_seconds=time.monotonic()-started,resources=resource_snapshot())
            save_checkpoint(Path(output_dir)/'last.pt',model,metadata,resume={'next_epoch':epoch+1,'successful_updates':steps,'optimizer':optimizer.state_dict(),
                            'scaler':scaler.state_dict(),'rng':rng_state(),'best':best,'patience':patience,'total_budget':total_budget,'sampler_policy':'epoch_seed_plus_epoch'})
            if patience>=settings['early_stop_patience'] or mode=='smoke':
                break
        load_checkpoint(Path(output_dir)/'best.pt',model,config['lineage'])
        metrics,diagnostics,ids = evaluate_encoder(model,val_data,device,settings['micro_batch'],Path(output_dir)/'predictions_val.jsonl',settings['diagnostic_samples'])
        evaluate_encoder(model,panel,device,settings['micro_batch'],Path(output_dir)/'predictions_train_panel.jsonl',panel_count=0)
        np.savez(Path(output_dir)/'diagnostic_samples.npz',**{k:np.stack([d[k] for d in diagnostics]) for k in diagnostics[0]})
        atomic_json(Path(output_dir)/'diagnostic_samples.json',{'policy':'first_fixed_eligible_validation_anchors_before_training','records':ids,'checkpoint_hash':file_sha256(Path(output_dir)/'best.pt')})
        recomputed = recompute_encoder_metrics(Path(output_dir)/'predictions_val.jsonl')
        for k in recomputed:
            if recomputed[k]['count']!=metrics[k]['count'] or not math.isclose(recomputed[k]['error_sum'],metrics[k]['error_sum'],abs_tol=1e-4,rel_tol=1e-5):
                raise ValueError('metric recomputation failed')
        atomic_json(Path(output_dir)/'metric_recomputation.json',{'status':'verified','metrics':recomputed,'predictions_sha256':file_sha256(Path(output_dir)/'predictions_val.jsonl')})
        root_gate = settings['root_quality_gate_m']
        summary = {'validation':metrics,'successful_updates':steps,'selected_epoch':best[2],
                   'root_gate':{'threshold_m':root_gate,'passed':None if root_gate is None else metrics['root']['mean_m']<=root_gate},
                   'scientific_freeze_eligible':config['data_kind']=='m4human' and root_gate is not None and metrics['root']['mean_m']<=root_gate,
                   'checkpoint_sha256':file_sha256(Path(output_dir)/'best.pt'),'test_status':'not_opened'}
        return logger.finish(summary)
    except Exception as exc:
        logger.fail(exc)
        raise
    finally:
        train_data.sensor_dataset.reader.close()
        val_data.sensor_dataset.reader.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='eksperimen_model/configs/m4human_encoder.yaml')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--mode',choices=('smoke','pilot','final'),default='pilot')
    p.add_argument('--resume')
    p.add_argument('--precision',choices=('fp32','fp16_amp_with_grad_scaler'))
    a = p.parse_args()
    with open(a.config,encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    if a.precision:
        config['training']['precision'] = a.precision
    print(json.dumps(train(config,a.output_dir,a.device,a.resume,a.mode),indent=2))

if __name__=='__main__':
    main()
