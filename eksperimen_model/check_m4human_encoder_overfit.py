"""Train-only fixed-sample E diagnostic; never exports a scientific checkpoint."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import yaml
from eksperimen_model.train_m4human_encoder import make_datasets, collate_samples, to_device, fit_input_normalizer
from eksperimen_model.models.m4human_encoder import M4HumanSetEncoderV2, encoder_loss
from eksperimen_model.profile_m4human_encoder import numerical_parity
from eksperimen_model.utils.m4human_performance import configure_runtime, finish_optimizer_step
from eksperimen_model.utils.m4human_runtime import atomic_json, canonical_hash, seed_everything, rng_state, restore_rng_state


def overfit(dataset, pelvis_index, normalizer, settings, device, samples=16, updates=200, loss_ratio_max=.25):
    if type(samples) is not int or not 1 <= samples <= min(128, len(dataset)):
        raise ValueError('samples must be 1..128 and available in train')
    if type(updates) is not int or updates < 1 or not 0 < loss_ratio_max < 1:
        raise ValueError('positive update budget and loss ratio in (0,1) required')
    configure_runtime(settings, device)
    seed_everything(settings.get('seed', 42))
    # Read once: identical radar sampling and targets at every diagnostic update.
    selected = [dataset[i] for i in range(samples)]
    full = collate_samples(selected)
    batches = [collate_samples(selected[i:i + settings['micro_batch']])
               for i in range(0, samples, settings['micro_batch'])]
    model = M4HumanSetEncoderV2(pelvis_index, normalizer).to(device)
    amp = settings.get('precision') == 'fp16_amp_with_grad_scaler'
    parity = numerical_parity(model, full, device, amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.get('learning_rate', .0002),
                                  weight_decay=settings.get('weight_decay', .01))
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    totals = {'pose': 0, 'root': 0}
    for batch in batches:
        sensor = to_device(batch['sensor'], device)
        with torch.no_grad():
            terms = encoder_loss(model(**sensor), to_device(batch['targets'], device), pelvis_index)
        for key in totals:
            totals[key] += terms[key + '_count']
    if not sum(totals.values()):
        raise ValueError('selected train samples have no sensor-supported supervision')

    @torch.no_grad()
    def measure():
        model.eval()
        loss = 0.
        for batch in batches:
            terms = encoder_loss(model(**to_device(batch['sensor'], device)),
                                 to_device(batch['targets'], device), pelvis_index,
                                 settings.get('smooth_l1_beta_m', .05))
            loss += sum(float(terms[key]) * terms[key + '_count'] / max(1, totals[key]) for key in totals)
        return loss

    initial = measure()
    history, overflow_retries = [], 0
    retries = settings.get('amp_max_retries', 8)
    if type(retries) is not int or retries < 0:
        raise ValueError('amp_max_retries must be a nonnegative integer')
    for update in range(1, updates + 1):
        model.train()
        replay = rng_state() if amp else None
        for attempt in range(retries + 1):
            if attempt:
                restore_rng_state(replay)
            optimizer.zero_grad(set_to_none=True)
            objective = 0.
            for batch in batches:
                with torch.autocast('cuda', dtype=torch.float16) if amp else nullcontext():
                    pred = model(**to_device(batch['sensor'], device, settings.get('non_blocking', False)))
                terms = encoder_loss(pred, to_device(batch['targets'], device), pelvis_index,
                                     settings.get('smooth_l1_beta_m', .05))
                loss = sum(terms[key] * terms[key + '_count'] / max(1, totals[key]) for key in totals)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite diagnostic objective')
                scaler.scale(loss).backward()
                objective += float(loss.detach())
            _, successful = finish_optimizer_step(optimizer, scaler, model.parameters(), settings.get('clip_grad_norm', 1.))
            if successful:
                overflow_retries += attempt
                history.append({'successful_update': update, 'train_loss_before_update': objective,
                                'overflow_retries': attempt})
                break
        else:
            raise ValueError('diagnostic AMP overflow retry budget exceeded')
    final = measure()
    ratio = final / initial if initial > 0 else None
    return {'kind': 'fixed_train_sample_overfit_diagnostic', 'scientific_gates_passed': False,
            'scientific_freeze_eligible': False, 'device': str(device), 'samples': samples,
            'successful_updates': updates, 'overflow_retries': overflow_retries,
            'frame_uids': [sample['provenance']['frame_uid'] for sample in selected],
            'initial_fp32_loss': initial, 'final_fp32_loss': final, 'loss_ratio': ratio,
            'loss_ratio_max': loss_ratio_max, 'numerical_parity': parity, 'history': history,
            'status': 'passed' if ratio is not None and ratio <= loss_ratio_max and parity['passed'] else 'failed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--updates', type=int, default=200)
    parser.add_argument('--loss-ratio-max', type=float, default=.25)
    parser.add_argument('--synthetic-smoke', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    train, val, joints, lineage = make_datasets(config, allow_synthetic=args.synthetic_smoke)
    try:
        normalizer = config.get('normalizer') or fit_input_normalizer(train.sensor_dataset)
        report = overfit(train, joints['pelvis_index'], normalizer, config['training'], args.device,
                         args.samples, args.updates, args.loss_ratio_max)
        report.update(data_kind=config['data_kind'], config_hash=canonical_hash(config), lineage=lineage,
                      normalizer_hash=canonical_hash(normalizer), scope='fixed_train_samples_only')
        atomic_json(args.output, report)
        print(json.dumps({key: report[key] for key in ('status', 'initial_fp32_loss', 'final_fp32_loss', 'loss_ratio')}, indent=2))
    finally:
        train.sensor_dataset.reader.close()
        val.sensor_dataset.reader.close()
    raise SystemExit(0 if report['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
