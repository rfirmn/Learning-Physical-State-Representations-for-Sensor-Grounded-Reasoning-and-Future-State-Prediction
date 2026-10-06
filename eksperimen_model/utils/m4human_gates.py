"""Predeclared encoder freeze contract backed by concrete readiness artifacts."""
import copy
import math
from pathlib import Path
import yaml
from .m4human_runtime import canonical_hash, file_sha256, read_json, validate_run_artifacts


def encoder_readiness_config_hash(config):
    cleaned = copy.deepcopy({k: v for k, v in config.items()
                            if k not in ('real_smoke_passed', 'resource_profile_passed', 'lineage', 'normalizer')})
    cleaned.get('training', {}).pop('root_quality_gate_m', None)
    return canonical_hash(cleaned)


def encoder_sensor_policy_hash(config):
    training = config.get('training', {})
    return canonical_hash({**{k: config.get(k) for k in ('contract_version', 'data_kind', 'paths', 'data', 'encoder')},
        'scientific_training': {k: training.get(k) for k in ('optimizer', 'selection_metric',
            'smooth_l1_beta_m', 'lambda_root', 'lambda_bone')}})


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0



def _resolved_normalizer_hash(config):
    normalizer = config.get('normalizer')
    if not isinstance(normalizer,dict):
        raise ValueError('resolved encoder normalizer required')
    for key in ('mean','std'):
        values = normalizer.get(key)
        if (not isinstance(values,(list,tuple)) or len(values) != 4
                or any(type(value) not in (int,float) or not math.isfinite(value)
                       or (key == 'std' and value <= 0) for value in values)):
            raise ValueError('resolved encoder normalizer requires finite mean/std4 and positive std')
    digest = canonical_hash(normalizer)
    if 'normalizer_hash' in config.get('lineage',{}) and config['lineage']['normalizer_hash'] != digest:
        raise ValueError('resolved normalizer/lineage hash mismatch')
    return digest


def _verify_readiness(config, name):
    evidence = config.get(name)
    if evidence is None or evidence is False:
        return None
    if not isinstance(evidence, dict) or not evidence.get('artifact_path') or not evidence.get('artifact_sha256'):
        raise ValueError(name + ' requires an artifact path and exact SHA256')
    path = Path(evidence['artifact_path'])
    if not path.is_file() or file_sha256(path) != evidence['artifact_sha256']:
        raise ValueError(name + ' artifact missing or hash mismatch')
    artifact = read_json(path)
    normalizer_hash = _resolved_normalizer_hash(config)
    policy = encoder_sensor_policy_hash(config)
    intended = encoder_readiness_config_hash(config)
    if name == 'real_smoke_passed':
        if path.name != 'metrics.json' or validate_run_artifacts(path.parent):
            raise ValueError('real smoke requires a complete validated E run')
        smoke_config = yaml.safe_load((path.parent / 'config_snapshot.yaml').read_text(encoding='utf-8'))
        root = artifact.get('validation', {}).get('root', {})
        if (artifact.get('stage') != 'E' or artifact.get('data_kind') != 'm4human'
                or artifact.get('status') != 'complete' or not _positive(artifact.get('successful_updates'))
                or not _positive(root.get('count')) or type(root.get('mean_m')) not in (int, float)
                or not math.isfinite(root['mean_m']) or root['mean_m'] < 0
                or encoder_sensor_policy_hash(smoke_config) != policy):
            raise ValueError('real smoke source/sensor policy or finite training evidence mismatch')
        if _resolved_normalizer_hash(smoke_config) != normalizer_hash:
            raise ValueError('real smoke normalizer mismatch')
        smoke_lineage = smoke_config.get('lineage', {})
        intended_lineage = config.get('lineage', {})
        for key in ('schema_hash', 'joint_map_hash', 'coordinate_hash', 'source_hash',
                    'target_hash', 'frame_manifest_hash', 'split_hash', 'recipe_hash', 'normalizer_hash'):
            if key in intended_lineage and (not intended_lineage[key] or smoke_lineage.get(key) != intended_lineage[key]):
                raise ValueError('real smoke stale lineage: ' + key)
        artifact_config_hash = encoder_readiness_config_hash(smoke_config)
    else:
        if artifact.get('normalizer_hash') != normalizer_hash:
            raise ValueError('resource profile normalizer hash missing/mismatch')
        recommended = artifact.get('recommended_settings')
        outcomes = [row for row in artifact.get('outcomes', [])
                    if {k: v for k, v in row.get('settings', {}).items() if k != 'profile_normalizer'} == recommended]
        if (artifact.get('kind') != 'bounded_train_only_resource_profile'
                or artifact.get('readiness_config_hash') != intended
                or artifact.get('data_kind') != 'm4human'
                or not str(artifact.get('device', '')).startswith('cuda')
                or not isinstance(recommended, dict)
                or any(config.get('training', {}).get(key) != value for key, value in recommended.items()
                       if key not in ('effective_batch', 'root_quality_gate_m')) or len(outcomes) != 1):
            raise ValueError('resource profile recommendation/config mismatch')
        for key in ('schema_hash', 'joint_map_hash', 'coordinate_hash', 'source_hash',
                    'target_hash', 'frame_manifest_hash', 'split_hash', 'recipe_hash'):
            if key in config.get('lineage', {}) and artifact.get('lineage', {}).get(key) != config['lineage'][key]:
                raise ValueError('resource profile stale lineage: ' + key)
        outcome = outcomes[0]
        if (outcome.get('status') != 'ok' or outcome.get('numerical_parity', {}).get('passed') is not True
                or outcome.get('numerical_parity', {}).get('cuda_measured') is not True
                or outcome.get('overflow_skips') != 0 or not _positive(outcome.get('successful_updates'))
                or not _positive(outcome.get('samples'))):
            raise ValueError('resource profile lacks successful parity/no-overflow evidence')
        timed_steps = outcome.get('timed_steps')
        micro_batch, accumulation = recommended.get('micro_batch'), recommended.get('accumulation')
        if (type(timed_steps) is not int or timed_steps < 100
                or type(micro_batch) is not int or micro_batch < 1
                or type(accumulation) is not int or accumulation < 1
                or outcome.get('successful_updates') != timed_steps
                or outcome.get('samples') != timed_steps * micro_batch * accumulation):
            raise ValueError('resource profile requires 100+ complete timed updates with exact exposure')
        safety = outcome.get('resource_safety', {})
        settings = config.get('training', {})
        reserves = (settings.get('ram_reserve_mib',2048), settings.get('vram_reserve_mib',1024))
        if any(not _positive(value) for value in reserves):
            raise ValueError('resource profile requires positive RAM/VRAM reserves')
        ram_reserve, vram_reserve = (value * 1024**2 for value in reserves)
        if (not isinstance(safety,dict) or safety.get('status') != 'measured'
                or safety.get('enabled') is not True or safety.get('cuda_measured') is not True
                or type(safety.get('sample_count')) is not int or safety['sample_count'] < 1
                or any(not _positive(safety.get(key)) for key in
                       ('ram_reserve_bytes','vram_reserve_bytes','min_ram_available_bytes','min_cuda_headroom_bytes'))
                or safety['ram_reserve_bytes'] != ram_reserve or safety['vram_reserve_bytes'] != vram_reserve
                or safety['min_ram_available_bytes'] < ram_reserve
                or safety['min_cuda_headroom_bytes'] < vram_reserve):
            raise ValueError('resource profile lacks measured safe system RAM/CUDA headroom')
        if str(artifact.get('device', '')).startswith('cuda'):
            reserved, reserve, budget = (outcome.get(key) for key in
                ('peak_reserved_bytes', 'reserve_bytes', 'available_cuda_budget_bytes'))
            if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (reserved, reserve, budget)) or reserved + reserve > budget:
                raise ValueError('resource profile memory safety mismatch')
        artifact_config_hash = intended
    return {'status': 'verified', 'artifact_path': str(path), 'artifact_sha256': evidence['artifact_sha256'],
            'config_hash': intended, 'artifact_config_hash': artifact_config_hash,
            'sensor_policy_hash': policy, 'normalizer_hash': normalizer_hash}


def encoder_gate_spec(config, mode, verified_evidence=None):
    if mode not in ('smoke', 'pilot', 'final'):
        raise ValueError('unknown encoder training mode')
    normalizer_hash = _resolved_normalizer_hash(config) if mode == 'final' or 'normalizer' in config else None
    # Extraction supplies previously verified evidence, preserving independence
    # from mounted smoke runs or profiler reports. Full config still binds it.
    evidence = verified_evidence if verified_evidence is not None else {
        'real_smoke': _verify_readiness(config, 'real_smoke_passed'),
        'resource_profile': _verify_readiness(config, 'resource_profile_passed')}
    return {'version': 2, 'mode': mode, 'data_kind': config.get('data_kind'),
            'config_hash': canonical_hash(config), 'readiness_config_hash': encoder_readiness_config_hash(config),
            'normalizer_hash': normalizer_hash,
            'threshold_m': config.get('training', {}).get('root_quality_gate_m'),
            'scientific_gate_predeclared': config.get('scientific_gate_predeclared') is True, **evidence}


def evaluate_encoder_gate(spec, root_error):
    reasons = []
    if spec.get('mode') not in ('pilot', 'final'):
        reasons.append('debug_mode')
    if spec.get('data_kind') != 'm4human':
        reasons.append('non_real_data')
    if spec.get('mode') == 'pilot' and not spec.get('scientific_gate_predeclared'):
        reasons.append('pilot_gate_not_predeclared')
    threshold = spec.get('threshold_m')
    if (type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold <= 0
            or type(root_error) not in (int, float) or not math.isfinite(root_error) or root_error < 0):
        reasons.append('root_gate_unresolved')
    elif root_error > threshold:
        reasons.append('root_gate_failed')
    for name in ('real_smoke', 'resource_profile'):
        evidence = spec.get(name)
        if (spec.get('version') != 2 or not isinstance(evidence, dict) or evidence.get('status') != 'verified'
                or evidence.get('config_hash') != spec.get('readiness_config_hash')
                or not evidence.get('artifact_sha256') or not evidence.get('artifact_config_hash')
                or not evidence.get('sensor_policy_hash') or not spec.get('normalizer_hash')
                or evidence.get('normalizer_hash') != spec['normalizer_hash']):
            reasons.append(name + '_readiness_unverified')
    return {'spec_hash': canonical_hash(spec), 'root_error_m': root_error,
            'scientific_eligible': not reasons, 'reasons': reasons}


def require_encoder_gate(payload, config, allow_debug=False):
    metadata = payload['metadata']
    spec, decision = metadata.get('encoder_gate_spec'), metadata.get('encoder_gate_decision')
    if not isinstance(spec, dict) or not isinstance(decision, dict):
        if allow_debug:
            return {'scientific_eligible': False, 'reasons': ['legacy_gate_metadata_missing']}
        raise ValueError('checkpoint has no verified encoder freeze decision')
    resolved = dict(config, lineage=metadata['lineage'], normalizer=metadata['normalizer'])
    expected = encoder_gate_spec(resolved, metadata.get('training_mode'), {
        'real_smoke': spec.get('real_smoke'), 'resource_profile': spec.get('resource_profile')})
    if canonical_hash(spec) != canonical_hash(expected) or metadata.get('config_hash') != spec['config_hash']:
        raise ValueError('encoder gate config mismatch; checkpoint cannot be promoted')
    for name, config_name in (('real_smoke', 'real_smoke_passed'), ('resource_profile', 'resource_profile_passed')):
        recorded, declared = spec.get(name), resolved.get(config_name)
        if recorded is not None and (not isinstance(declared, dict)
                or recorded.get('artifact_sha256') != declared.get('artifact_sha256')
                or recorded.get('artifact_path') != str(Path(declared.get('artifact_path', '')))
                or recorded.get('sensor_policy_hash') != encoder_sensor_policy_hash(resolved)):
            raise ValueError('checkpoint readiness evidence mismatch')
    root_error = metadata.get('validation_metrics', {}).get('root', {}).get('mean_m')
    verified = evaluate_encoder_gate(spec, root_error)
    if canonical_hash(decision) != canonical_hash(verified):
        raise ValueError('checkpoint encoder gate decision mismatch')
    if not verified['scientific_eligible'] and not allow_debug:
        raise ValueError('encoder freeze gate failed: ' + ', '.join(verified['reasons']))
    return verified
