"""Synthetic contract checks only; real-tag fixtures are not real-data evidence."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import copy
import json
import tempfile
from unittest.mock import patch
import zipfile
import numpy as np
import torch
import yaml
from eksperimen_model.utils import m4human_runtime as runtime
from eksperimen_model.utils.m4human_gates import (encoder_gate_spec, evaluate_encoder_gate,
    require_encoder_gate, encoder_readiness_config_hash)
from eksperimen_model.package_m4human_run import package_run


def rejects(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('expected rejection')


def gate_checks(root):
    config = {'data_kind': 'm4human', 'training': {'root_quality_gate_m': .1, 'micro_batch': 1, 'accumulation': 1, 'ram_reserve_mib':2048, 'vram_reserve_mib':1024, 'resource_poll_ms':250},
              'scientific_gate_predeclared': True, 'lineage': {'source_hash': 'fake_fixture_source', 'target_hash': 'fake_fixture_target'}, 'normalizer': {'mean':[0.]*4,'std':[1.]*4}}
    config['lineage']['normalizer_hash'] = runtime.canonical_hash(config['normalizer'])
    smoke = root / 'fake_real_tag_smoke'
    smoke.mkdir()
    synthetic_run(smoke, config)
    profile = root / 'resource.json'
    runtime.atomic_json(profile, {'kind': 'bounded_train_only_resource_profile', 'device': 'cuda:0', 'data_kind': 'm4human',
        'normalizer_hash':runtime.canonical_hash(config['normalizer']),
        'lineage': config['lineage'], 'fixture_label': 'synthetic CUDA metadata only; no real CUDA evidence',
        'recommended_config_hash': runtime.canonical_hash(config),
        'readiness_config_hash': encoder_readiness_config_hash(config),
        'recommended_settings': config['training'],
        'outcomes': [{'settings': dict(config['training'], profile_normalizer=config['normalizer']), 'status': 'ok',
                     'numerical_parity': {'passed': True, 'cuda_measured': True},
                     'resource_safety': {'status':'measured','enabled':True,'cuda_measured':True,'sample_count':2,
                         'ram_reserve_bytes':2048*1024**2,'vram_reserve_bytes':1024*1024**2,
                         'min_ram_available_bytes':3072*1024**2,'min_cuda_headroom_bytes':2048*1024**2},
                     'peak_reserved_bytes': 10, 'reserve_bytes': 20, 'available_cuda_budget_bytes': 100,
                     'successful_updates': 100, 'timed_steps': 100, 'samples': 100, 'overflow_skips': 0}]})
    for key, path in (('real_smoke_passed', smoke / 'metrics.json'), ('resource_profile_passed', profile)):
        config[key] = {'artifact_path': str(path), 'artifact_sha256': runtime.file_sha256(path)}
    spec = encoder_gate_spec(config, 'pilot')
    payload = {'metadata': {'lineage': config['lineage'], 'normalizer': config['normalizer'], 'training_mode': 'pilot',
        'config_hash': runtime.canonical_hash(config), 'encoder_gate_spec': spec,
        'encoder_gate_decision': evaluate_encoder_gate(spec, .05),
        'validation_metrics': {'root': {'mean_m': .05}}}}
    assert require_encoder_gate(payload, config)['scientific_eligible']
    for normalizer in (None,{}, {'mean':[0.]*3,'std':[1.]*4}, {'mean':[0.]*4,'std':[0.]*4}):
        malformed = copy.deepcopy(config)
        malformed['normalizer'] = normalizer
        rejects(lambda:encoder_gate_spec(malformed,'final'))
    missing = copy.deepcopy(config)
    del missing['normalizer']
    rejects(lambda:encoder_gate_spec(missing,'final'))
    altered = copy.deepcopy(config)
    altered['normalizer']['mean'][0] = 1.
    rejects(lambda:encoder_gate_spec(altered,'final'))
    # Even a mutually consistent changed config/hash must reject the original artifacts.
    altered['lineage']['normalizer_hash'] = runtime.canonical_hash(altered['normalizer'])
    rejects(lambda:encoder_gate_spec(altered,'final'))
    for digest in (None, '0'*64):
        bad = runtime.read_json(profile)
        if digest is None:
            del bad['normalizer_hash']
        else:
            bad['normalizer_hash'] = digest
        bad_path = root/('normalizer_' + str(digest) + '.json')
        runtime.atomic_json(bad_path,bad)
        candidate = copy.deepcopy(config)
        candidate['resource_profile_passed'] = {'artifact_path':str(bad_path),'artifact_sha256':runtime.file_sha256(bad_path)}
        rejects(lambda:encoder_gate_spec(candidate,'final'))
    changed_smoke_config = copy.deepcopy(config)
    changed_smoke_config['normalizer']['mean'][0] = 1.
    changed_smoke_config['lineage']['normalizer_hash'] = runtime.canonical_hash(changed_smoke_config['normalizer'])
    changed_smoke = root/'changed_normalizer_smoke'
    changed_smoke.mkdir()
    synthetic_run(changed_smoke,changed_smoke_config)
    candidate = copy.deepcopy(config)
    candidate['real_smoke_passed'] = {'artifact_path':str(changed_smoke/'metrics.json'),
                                     'artifact_sha256':runtime.file_sha256(changed_smoke/'metrics.json')}
    rejects(lambda:encoder_gate_spec(candidate,'final'))
    stale_normalizer = copy.deepcopy(config)
    stale_normalizer['lineage']['normalizer_hash'] = '0'*64
    rejects(lambda:encoder_gate_spec(stale_normalizer,'final'))
    portable = copy.deepcopy(payload)
    portable['metadata']['encoder_gate_spec']['resource_profile']['normalizer_hash'] = '0'*64
    rejects(lambda:require_encoder_gate(portable,config,True))
    changed = copy.deepcopy(config)
    changed['training']['root_quality_gate_m'] = 100
    assert encoder_readiness_config_hash(config) == encoder_readiness_config_hash(changed)
    rejects(lambda: require_encoder_gate(payload, changed, True))
    # Threshold may change before a new selected run; readiness policy excludes it.
    assert encoder_gate_spec(changed, 'pilot')['threshold_m'] == 100
    smoke_spec = encoder_gate_spec(config, 'smoke')
    assert not evaluate_encoder_gate(smoke_spec, .01)['scientific_eligible']
    legacy = {'metadata': {}}
    rejects(lambda: require_encoder_gate(legacy, config))
    assert not require_encoder_gate(legacy, changed, True)['scientific_eligible']
    forged = copy.deepcopy(payload)
    forged['metadata']['encoder_gate_decision']['scientific_eligible'] = False
    rejects(lambda: require_encoder_gate(forged, config, True))
    stale = copy.deepcopy(config)
    stale['lineage']['source_hash'] = 'changed_source_bytes_under_same_paths'
    rejects(lambda: encoder_gate_spec(stale, 'final'))
    for field, value in (('data_kind', 'synthetic'), ('device', 'cpu')):
        non_cuda_report = runtime.read_json(profile)
        non_cuda_report[field] = value
        non_cuda_path = root / (field + '_profile.json')
        runtime.atomic_json(non_cuda_path, non_cuda_report)
        non_cuda = copy.deepcopy(config)
        non_cuda['resource_profile_passed'] = {'artifact_path': str(non_cuda_path), 'artifact_sha256': runtime.file_sha256(non_cuda_path)}
        rejects(lambda: encoder_gate_spec(non_cuda, 'final'))
    # Mutated synthetic CUDA metadata exercises the contract; no hardware claim.
    for field, value in (('status','pending'),('enabled',False),('cuda_measured',False),
                         ('sample_count',0),('sample_count',True),('ram_reserve_bytes',1),
                         ('vram_reserve_bytes',1),('min_ram_available_bytes',1024*1024**2),
                         ('min_cuda_headroom_bytes',512*1024**2),('min_ram_available_bytes',None),
                         ('min_cuda_headroom_bytes',float('nan')),('min_ram_available_bytes',float('inf'))):
        bad = runtime.read_json(profile)
        bad['outcomes'][0]['resource_safety'][field] = value
        # RSS can double-count shared pages and cannot substitute for available RAM.
        bad['outcomes'][0]['resource_safety']['peak_process_tree_rss_sum_bytes'] = 1
        bad_path = root / ('resource_' + field + '_' + str(value) + '.json')
        bad_path.write_text(json.dumps(bad))
        candidate = copy.deepcopy(config)
        candidate['resource_profile_passed'] = {'artifact_path':str(bad_path),'artifact_sha256':runtime.file_sha256(bad_path)}
        rejects(lambda:encoder_gate_spec(candidate,'final'))
    shared_rss = runtime.read_json(profile)
    shared_rss['outcomes'][0]['resource_safety']['peak_process_tree_rss_sum_bytes'] = 128*1024**3
    rss_path = root/'shared_rss_not_physical_usage.json'
    runtime.atomic_json(rss_path,shared_rss)
    candidate = copy.deepcopy(config)
    candidate['resource_profile_passed'] = {'artifact_path':str(rss_path),'artifact_sha256':runtime.file_sha256(rss_path)}
    assert encoder_gate_spec(candidate,'final')['resource_profile']['status'] == 'verified'
    missing_safety = runtime.read_json(profile)
    del missing_safety['outcomes'][0]['resource_safety']
    missing_path = root/'resource_safety_missing.json'
    runtime.atomic_json(missing_path,missing_safety)
    candidate = copy.deepcopy(config)
    candidate['resource_profile_passed'] = {'artifact_path':str(missing_path),'artifact_sha256':runtime.file_sha256(missing_path)}
    rejects(lambda:encoder_gate_spec(candidate,'final'))
    short_report = runtime.read_json(profile)
    short_report['outcomes'][0].update(timed_steps=10, successful_updates=10, samples=10)
    short_path = root / 'screening_profile.json'
    runtime.atomic_json(short_path, short_report)
    short = copy.deepcopy(config)
    short['resource_profile_passed'] = {'artifact_path': str(short_path), 'artifact_sha256': runtime.file_sha256(short_path)}
    rejects(lambda: encoder_gate_spec(short, 'final'))
    bad_report = runtime.read_json(profile)
    bad_report['outcomes'][0]['overflow_skips'] = 1
    overflow_path = root / 'overflow_profile.json'
    runtime.atomic_json(overflow_path, bad_report)
    overflow = copy.deepcopy(config)
    overflow['resource_profile_passed'] = {'artifact_path': str(overflow_path), 'artifact_sha256': runtime.file_sha256(overflow_path)}
    rejects(lambda: encoder_gate_spec(overflow, 'final'))
    bad_report['outcomes'][0]['overflow_skips'] = 0
    bad_report['device'] = 'cuda:0'
    bad_report['outcomes'][0].update(peak_reserved_bytes=100, reserve_bytes=20, available_cuda_budget_bytes=110)
    memory_path = root / 'unsafe_profile.json'
    runtime.atomic_json(memory_path, bad_report)
    unsafe = copy.deepcopy(config)
    unsafe['resource_profile_passed'] = {'artifact_path': str(memory_path), 'artifact_sha256': runtime.file_sha256(memory_path)}
    rejects(lambda: encoder_gate_spec(unsafe, 'final'))
    invented = copy.deepcopy(config)
    invented['resource_profile_passed']['artifact_sha256'] = '0' * 64
    rejects(lambda: encoder_gate_spec(invented, 'final'))
    profile.write_text('{}', encoding='utf-8')
    rejects(lambda: encoder_gate_spec(config, 'final'))
    # Portable extraction verifies checkpoint evidence/config without reopening files.
    profile.unlink()
    assert require_encoder_gate(payload, config)['scientific_eligible']


def synthetic_run(root, config=None):
    config = config or {'data_kind': 'synthetic'}
    (root / 'config_snapshot.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    runtime.atomic_json(root / 'lineage.json', {'contract_version': runtime.CONTRACT_VERSION,
                                             'config_hash': runtime.canonical_hash(config)})
    for name in ('history.jsonl', 'events.jsonl'):
        (root / name).write_text('{"event":"fixture"}\n', encoding='utf-8')
    row = {'frame_uid': 'fixture:0', 'subject_id': 1, 'recording_id': 'fixture',
           'metrics': {'root': {'sum': .1, 'count': 1}}}
    (root / 'predictions_val.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')
    metric = {'root': {'sum': .1, 'count': 1, 'mean': .1, 'mean_m': .1}}
    runtime.atomic_json(root / 'metrics.json', {'validation': metric, 'data_kind': config['data_kind'], 'stage': 'E', 'status': 'complete', 'successful_updates': 1})
    model = torch.nn.Linear(1, 1)
    for name in ('best.pt', 'last.pt'):
        runtime.save_checkpoint(root / name, model, {})
    np.savez(root / 'diagnostic_samples.npz', sample=np.zeros((1, 1)))
    runtime.atomic_json(root / 'diagnostic_samples.json', {'checkpoint_hash': runtime.file_sha256(root / 'best.pt'), 'records': ['fixture:0']})
    runtime.atomic_json(root / 'metric_recomputation.json', {'status': 'verified', 'kind': 'physical',
        'predictions_sha256': runtime.file_sha256(root / 'predictions_val.jsonl'), 'metrics': metric})
    runtime.atomic_json(root / 'completion.json', {'status': 'complete', 'text_hash_algorithm': 'utf8_lf_sha256',
        'metrics_hash': runtime.text_sha256(root / 'metrics.json'), 'history_hash': runtime.text_sha256(root / 'history.jsonl')})


def artifact_checks(root):
    run = root / 'run'
    run.mkdir()
    rejects(lambda: package_run(run, root / 'incomplete.zip'))
    synthetic_run(run)
    assert runtime.validate_run_artifacts(run) == []
    for name in ('metrics.json', 'history.jsonl'):
        path = run / name
        old_bytes = path.read_bytes()
        old_hash = runtime.file_sha256(path)
        path.write_bytes(old_bytes.replace(b'\n', b'\r\n'))
        assert runtime.completion_text_matches(path, old_hash)
        assert runtime.completion_text_matches(path, runtime.text_sha256(path, 'crlf'))
        path.write_bytes(old_bytes)
    assert runtime.validate_run_artifacts(run) == []
    archive = root / 'run.zip'
    package_run(run, archive)
    with zipfile.ZipFile(archive) as bundle:
        out = root / 'unpacked'
        bundle.extractall(out)
        assert bundle.read('best.pt') == (run / 'best.pt').read_bytes()
        assert bundle.read('predictions_val.jsonl') == (run / 'predictions_val.jsonl').read_bytes()
    assert runtime.validate_run_artifacts(out) == []
    metric_path = out / 'metrics.json'
    metric_path.write_bytes(metric_path.read_bytes().replace(b'synthetic', b'forged'))
    assert 'completion_hash_mismatch' in runtime.validate_run_artifacts(out)
    newline_fixture = root / 'newlines.txt'
    newline_fixture.write_bytes(b'a\nb\n')
    newline_hash = runtime.text_sha256(newline_fixture)
    newline_fixture.write_bytes(b'a\rb\n')
    assert not runtime.completion_text_matches(newline_fixture, newline_hash, 'utf8_lf_sha256')
    with (run / 'history.jsonl').open('a', encoding='utf-8') as handle:
        handle.write('{"tampered":true}\n')
    assert 'completion_history_hash_mismatch' in runtime.validate_run_artifacts(run)
    rejects(lambda: package_run(run, root / 'tampered.zip'))
    with patch.dict(sys.modules, {'resource': None}), patch('psutil.Process') as process, patch.object(torch.cuda, 'is_available', return_value=False):
        from types import SimpleNamespace
        process.return_value.memory_info.return_value = SimpleNamespace(rss=123, peak_wset=456)
        snapshot = runtime.resource_snapshot()
        assert snapshot['process_current_rss_bytes'] == 123 and snapshot['process_peak_rss_bytes'] == 456
        process.return_value.memory_info.return_value = SimpleNamespace(rss=123)
        assert runtime.resource_snapshot()['process_peak_rss_bytes'] is None


def main():
    with tempfile.TemporaryDirectory() as directory:
        gate_checks(Path(directory))
        artifact_checks(Path(directory))
    print('gate/artifact contract checks passed (synthetic fixtures only)')

if __name__ == '__main__':
    main()
