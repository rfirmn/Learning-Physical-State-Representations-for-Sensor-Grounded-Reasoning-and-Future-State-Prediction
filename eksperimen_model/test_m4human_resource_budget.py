"""Resource accounting fixtures; no CUDA or real-data readiness claims."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import subprocess
import tempfile
import yaml
import time
from types import SimpleNamespace
from unittest.mock import patch
from eksperimen_model.utils.m4human_performance import ResourceMonitor, validate_performance
from eksperimen_model.utils.m4human_runtime import process_tree_resources


def rejects(call, exception=ValueError):
    try:
        call()
    except exception:
        return
    raise AssertionError('expected rejection')


def training_budget_failures():
    """Actual M train fixture must never finish complete after a budget failure."""
    from eksperimen_model.test_m4human_integration import fixture
    from eksperimen_model import m4human_training as training
    from eksperimen_model.utils.m4human_runtime import RunLogger, read_json
    for failure_at in ('loop', 'final_exit'):
        observed = []
        class TrackingLogger(RunLogger):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                observed.append(self)
        class FailingMonitor:
            def __init__(self, settings, device):
                self.checks = 0
            def __enter__(self):
                return self
            def check(self):
                self.checks += 1
                if failure_at == 'loop' and self.checks >= 2:
                    raise RuntimeError('injected available RAM reserve failure')
                return self.snapshot()
            def snapshot(self):
                return {'status': 'fixture_only', 'cuda_measured': False}
            def __exit__(self, kind, exception, traceback):
                if failure_at == 'final_exit' and kind is None:
                    raise RuntimeError('injected final resource measurement failure')
                return False
        with tempfile.TemporaryDirectory(prefix='m4human_budget_train_') as directory:
            root = Path(directory)
            config = fixture(root)
            config_path = root / 'config.yaml'
            config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
            with patch.object(training, 'ResourceMonitor', FailingMonitor), patch.object(training, 'RunLogger', TrackingLogger):
                rejects(lambda: training.train(config_path, 'M'), RuntimeError)
            assert len(observed) == 1 and observed[0].closed
            run = Path(config['output_dir']) / 'M'
            assert read_json(run / 'metrics.json')['status'] == 'failed'
            # Failed completion markers are diagnostic artifacts; never success.
            assert read_json(run / 'completion.json')['status'] == 'failed'


def main():
    for settings in ({'ram_reserve_mib': -1}, {'vram_reserve_mib': True},
                     {'resource_poll_ms': 0}, {'resource_poll_ms': 49}, {'ram_reserve_mib': 1.5}):
        rejects(lambda: validate_performance(settings))
    with patch('eksperimen_model.utils.m4human_performance.process_tree_resources', side_effect=AssertionError('must not sample')):
        with ResourceMonitor({}, 'cpu') as disabled:
            assert disabled.check()['status'] == 'not_measured'
            assert not disabled.snapshot()['cuda_measured']
    baseline = process_tree_resources()
    low = dict(baseline, ram_available_bytes=1024)
    with patch('eksperimen_model.utils.m4human_performance.process_tree_resources', return_value=low):
        rejects(lambda: ResourceMonitor({'ram_reserve_mib': 1}, 'cpu').__enter__(), RuntimeError)
    child = subprocess.Popen([sys.executable, '-c',
        "import sys; allocation = bytearray(32 * 1024**2); print('ready', flush=True); sys.stdin.readline()"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'ready'
        with ResourceMonitor({'ram_reserve_mib': 0, 'vram_reserve_mib': 0, 'resource_poll_ms': 100}, 'cpu') as monitor:
            time.sleep(.25)
            snapshot = monitor.check()
            assert snapshot['sample_count'] >= 2
            assert snapshot['process_scope'] == 'parent_and_recursive_descendants'
            rows = {row['pid']: row for row in snapshot['processes']}
            assert child.pid in rows and rows[child.pid]['rss_bytes'] >= 32 * 1024**2
            assert snapshot['process_tree_rss_sum_bytes'] == sum(row['rss_bytes'] for row in rows.values())
            assert 'not_physical_RAM_usage' in snapshot['rss_sum_semantics']
            if snapshot['process_tree_uss_sum_bytes'] is not None:
                assert snapshot['process_tree_uss_sum_bytes'] == sum(row['uss_bytes'] for row in rows.values())
            assert snapshot['ram_total_bytes'] > snapshot['min_ram_available_bytes'] >= 0
            assert not snapshot['cuda_measured'] and 'cuda_total_bytes' not in snapshot
            assert snapshot['sampling_seconds'] >= 0 and snapshot['sampling_wall_fraction'] >= 0
    finally:
        child.communicate('\n', timeout=5)
    def budget_drop():
        with ResourceMonitor({'ram_reserve_mib': 1, 'resource_poll_ms': 100}, 'cpu') as monitor:
            with patch('eksperimen_model.utils.m4human_performance.process_tree_resources', return_value=low):
                time.sleep(.15)
                monitor.check()
    rejects(budget_drop, RuntimeError)
    # A stalled sampler cannot certify completion; retain a pre-existing error.
    stalled=ResourceMonitor({},'cpu')
    stalled.__enter__()
    stalled._thread=SimpleNamespace(join=lambda **kwargs:None,is_alive=lambda:True)
    rejects(lambda:stalled.__exit__(None,None,None),RuntimeError)
    assert stalled.snapshot()['status']=='measurement_failed'
    assert stalled.__exit__(ValueError,ValueError('original failure'),None) is False
    training_budget_failures()
    print('Resource budget checks passed (CPU accounting and failed actual training fixtures only)')

if __name__ == '__main__':
    main()
