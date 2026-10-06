"""Offline numerical training regressions; no real-data or CUDA certification."""
import copy
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM

from eksperimen_model.models.m4human_projector import FrozenM4HumanLanguage
from eksperimen_model.test_m4human_language_contract import TinyTokenizer
from eksperimen_model.train_m4human_projector import _projector_logical_step, generate_predictions
from eksperimen_model import m4human_training as physical
from eksperimen_model.utils.m4human_runtime import state_dict_hash


def language_checks():
    torch.manual_seed(71)
    config = Qwen2Config(vocab_size=512, hidden_size=24, intermediate_size=48,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, max_position_embeddings=1024,
                        attention_dropout=.2)
    config._attn_implementation = 'eager'
    plain = FrozenM4HumanLanguage(Qwen2ForCausalLM(config), TinyTokenizer(), 1024, 1200)
    samples = [dict(U=torch.randn(4, 256), token_mask=torch.tensor([1, 1, 0, 1], dtype=torch.bool),
                    question='Speed?', task='root_speed_trend', answers=['speeding_up', 'slowing_down'],
                    canonical_answer=answer) for answer in ('short', 'a substantially longer answer')]
    checkpointed = copy.deepcopy(plain)
    checkpointed.checkpoint_language = True
    frozen_hash = state_dict_hash(plain.llm)
    losses, gradients = [], []
    for model in (plain, checkpointed):
        model.train()
        assert not model.llm.training
        loss = model(samples)
        loss.backward()
        losses.append(loss.detach())
        gradients.append(torch.cat([p.grad.flatten() for p in model.projector.parameters()]))
        assert gradients[-1].norm() > 0
        assert all(not p.requires_grad and p.grad is None for p in model.llm.parameters())
        assert state_dict_hash(model.llm) == frozen_hash
    torch.testing.assert_close(losses[0], losses[1], atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(gradients[0], gradients[1], atol=1e-7, rtol=1e-5)
    # Unequal answer lengths catch token-weighted instead of equal-example CE.
    results = []
    for micro in (1, 2):
        model = copy.deepcopy(plain)
        optimizer = torch.optim.SGD(model.projector.parameters(), lr=.01)
        result = _projector_logical_step(model, samples,
            {'micro_batch': micro, 'precision': 'fp32', 'clip_grad_norm': 1.},
            optimizer, torch.amp.GradScaler('cpu', enabled=False), 'cpu')
        results.append((result[0], torch.cat([p.detach().flatten() for p in model.projector.parameters()])))
    torch.testing.assert_close(torch.tensor(results[0][0]), torch.tensor(results[1][0]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(results[0][1], results[1][1], atol=1e-7, rtol=1e-5)

    # Native SDPA must preserve the frozen eager model's masked batch contract.
    sdpa_config = copy.deepcopy(config)
    sdpa_config._attn_implementation = 'sdpa'
    sdpa = FrozenM4HumanLanguage(Qwen2ForCausalLM(sdpa_config), TinyTokenizer(), 1024, 1200)
    sdpa.load_state_dict(plain.state_dict())
    sdpa_samples = [samples[0], dict(samples[1], question='A longer physical speed question?')]
    outputs, grads = [], []
    for model in (plain, sdpa):
        model.train()
        model.zero_grad(set_to_none=True)
        inputs, labels = model.assemble(sdpa_samples, training=True)
        outputs.append(model.llm(**inputs, use_cache=False).logits.detach())
        model(sdpa_samples).backward()
        grads.append(torch.cat([p.grad.flatten() for p in model.projector.parameters()]))
        assert all(p.grad is None for p in model.llm.parameters())
        assert state_dict_hash(model.llm) == frozen_hash
    torch.testing.assert_close(outputs[0], outputs[1], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(grads[0], grads[1], atol=2e-7, rtol=2e-5)
    generated = []
    for model in (plain, sdpa):
        batch_rows, batch_first = model.generate(sdpa_samples, max_new_tokens=8)
        single_rows, single_first = [], []
        for sample in sdpa_samples:
            rows, first = model.generate([sample], max_new_tokens=8)
            single_rows.extend(rows)
            single_first.append(first)
        torch.testing.assert_close(batch_first, torch.cat(single_first), atol=2e-6, rtol=2e-5)
        assert batch_rows == single_rows
        generated.append((batch_rows, batch_first))
    assert generated[0][0] == generated[1][0]
    torch.testing.assert_close(generated[0][1], generated[1][1], atol=2e-6, rtol=2e-5)


class ReplayQA(nn.Module):
    def __init__(self):
        super().__init__()
        self.projector = nn.Linear(1, 1, bias=False)
        self.projector.weight.data.fill_(.25)
        self.seen = []

    def forward(self, samples):
        noise = torch.rand(len(samples), 1)
        self.seen.append(([s['qa_id'] for s in samples], noise.clone()))
        x = torch.tensor([[s['x']] for s in samples])
        return (self.projector(x) - noise).square().mean()


def inject_first_overflow(parameter):
    calls = [0]
    def hook(gradient):
        calls[0] += 1
        return gradient * float('inf') if calls[0] == 1 else gradient
    return parameter.register_hook(hook)


def replay_checks():
    samples = [dict(qa_id='q1', x=1.), dict(qa_id='q2', x=2.)]
    qa_runs = []
    for overflow in (False, True):
        model = ReplayQA()
        if overflow:
            inject_first_overflow(model.projector.weight)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        torch.manual_seed(91)
        result = _projector_logical_step(model, samples,
            {'micro_batch': 1, 'precision': 'fp32', 'clip_grad_norm': 1.},
            optimizer, torch.amp.GradScaler('cpu', init_scale=8.), 'cpu')
        qa_runs.append((model, result, torch.get_rng_state()))
    assert qa_runs[0][1][2] == 0 and qa_runs[1][1][2] == 1
    assert len(qa_runs[1][0].seen) == 4
    for first, replay in zip(qa_runs[1][0].seen[:2], qa_runs[1][0].seen[2:]):
        assert first[0] == replay[0]
        assert torch.equal(first[1], replay[1])
    torch.testing.assert_close(qa_runs[0][0].projector.weight, qa_runs[1][0].projector.weight)
    assert qa_runs[0][1][:2] == qa_runs[1][1][:2]
    assert torch.equal(qa_runs[0][2], qa_runs[1][2])

    logical = []
    for index in (1, 2):
        target = {key: torch.ones((1, 2), dtype=torch.bool) if key.endswith('joint')
                  else torch.ones(1, dtype=torch.bool) for key in
                  ('position_valid_joint', 'position_valid_root', 'velocity_valid_joint', 'velocity_valid_root')}
        logical.append(dict(sensor={'identity': torch.tensor(float(index))}, targets=target,
                            provenance=[{'window_id': f'w{index}'}]))
    runs = []
    for overflow in (False, True):
        model = nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(.25)
        if overflow:
            inject_first_overflow(model.weight)
        seen = []
        def forward(model, stage, condition, sensor, targets, stats, config):
            noise = torch.rand(())
            seen.append((sensor['identity'].item(), noise.clone()))
            term = (model.weight.squeeze() * sensor['identity'] - noise).square()
            return None, {'L_' + key: term for key in ('p_joint', 'p_root', 'v_joint', 'v_root')}, None
        torch.manual_seed(92)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        # CPU GradScaler exercises real overflow detection; omit CUDA autocast only.
        with patch.object(physical, '_forward', side_effect=forward), patch.object(torch, 'autocast', return_value=nullcontext()):
            result = physical._logical_training_step(model, 'M', None, logical, {},
                {'pelvis_index': 0}, torch.device('cpu'), optimizer,
                torch.amp.GradScaler('cpu', init_scale=8.))
        runs.append((model.weight.detach().clone(), result, seen, torch.get_rng_state()))
    assert runs[0][1][-1] == 0 and runs[1][1][-1] == 1
    torch.testing.assert_close(runs[0][0], runs[1][0])
    assert runs[0][1][:-1] == runs[1][1][:-1]
    assert torch.equal(runs[0][3], runs[1][3])
    for first, replay in zip(runs[1][2][:2], runs[1][2][2:]):
        assert first[0] == replay[0] and torch.equal(first[1], replay[1])


def generation_checks():
    class Dataset:
        metadata = {'condition': 'C_base', 'K': 16}
        records = [dict(qa_id=f'q{i}', window_id=f'w{i}', subject_id=i,
                        recording_id=f'r{i}', task='root_speed_trend') for i in range(4)]
        def __len__(self): return len(self.records)
        def __getitem__(self, index):
            return dict(self.records[index], token_mask=torch.ones(index + 1, dtype=torch.bool))
    class Generator:
        calls = []
        def generate(self, samples, max_new_tokens):
            ids = [s['qa_id'] for s in samples]
            self.calls.append(ids)
            if 'q1' in ids:
                raise ValueError('invalid fixture')
            return [dict(raw_ids=[int(s['qa_id'][1:])],
                         raw_text='{"task":"root_speed_trend","answer":"speeding_up"}')
                    for s in samples], None
    model = Generator()
    rows = generate_predictions(model, Dataset(), 32, batch_size=2)
    assert [r['qa_id'] for r in rows] == ['q0', 'q1', 'q2', 'q3']
    assert [r['window_id'] for r in rows] == ['w0', 'w1', 'w2', 'w3']
    assert model.calls == [['q0', 'q1'], ['q0'], ['q1'], ['q2', 'q3']]
    assert rows[1]['error'] == 'invalid fixture' and not rows[1]['parse_valid']
    for i in (0, 2, 3):
        assert rows[i]['error'] is None and rows[i]['parse_valid']
        assert rows[i]['raw_ids'] == [i] and rows[i]['n_valid_tokens'] == i + 1
    class OOMGenerator:
        def __init__(self):
            self.failures, self.calls = [], []
        def generate(self, samples, max_new_tokens):
            self.calls.append([s['qa_id'] for s in samples])
            if len(samples) > 1:
                error = torch.cuda.OutOfMemoryError('simulated batch allocation failure')
                self.failures.append(error)
                raise error
            return [dict(raw_ids=[int(samples[0]['qa_id'][1:])],
                         raw_text='{"task":"root_speed_trend","answer":"speeding_up"}')], None
    oom_model = OOMGenerator()
    def release_cache():
        # Verify failed forward frames are released before retrying singletons.
        assert oom_model.failures[-1].__traceback__ is None
    with patch.object(torch.cuda, 'is_available', return_value=True), \
            patch.object(torch.cuda, 'empty_cache', side_effect=release_cache) as cache:
        recovered = generate_predictions(oom_model, Dataset(), 32, batch_size=2)
    assert cache.call_count == 2
    assert oom_model.calls == [['q0', 'q1'], ['q0'], ['q1'], ['q2', 'q3'], ['q2'], ['q3']]
    assert [r['qa_id'] for r in recovered] == ['q0', 'q1', 'q2', 'q3']
    assert all(r['error'] is None and r['parse_valid'] for r in recovered)
    assert [r['raw_ids'] for r in recovered] == [[0], [1], [2], [3]]


def epoch_boundary_checks():
    from eksperimen_model.test_m4human_integration import fixture
    original_step = physical._logical_training_step
    with tempfile.TemporaryDirectory() as temporary:
        config = fixture(Path(temporary))
        config.update(accumulation=1, successful_updates=2)
        path = Path(temporary) / 'training.yaml'
        path.write_text(yaml.safe_dump(config))
        calls = [0]
        def sparse_objectives(*args, **kwargs):
            calls[0] += 1
            # Four train windows: one supported update then three empty batches.
            return original_step(*args, **kwargs) if calls[0] % 4 == 1 else None
        with patch.object(physical, '_logical_training_step', side_effect=sparse_objectives), \
                patch.object(physical, 'save_checkpoint', wraps=physical.save_checkpoint) as checkpoints:
            physical.train(path, 'M')
        last = [call.kwargs for call in checkpoints.call_args_list if Path(call.args[0]).name == 'last.pt']
        assert len(last) == 2
        assert last[0]['successful_updates'] == 1 and last[0]['epoch_complete'] is True
        # The next epoch stops on its update budget before consuming other batches.
        assert last[1]['successful_updates'] == 2 and last[1]['epoch_complete'] is False


def main():
    torch.set_num_threads(1)
    language_checks()
    replay_checks()
    generation_checks()
    epoch_boundary_checks()
    print('M4Human training hardening: PASS (offline CPU; CUDA/real data not measured)')


if __name__ == '__main__':
    main()
