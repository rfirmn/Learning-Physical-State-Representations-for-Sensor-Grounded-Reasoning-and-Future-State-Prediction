"""CPU-only executable architecture/gradient checks; no real-data claim."""
import copy
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from eksperimen_model.models.m4human_motion import CausalDSTformerLiteV1, make_rich_memory, build_rich_attention_memory, fixed_time_encoding
from eksperimen_model.models.m4human_tokenizer import KinematicTokenLearnerV1
from eksperimen_model.models.m4human_readout import KinematicReadoutV2, FullHFidelityDecoderV1, TokenKinematicReadoutV1, fidelity_loss, clip_compression_gradients
from eksperimen_model.utils.m4human_runtime import state_dict_hash


def main():
    torch.set_num_threads(1); torch.manual_seed(42)
    time = torch.arange(32).double()[None] / 12
    sensor = {'P_enc_relative_m':torch.randn(1,32,22,3),'r_enc_m':torch.randn(1,32,3),'f_enc':torch.randn(1,32,256)}
    sensor['P_enc_relative_m'][:,:,0] = 0.
    masks = {'joint_sensor_valid':torch.ones(1,32,22,dtype=torch.bool), **{k:torch.ones(1,32,dtype=torch.bool) for k in ('root_sensor_valid','feature_valid','context_time_valid')}}
    motion = CausalDSTformerLiteV1(stages=4,dropout=0.).eval()
    out = motion(sensor,masks,time)
    changed = {k:v.clone() for k,v in sensor.items()}
    for v in changed.values():
        v[:,17:] += 50.
    suffix = motion(changed,masks,time)
    torch.testing.assert_close(out['H'][:,:17],suffix['H'][:,:17],rtol=0,atol=0)
    assert out['H'].shape == (1,32,23,128)
    invalid_masks = {k:v.clone() for k,v in masks.items()}
    for v in invalid_masks.values():
        v[:,0:2] = False
    invalid_sensor = {k:v.clone() for k,v in sensor.items()}
    for v in invalid_sensor.values():
        v[:,0:2] = float('nan')
    invalid = motion(invalid_sensor,invalid_masks,time)
    assert torch.isfinite(invalid['H']).all() and not invalid['H'][:,:2].any()
    for v in invalid_sensor.values():
        v[:,0:2] = 1e9
    torch.testing.assert_close(invalid['H'],motion(invalid_sensor,invalid_masks,time)['H'],rtol=0,atol=0)
    bad = {k:v.clone() for k,v in sensor.items()}; bad['f_enc'][0,5,0] = float('nan')
    try:
        motion(bad,masks,time)
        raise AssertionError('valid NaN was accepted')
    except ValueError:
        pass
    scales = {'s_delta_joint':.1,'s_delta_root':.1,'s_v_joint':1.,'s_v_root':1.}
    head = KinematicReadoutV2(scales)
    physical = head(out['H'],sensor['P_enc_relative_m'],sensor['r_enc_m'],out['latent_valid'])
    torch.testing.assert_close(physical['P_relative_m'],sensor['P_enc_relative_m'])
    torch.testing.assert_close(physical['r_m'],sensor['r_enc_m'])
    physical['v_relative_mps'].square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in motion.parameters())
    h = out['H'].detach(); raw = make_rich_memory(h); valid = out['latent_valid'] & out['common_frame_valid'][...,None]
    memory = build_rich_attention_memory(raw,time,valid)
    torch.testing.assert_close(memory,raw + fixed_time_encoding(time,256).unsqueeze(2))
    compressor = KinematicTokenLearnerV1(); fidelity = FullHFidelityDecoderV1(); auxiliary = TokenKinematicReadoutV1(scales)
    for K in (8,16,32):
        tokens = compressor(raw,time,valid,K)
        assert tokens['U'].shape == (1,K,256) and tokens['token_mask'].all()
        assert tokens['bin_counts'].eq(32//K).all()
        torch.testing.assert_close(tokens['bin_start'],torch.arange(K)*32//K)
        torch.testing.assert_close(tokens['bin_end'],(torch.arange(K)+1)*32//K)
        empty_valid = valid.clone(); empty_valid[:,:32//K] = False
        empty_raw = raw.clone(); empty_raw[:,:32//K] = float('nan')
        empty = compressor(empty_raw,time,empty_valid,K)
        assert not empty['token_mask'][0,0] and not empty['U'][0,0].any() and torch.isfinite(empty['U']).all()
    try:
        compressor(raw,time,torch.zeros_like(valid),16)
        raise AssertionError('all-empty scientific window accepted')
    except ValueError:
        pass
    tokens = compressor(raw,time,valid,16)
    base = auxiliary(memory.flatten(1,2),valid.flatten(1,2),time)
    base_loss = sum(v.square().mean() for v in base.values())
    assert all(g is None for g in torch.autograd.grad(base_loss,list(compressor.parameters()),allow_unused=True))
    kin = auxiliary(tokens['U'],tokens['token_mask'],time)
    kin_loss = sum(v.square().mean() for v in kin.values())
    assert any(g is not None and g.abs().sum()>0 for g in torch.autograd.grad(kin_loss,list(compressor.parameters()),retain_graph=True,allow_unused=True))
    reconstructed = fidelity(tokens['U'],tokens['token_mask'],time)
    assert reconstructed.shape == h.shape
    objective = fidelity_loss(reconstructed,h,valid,torch.zeros(128),torch.ones(128))['loss']
    assert any(g.abs().sum()>0 for g in torch.autograd.grad(objective,list(compressor.parameters()),retain_graph=True))
    torch.testing.assert_close(fidelity(tokens['U'],tokens['token_mask'],time,query_chunk=71),reconstructed,rtol=1e-5,atol=1e-5)
    # Baseline auxiliary scale cannot alter compressor updates via a shared clip norm.
    initial = (copy.deepcopy(compressor),copy.deepcopy(fidelity),copy.deepcopy(auxiliary))
    outcomes=[]
    for multiplier in (1.,1e6):
        c,f,a = (copy.deepcopy(x) for x in initial)
        optimizer = torch.optim.SGD(list(c.parameters())+list(f.parameters())+list(a.parameters()),lr=.01)
        u=c(raw,time,valid,16); h_hat=f(u['U'],u['token_mask'],time)
        p=a(memory.flatten(1,2),valid.flatten(1,2),time)
        loss=fidelity_loss(h_hat,h,valid,torch.zeros(128),torch.ones(128))['loss'] + multiplier*sum(v.square().mean() for v in p.values())
        loss.backward(); clip_compression_gradients(c,f,a); optimizer.step(); outcomes.append(state_dict_hash(c))
    assert outcomes[0] == outcomes[1]
    # Invalid memory must be sanitized before LayerNorm, including backward.
    padded = tokens['U'].detach().clone()
    token_mask = tokens['token_mask'].clone()
    token_mask[:,0] = False
    padded[:,0] = float('nan')
    auxiliary.zero_grad(set_to_none=True)
    padded_result = auxiliary(padded,token_mask,time)
    sum(value.square().mean() for value in padded_result.values()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in auxiliary.parameters())
    assert sum(p.numel() for p in head.parameters()) == 35084
    print('PASS synthetic motion/tokenizer/full-H/auxiliary-gradient/separate-clipping contract; real-data gates remain unverified')


if __name__ == '__main__':
    main()
