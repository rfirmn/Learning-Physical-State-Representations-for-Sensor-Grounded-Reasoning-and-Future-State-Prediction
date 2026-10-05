"""Sensor-only causal joint/time backbone for m4human_kinetok_v3."""
import math
import torch
from torch import nn


def fixed_time_encoding(time_s, width, origin_s=None):
    if time_s.ndim != 2 or width % 2 or not torch.isfinite(time_s).all():
        raise ValueError('finite [B,T] timestamps and even PE width required')
    origin = time_s[:, :1] if origin_s is None else origin_s.reshape(-1, 1)
    x = (time_s - origin).float() * 12.0
    frequency = torch.exp(torch.arange(0, width, 2, device=x.device).float() * (-math.log(10000.0) / width))
    phase = x.unsqueeze(-1) * frequency
    return torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)


def sanitize_valid(x, valid, name):
    if valid.dtype != torch.bool or x.shape[:valid.ndim] != valid.shape:
        raise ValueError(f'{name}: inconsistent boolean validity')
    if not torch.isfinite(x[valid]).all():
        raise ValueError(f'{name}: non-finite value on valid sensor support')
    mask = valid.reshape(*valid.shape, *([1] * (x.ndim - valid.ndim)))
    return torch.where(mask, x, torch.zeros_like(x))


def initialize_neural(module):
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.MultiheadAttention):
            nn.init.xavier_uniform_(layer.in_proj_weight)
            if layer.in_proj_bias is not None:
                nn.init.zeros_(layer.in_proj_bias)
        elif isinstance(layer, nn.LayerNorm) and layer.elementwise_affine:
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)


def guarded_attention(attention, query, memory, key_valid, query_valid, blocked=None):
    """MHA True=blocked; dummy key only opens for an invalid query."""
    n, q, _ = query.shape
    k = memory.shape[1]
    deny = (~key_valid[:, None, :]).expand(n, q, k).clone()
    if blocked is not None:
        deny |= blocked
    empty = deny.all(-1)
    if (empty & query_valid).any():
        raise ValueError('valid attention query has no eligible key')
    deny[..., 0] &= ~empty
    memory = sanitize_valid(memory, key_valid, 'attention memory')
    query = sanitize_valid(query, query_valid, 'attention query')
    output = attention(query, memory, memory, attn_mask=deny.repeat_interleave(attention.num_heads, 0), need_weights=False)[0]
    return sanitize_valid(output, query_valid, 'attention output')


class _AxisAttention(nn.Module):
    def __init__(self, axis, dropout):
        super().__init__()
        self.axis = axis
        self.norm = nn.LayerNorm(128)
        self.attention = nn.MultiheadAttention(128, 4, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid):
        b, t, j, d = x.shape
        if self.axis == 'time':
            view = x.permute(0, 2, 1, 3).reshape(b * j, t, d)
            mask = valid.permute(0, 2, 1).reshape(b * j, t)
            blocked = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
        else:
            view, mask = x.reshape(b * t, j, d), valid.reshape(b * t, j)
            blocked = None
        normalized = sanitize_valid(self.norm(view), mask, 'normalized axis')
        out = guarded_attention(self.attention, normalized, normalized, mask, mask, blocked)
        out = sanitize_valid(view + self.dropout(out), mask, 'axis residual')
        return out.reshape(b, j, t, d).permute(0, 2, 1, 3) if self.axis == 'time' else out.reshape(b, t, j, d)


class _Stage(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.s_st, self.t_st = _AxisAttention('space', dropout), _AxisAttention('time', dropout)
        self.t_ts, self.s_ts = _AxisAttention('time', dropout), _AxisAttention('space', dropout)
        self.ffn = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128), nn.Dropout(dropout))

    def forward(self, x, valid):
        y = 0.5 * (self.t_st(self.s_st(x, valid), valid) + self.s_ts(self.t_ts(x, valid), valid))
        return sanitize_valid(y + self.ffn(y), valid, 'stage residual')


class CausalDSTformerLiteV1(nn.Module):
    def __init__(self, root_mean=None, root_std=None, stages=4, dropout=0.1):
        super().__init__()
        self.register_buffer('root_mean', torch.as_tensor([0., 0., 0.] if root_mean is None else root_mean).float())
        self.register_buffer('root_std', torch.as_tensor([1., 1., 1.] if root_std is None else root_std).float().clamp_min(1e-3))
        self.joint_projection = nn.Linear(3, 128)
        self.global_projection = nn.Linear(259, 128)
        self.feature_norm = nn.LayerNorm(256, elementwise_affine=False)
        self.joint_id = nn.Parameter(torch.empty(23, 128))
        self.stages = nn.ModuleList(_Stage(dropout) for _ in range(stages))
        self.norm = nn.LayerNorm(128)
        initialize_neural(self)
        nn.init.normal_(self.joint_id, std=0.02)

    def forward(self, sensor_state, sensor_masks, time_s):
        p, r, f = (sensor_state[k] for k in ('P_enc_relative_m', 'r_enc_m', 'f_enc'))
        if p.shape != (*time_s.shape, 22, 3) or r.shape != (*time_s.shape, 3) or f.shape != (*time_s.shape, 256):
            raise ValueError('motion sensor tensor shape mismatch')
        if time_s.shape[1] != 32 or (time_s.diff(dim=1) <= 0).any():
            raise ValueError('motion requires exactly 32 increasing timestamps')
        context = sensor_masks['context_time_valid']
        jvalid = sensor_masks['joint_sensor_valid'] & context[..., None]
        gvalid = sensor_masks['root_sensor_valid'] & sensor_masks['feature_valid'] & context
        p = sanitize_valid(p, jvalid, 'predicted joints')
        r = sanitize_valid(r, gvalid, 'predicted pelvis')
        f = sanitize_valid(f, gvalid, 'predicted features')
        joint = self.joint_projection(p)
        global_token = self.global_projection(torch.cat(((r - self.root_mean) / self.root_std, self.feature_norm(f)), -1))
        valid = torch.cat((jvalid, gvalid[..., None]), -1)
        x = torch.cat((joint, global_token.unsqueeze(2)), 2) + self.joint_id + fixed_time_encoding(time_s, 128).unsqueeze(2)
        x = sanitize_valid(x, valid, 'motion embedding')
        for stage in self.stages:
            x = stage(x, valid)
        h = sanitize_valid(self.norm(x), valid, 'H')
        mean = h[:, :, :22].sum(2) / jvalid.sum(2).clamp_min(1).unsqueeze(-1)
        z = torch.cat((mean, h[:, :, 22]), -1)
        return {'H': h, 'Z': z, 'latent_valid': valid, 'common_frame_valid': gvalid & jvalid.all(-1)}


def make_rich_memory(h):
    if h.ndim != 4 or h.shape[1:] != (32, 23, 128):
        raise ValueError('rich memory requires full window H[32,23,128]')
    return torch.cat((h, h[:, :, 22:23].expand_as(h)), -1)


def build_rich_attention_memory(m_raw, time_s, rich_valid):
    if m_raw.shape != (*time_s.shape, 23, 256) or rich_valid.shape != (*time_s.shape, 23):
        raise ValueError('rich memory shape/mask mismatch')
    return sanitize_valid(sanitize_valid(m_raw, rich_valid, 'M_raw') + fixed_time_encoding(time_s, 256).unsqueeze(2), rich_valid, 'M_attn')
