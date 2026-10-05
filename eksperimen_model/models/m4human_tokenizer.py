"""Chronological continuous token learning; a single shared query for all budgets."""
import torch
from torch import nn
from .m4human_motion import build_rich_attention_memory, fixed_time_encoding, guarded_attention, initialize_neural, sanitize_valid


class CrossQueryBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_norm, self.memory_norm = nn.LayerNorm(256), nn.LayerNorm(256)
        self.attention = nn.MultiheadAttention(256, 4, dropout=0., batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 256))
        self.final_norm = nn.LayerNorm(256)

    def forward(self, query, memory, memory_valid, query_valid, blocked=None):
        query = sanitize_valid(query, query_valid, 'raw cross query')
        memory = sanitize_valid(memory, memory_valid, 'raw cross memory')
        q = sanitize_valid(self.query_norm(query), query_valid, 'cross query')
        m = sanitize_valid(self.memory_norm(memory), memory_valid, 'cross memory')
        y = query + guarded_attention(self.attention, q, m, memory_valid, query_valid, blocked)
        y = sanitize_valid(y, query_valid, 'cross residual')
        return sanitize_valid(self.final_norm(y + self.ffn(y)), query_valid, 'cross final')


class KinematicTokenLearnerV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.q0 = nn.Parameter(torch.empty(256))
        self.block = CrossQueryBlock()
        initialize_neural(self)
        nn.init.normal_(self.q0, std=0.02)

    def forward(self, m_raw, time_s, rich_valid, K=16):
        return self.from_attention_memory(build_rich_attention_memory(m_raw, time_s, rich_valid), time_s, rich_valid, K)

    def from_attention_memory(self, memory, time_s, rich_valid, K=16):
        if K not in (8, 16, 32) or time_s.ndim != 2 or time_s.shape[1] != 32:
            raise ValueError('predefined budgets are K8/K16/K32 on exactly 32 anchors')
        if memory.shape != (*time_s.shape, 23, 256) or rich_valid.shape != (*time_s.shape, 23) or rich_valid.dtype != torch.bool:
            raise ValueError('tokenizer memory shape/mask mismatch')
        if (time_s.diff(dim=1) <= 0).any():
            raise ValueError('token bins require strictly increasing source time')
        b, t, j, _ = memory.shape
        if (rich_valid.any(-1) != rich_valid.all(-1)).any():
            raise ValueError('stage C requires common whole-frame support')
        frame_valid = rich_valid.all(-1)
        if (~frame_valid.any(-1)).any():
            raise ValueError('all-empty scientific window')
        start = torch.arange(K, device=memory.device) * 32 // K
        end = (torch.arange(K, device=memory.device) + 1) * 32 // K
        bin_time = 0.5 * (time_s[:, start] + time_s[:, end - 1])
        indices = torch.arange(t, device=memory.device)
        in_bin = (indices[None] >= start[:, None]) & (indices[None] < end[:, None])
        counts = (frame_valid[:, None] & in_bin[None]).sum(-1)
        token_valid = counts > 0
        blocked = (~in_bin).repeat_interleave(j, -1)[None].expand(b, K, t * j)
        query = self.q0 + fixed_time_encoding(bin_time, 256, time_s[:, 0])
        u = self.block(query, memory.reshape(b, t * j, 256), rich_valid.reshape(b, t * j), token_valid, blocked)
        return {'U': u, 'token_valid': token_valid, 'token_mask': token_valid, 'bin_time_s': bin_time,
                'bin_start': start, 'bin_end': end, 'bin_counts': counts}
