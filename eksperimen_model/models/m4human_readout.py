"""Causal H readout and retrospective full-window M/U query decoders."""
import math
import torch
from torch import nn
from .m4human_motion import fixed_time_encoding, initialize_neural, sanitize_valid
from .m4human_tokenizer import CrossQueryBlock


class KinematicReadoutV2(nn.Module):
    def __init__(self, scales, pelvis_index=0):
        super().__init__()
        self.pelvis_index = pelvis_index
        self.scales = {k: float(scales[k]) for k in ('s_delta_joint', 's_delta_root', 's_v_joint', 's_v_root')}
        if not 0 <= pelvis_index < 22 or any(not math.isfinite(v) or v <= 0 for v in self.scales.values()):
            raise ValueError('physical parameter scales must be positive')
        self.joint_head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 6))
        self.root_head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 6))
        initialize_neural(self)
        for head in (self.joint_head, self.root_head):
            nn.init.zeros_(head[-1].weight[:3])
            nn.init.normal_(head[-1].weight[3:], std=0.01)
        self.register_buffer('nonpelvis', torch.arange(22) != pelvis_index)

    def forward(self, h, p_enc, r_enc, latent_valid):
        h = sanitize_valid(h, latent_valid, 'H readout input')
        with torch.autocast(h.device.type, enabled=False):
            joint = sanitize_valid(self.joint_head(h[:, :, :22].float()), latent_valid[:, :, :22], 'H physical joints')
            root = sanitize_valid(self.root_head(h[:, :, 22].float()), latent_valid[:, :, 22], 'H physical root')
        p = sanitize_valid(p_enc.float(), latent_valid[:, :, :22], 'residual joints')
        r = sanitize_valid(r_enc.float(), latent_valid[:, :, 22], 'residual root')
        return {'P_relative_m': (p + self.scales['s_delta_joint'] * joint[..., :3]) * self.nonpelvis[None, None, :, None],
                'r_m': r + self.scales['s_delta_root'] * root[..., :3],
                'v_relative_mps': self.scales['s_v_joint'] * joint[..., 3:] * self.nonpelvis[None, None, :, None],
                'v_root_mps': self.scales['s_v_root'] * root[..., 3:]}


class _TimeJointDecoder(nn.Module):
    def __init__(self, output_width):
        super().__init__()
        self.joint_id = nn.Parameter(torch.empty(23, 256))
        self.block = CrossQueryBlock()
        self.output = nn.Linear(256, output_width)
        initialize_neural(self)
        nn.init.normal_(self.joint_id, std=0.02)

    def forward(self, memory, memory_valid, query_time_s, query_chunk=0):
        if memory.ndim != 3 or memory.shape[-1] != 256 or query_time_s.ndim != 2 or query_time_s.shape != (memory.shape[0], 32):
            raise ValueError('decoder requires source width256 and full32 query times')
        if memory_valid.shape != memory.shape[:2] or memory_valid.dtype != torch.bool or (query_time_s.diff(dim=1) <= 0).any():
            raise ValueError('decoder mask or chronological query time mismatch')
        if (~memory_valid.any(-1)).any():
            raise ValueError('decoder cannot read an all-empty scientific sample')
        b = memory.shape[0]
        query = (fixed_time_encoding(query_time_s, 256).unsqueeze(2) + self.joint_id).reshape(b, 736, 256)
        query_valid = torch.ones(b, 736, dtype=torch.bool, device=memory.device)
        size = query_chunk or 736
        if size <= 0:
            raise ValueError('query chunk must be nonnegative')
        out = []
        for lo in range(0, 736, size):
            hidden = self.block(query[:, lo:lo + size], memory, memory_valid, query_valid[:, lo:lo + size])
            with torch.autocast(memory.device.type, enabled=False):
                out.append(self.output(hidden.float()))
        return torch.cat(out, 1).reshape(b, 32, 23, -1).float()


class FullHFidelityDecoderV1(_TimeJointDecoder):
    def __init__(self):
        super().__init__(128)


class TokenKinematicReadoutV1(_TimeJointDecoder):
    def __init__(self, scales, pelvis_index=0):
        super().__init__(6)
        self.scales = {k: float(scales[k]) for k in ('s_v_joint', 's_v_root')}
        if not 0 <= pelvis_index < 22 or any(not math.isfinite(v) or v <= 0 for v in self.scales.values()):
            raise ValueError('physical scales must be positive')
        self.register_buffer('nonpelvis', torch.arange(22) != pelvis_index)

    def forward(self, memory, memory_valid, query_time_s, query_chunk=0):
        raw = super().forward(memory, memory_valid, query_time_s, query_chunk)
        joint, root = raw[:, :, :22], raw[:, :, 22]
        return {'P_relative_m': joint[..., :3] * self.nonpelvis[None, None, :, None],
                'r_m': root[..., :3],
                'v_relative_mps': self.scales['s_v_joint'] * joint[..., 3:] * self.nonpelvis[None, None, :, None],
                'v_root_mps': self.scales['s_v_root'] * root[..., 3:]}


def fidelity_loss(predicted_norm, h, valid, mean, std):
    target = sanitize_valid(h.detach().float(), valid, 'fidelity H')
    predicted = sanitize_valid(predicted_norm.float(), valid, 'fidelity prediction')
    if mean.shape != (128,) or std.shape != (128,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or std.min() < 1e-3:
        raise ValueError('H standard deviation below registered floor')
    error = ((predicted - (target - mean) / std)[valid]).square()
    return {'loss': error.mean() if error.numel() else predicted.sum() * 0., 'count': error.numel(), 'sum': error.sum()}


def clip_compression_gradients(compressor, fidelity, auxiliary, max_norm=1.0):
    """Separate groups preserve zero auxiliary attachment on C_base after clipping."""
    groups = {'compressor_fidelity': list(compressor.parameters()) + list(fidelity.parameters()),
              'auxiliary': list(auxiliary.parameters())}
    return {name: float(nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True))
            for name, parameters in groups.items()}
