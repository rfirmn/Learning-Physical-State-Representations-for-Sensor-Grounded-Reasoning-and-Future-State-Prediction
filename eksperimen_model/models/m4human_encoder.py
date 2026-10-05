"""Scratch radar set encoder; units, masks and chronology follow m4human_kinetok_v3."""
import torch
from torch import nn
import torch.nn.functional as F


def block(n, m):
    return nn.Sequential(nn.Linear(n, m), nn.LayerNorm(m, eps=1e-5), nn.GELU())


class M4HumanSetEncoderV2(nn.Module):
    def __init__(self, pelvis_index=0, normalizer=None):
        super().__init__()
        if not 0 <= pelvis_index < 22:
            raise ValueError('pelvis_index must be audited in joint map22')
        self.pelvis_index = pelvis_index
        self.point_stem = nn.Sequential(block(4, 64), block(64, 128), block(128, 256))
        self.frame_feature = block(512, 256)
        self.temporal = nn.Sequential(block(1032, 512), block(512, 256))
        self.pose_head = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 63))
        self.root_head = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3))
        self.register_buffer('input_mean', torch.zeros(4))
        self.register_buffer('input_std', torch.ones(4))
        if normalizer is not None:
            self.set_normalizer(normalizer)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        for head in (self.pose_head, self.root_head):
            nn.init.normal_(head[-1].weight, std=.01)
        self.register_buffer('non_pelvis', torch.tensor([j for j in range(22) if j != pelvis_index]))

    def set_normalizer(self, normalizer):
        mean = torch.as_tensor(normalizer['mean'], dtype=torch.float32)
        std = torch.as_tensor(normalizer['std'], dtype=torch.float32)
        if mean.shape != (4,) or std.shape != (4,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError('invalid locked input normalizer')
        self.input_mean.copy_(mean)
        self.input_std.copy_(std)

    def forward(self, rpc_m, point_mask, time_s):
        if rpc_m.ndim != 4 or rpc_m.shape[1:] != (4, 512, 4) or point_mask.shape != rpc_m.shape[:-1] or point_mask.dtype != torch.bool or time_s.shape != rpc_m.shape[:2]:
            raise ValueError('expected RPC[B,4,512,4], boolean mask[B,4,512], seconds[B,4]')
        if not torch.isfinite(rpc_m[point_mask]).all():
            raise ValueError('nonfinite valid radar return')
        x = torch.where(point_mask[..., None], rpc_m.float(), 0)
        intensity = x[..., 3:4]
        x = torch.cat((x[..., :3], intensity.sign() * intensity.abs().log1p()), -1)
        x = (x - self.input_mean) / self.input_std
        x = torch.where(point_mask[..., None], x, 0)
        h = self.point_stem(x)
        count = point_mask.sum(-1)
        frame_valid = count > 0
        mean = torch.where(point_mask[..., None], h, 0).sum(-2) / count.clamp_min(1)[..., None]
        maximum = h.masked_fill(~point_mask[..., None], -torch.inf).amax(-2)
        maximum = torch.where(frame_valid[..., None], maximum, 0)
        g = self.frame_feature(torch.cat((maximum, mean), -1))
        g = torch.where(frame_valid[..., None], g, 0)
        times = time_s.double()
        time_valid = torch.isfinite(times).all(-1) & (times.diff(dim=-1) > 0).all(-1)
        rel_time = (times - times[:, -1:]) * 12.
        rel_time = torch.where(torch.isfinite(rel_time), rel_time, 0).float()
        f = self.temporal(torch.cat((g.flatten(1), rel_time, count.float() / 512.), -1))
        valid = frame_valid.all(-1) & time_valid
        f = torch.where(valid[:, None], f, 0)
        pose = rpc_m.new_zeros((len(rpc_m), 22, 3), dtype=f.dtype)
        pose[:, self.non_pelvis] = self.pose_head(f).reshape(-1, 21, 3)
        root = self.root_head(f)
        pose = torch.where(valid[:, None, None], pose, 0).float()
        root = torch.where(valid[:, None], root, 0).float()
        return {'P_enc_relative_m': pose, 'r_enc_m': root, 'f_enc': f,
                'sensor_valid': valid, 'root_sensor_valid': valid,
                'joint_sensor_valid': valid[:, None].expand(-1, 22), 'feature_valid': valid,
                'context_time_valid': time_valid}


def encoder_loss(prediction, target, pelvis_index=0, beta=.05):
    valid = prediction['sensor_valid']
    root_mask = target['annotation_root_valid'].bool() & valid
    pose_mask = target['annotation_joint_valid'].bool() & root_mask[:, None]
    pose_mask = pose_mask.clone()
    pose_mask[:, pelvis_index] = False
    terms = {}
    for name, key, target_key, mask in [('pose', 'P_enc_relative_m', 'joint_relative_m', pose_mask), ('root', 'r_enc_m', 'root_m', root_mask)]:
        p, t = prediction[key].float()[mask], target[target_key].float()[mask]
        if not torch.isfinite(t).all():
            raise ValueError('nonfinite annotation declared valid')
        terms[name] = F.smooth_l1_loss(p, t, beta=beta, reduction='mean') if p.numel() else prediction[key].sum() * 0
        terms[name + '_count'] = p.numel()
    terms['total'] = terms['pose'] + terms['root']
    return terms
