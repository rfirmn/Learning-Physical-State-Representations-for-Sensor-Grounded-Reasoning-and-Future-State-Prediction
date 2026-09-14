import torch
import torch.nn as nn

class MPJPELoss(nn.Module):
    """
    Mean Per-Joint Position Error (MPJPE).
    Computes the mean Euclidean distance (in meters) between predicted 3D keypoints and ground truth.
    Loss = (1 / K) * sum_{k=1}^K || pred_k - target_k ||_2
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target shape: (B, num_joints, 3)
        diff = pred - target
        dist = torch.norm(diff, p=2, dim=-1) # (B, num_joints)

        if self.reduction == 'mean':
            return dist.mean()
        elif self.reduction == 'sum':
            return dist.sum()
        return dist


class BoneLengthLoss(nn.Module):
    r"""
    Bone Length Consistency Loss (Anatomical Kinematic Prior).
    Prevents skeleton deformation by enforcing that predicted bone lengths match
    ground-truth anatomical bone lengths.
    Loss = mean_{b \in Bones} | ||pred_u - pred_v||_2 - ||gt_u - gt_v||_2 |
    """
    BONE_PAIRS = [
        (0, 7), (7, 8), (8, 9), (9, 10),   # Spine & Neck (Pelvis-Spine-Thorax-Neck-Head)
        (0, 1), (1, 2), (2, 3),            # Right Leg (Pelvis-Hip-Knee-Ankle)
        (0, 4), (4, 5), (5, 6),            # Left Leg (Pelvis-Hip-Knee-Ankle)
        (8, 11), (11, 12), (12, 13),       # Left Arm (Thorax-Shoulder-Elbow-Wrist)
        (8, 14), (14, 15), (15, 16)        # Right Arm (Thorax-Shoulder-Elbow-Wrist)
    ]

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, 17, 3)
        bone_diffs = []
        for u, v in self.BONE_PAIRS:
            pred_bone = torch.norm(pred[:, u, :] - pred[:, v, :], p=2, dim=-1) # (B,)
            gt_bone = torch.norm(target[:, u, :] - target[:, v, :], p=2, dim=-1) # (B,)
            bone_diffs.append(torch.abs(pred_bone - gt_bone))

        loss = torch.stack(bone_diffs, dim=-1).mean()
        return loss


class WeightedMPJPELoss(nn.Module):
    """
    Weighted Mean Per-Joint Position Error.
    Assigns higher weight to high-error peripheral joints (wrists, elbows, ankles)
    to force the optimizer to allocate gradient capacity to extremities.
    """
    DEFAULT_WEIGHTS = {
        0: 1.0,  # Pelvis
        1: 0.9, 2: 0.9, 3: 1.1,   # R_Hip, R_Knee, R_Ankle
        4: 0.9, 5: 0.9, 6: 1.1,   # L_Hip, L_Knee, L_Ankle
        7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0,  # Spine, Thorax, Neck, Head
        11: 1.1, 12: 1.4, 13: 1.6, # L_Shoulder, L_Elbow, L_Wrist
        14: 1.1, 15: 1.4, 16: 1.6  # R_Shoulder, R_Elbow, R_Wrist
    }

    def __init__(self, weights: torch.Tensor = None):
        super().__init__()
        if weights is None:
            w = [self.DEFAULT_WEIGHTS.get(i, 1.0) for i in range(17)]
            weights = torch.tensor(w, dtype=torch.float32)
            # Normalize so mean weight is 1.0
            weights = weights / weights.mean()
        self.register_buffer('weights', weights.view(1, 17))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, 17, 3)
        diff = pred - target
        dist = torch.norm(diff, p=2, dim=-1) # (B, 17)
        weighted_dist = dist * self.weights.to(dist.device)
        return weighted_dist.mean()


class CompositePoseLoss(nn.Module):
    """
    Composite Loss for 3D Pose Estimation:
    Loss = alpha * WeightedMPJPELoss + beta * BoneLengthLoss
    """
    def __init__(self, alpha: float = 1.0, beta: float = 0.5, use_weighted: bool = True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mpjpe_loss = WeightedMPJPELoss() if use_weighted else MPJPELoss()
        self.bone_loss = BoneLengthLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        loss_mpjpe = self.mpjpe_loss(pred, target)
        loss_bone = self.bone_loss(pred, target)
        total_loss = self.alpha * loss_mpjpe + self.beta * loss_bone

        raw_mpjpe_mm = (torch.norm(pred - target, p=2, dim=-1).mean() * 1000.0).item()

        return total_loss, {
            'loss': total_loss.item(),
            'loss_mpjpe': loss_mpjpe.item(),
            'loss_bone': loss_bone.item(),
            'mpjpe_mm': raw_mpjpe_mm
        }


class ChamferDistanceLoss(nn.Module):
    """
    Pure PyTorch implementation of Chamfer Distance for point cloud reconstruction.
    Operates on tensors of shape (B, N, 3) and (B, M, 3).
    No compiled C++/CUDA extensions required.
    """
    def __init__(self, metric: str = 'l2'):
        super().__init__()
        self.metric = metric

    def forward(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        # p1: (B, N, 3), p2: (B, M, 3)
        # Pairwise distance matrix: (B, N, M)
        p1_sq = torch.sum(p1 ** 2, dim=-1, keepdim=True) # (B, N, 1)
        p2_sq = torch.sum(p2 ** 2, dim=-1, keepdim=True) # (B, M, 1)
        p1_p2 = torch.bmm(p1, p2.transpose(1, 2))       # (B, N, M)

        dist_matrix_sq = p1_sq - 2.0 * p1_p2 + p2_sq.transpose(1, 2)
        dist_matrix_sq = torch.clamp(dist_matrix_sq, min=1e-8)

        if self.metric == 'l1':
            dist_matrix = torch.sqrt(dist_matrix_sq)
            min_dist_1, _ = torch.min(dist_matrix, dim=2) # (B, N)
            min_dist_2, _ = torch.min(dist_matrix, dim=1) # (B, M)
        else: # l2
            min_dist_1, _ = torch.min(dist_matrix_sq, dim=2)
            min_dist_2, _ = torch.min(dist_matrix_sq, dim=1)

        loss = torch.mean(min_dist_1) + torch.mean(min_dist_2)
        return loss
