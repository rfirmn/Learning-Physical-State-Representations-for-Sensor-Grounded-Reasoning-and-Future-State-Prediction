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
