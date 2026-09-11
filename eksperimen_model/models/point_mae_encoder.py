import torch
import torch.nn as nn
from typing import Tuple, Optional
from .point_mae import farthest_point_sampling, knn_group, MiniPointNetPatchEmbed, Block

class PointMAEPoseEstimator(nn.Module):
    """
    Sensor Encoder Backbone + 3D Human Pose Estimation Head (Tahap 2).
    Extracts Physical State Representation (Z_t) and regresses 17 3D joint keypoints.
    """
    def __init__(
        self,
        num_points: int = 128,
        num_groups: int = 16,
        group_size: int = 16,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        num_joints: int = 17,
        freeze_encoder: bool = False
    ):
        super().__init__()
        self.num_points = num_points
        self.num_groups = num_groups
        self.group_size = group_size
        self.embed_dim = embed_dim
        self.num_joints = num_joints

        # Backbone: Point-MAE Transformer Encoder
        self.MAE_encoder = nn.ModuleDict({
            'encoder': MiniPointNetPatchEmbed(in_channel=3, out_dim=embed_dim),
            'pos_embed': nn.Sequential(
                nn.Linear(3, 128),
                nn.GELU(),
                nn.Linear(128, embed_dim)
            ),
            'blocks': nn.ModuleDict({
                'blocks': nn.ModuleList([
                    Block(dim=embed_dim, num_heads=num_heads, qkv_bias=True)
                    for _ in range(depth)
                ])
            }),
            'norm': nn.LayerNorm(embed_dim)
        })

        # 3D Skeleton Pose Estimation Head (Tahap 2)
        self.pose_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, num_joints * 3)
        )

        if freeze_encoder:
            self.freeze_backbone(True)

    def freeze_backbone(self, freeze: bool = True):
        """Freezes/unfreezes the Point-MAE backbone encoder parameters."""
        for param in self.MAE_encoder.parameters():
            param.requires_grad = not freeze
        print(f"[PointMAEPoseEstimator] Backbone encoder frozen = {freeze}")

    def extract_features(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts Physical State Representation Z_t.
        Returns:
            z_t: (B, embed_dim) Global Physical State Vector
            tokens: (B, G, embed_dim) Local token representations
        """
        # 1. FPS Center Sampling
        center_idx = farthest_point_sampling(xyz, self.num_groups)
        B, N, _ = xyz.shape
        centers = torch.gather(xyz, 1, center_idx.unsqueeze(-1).expand(-1, -1, 3))

        # 2. kNN Grouping
        patches_norm, _ = knn_group(self.group_size, xyz, centers)
        B, G, K, _ = patches_norm.shape

        # 3. Patch Embedding
        patches_flat = patches_norm.view(B * G, K, 3)
        tokens = self.MAE_encoder['encoder'](patches_flat).view(B, G, self.embed_dim)

        # 4. Positional Embedding
        pos = self.MAE_encoder['pos_embed'](centers)
        x = tokens + pos

        # 5. Transformer Encoder
        for blk in self.MAE_encoder['blocks']['blocks']:
            x = blk(x)
        x = self.MAE_encoder['norm'](x) # (B, G, embed_dim)

        # 6. Global Physical State Representation Z_t (mean pool across patch tokens)
        z_t = torch.mean(x, dim=1) # (B, embed_dim)
        return z_t, x

    def forward(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass predicting 3D human pose keypoints.
        xyz: (B, num_points, 3)
        Returns:
            pred_joints: (B, num_joints, 3) in meters
            z_t: (B, embed_dim) Physical State Representation
        """
        z_t, _ = self.extract_features(xyz)
        pred_flat = self.pose_head(z_t) # (B, num_joints * 3)
        pred_joints = pred_flat.view(-1, self.num_joints, 3)
        return pred_joints, z_t
