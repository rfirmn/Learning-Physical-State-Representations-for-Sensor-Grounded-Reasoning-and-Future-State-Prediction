import torch
import torch.nn as nn
from typing import Tuple, Optional
from .point_mae import farthest_point_sampling, knn_group, MiniPointNetPatchEmbed, Block

class JointQueryPoseHead(nn.Module):
    """
    Cross-Attention Pose Head: 17 learnable joint queries attend to G patch tokens.
    Each query independently attends to spatially relevant patches, allowing peripheral
    joints (wrists, elbows, ankles) to focus on distinct local patch tokens.
    """
    def __init__(
        self,
        embed_dim: int = 384,
        num_joints: int = 17,
        num_heads: int = 6,
        depth: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_joints = num_joints
        self.embed_dim = embed_dim

        # 17 learnable queries for each joint
        self.joint_queries = nn.Parameter(torch.randn(1, num_joints, embed_dim) * 0.02)

        # Cross-Attention Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)

        # Per-joint 3D coordinate regressor
        self.regressor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        patch_tokens: (B, G, embed_dim)
        Returns:
            pred_joints: (B, num_joints, 3) in meters
        """
        B = patch_tokens.shape[0]
        queries = self.joint_queries.expand(B, -1, -1) # (B, num_joints, embed_dim)
        decoded = self.decoder(tgt=queries, memory=patch_tokens) # (B, num_joints, embed_dim)
        pred_joints = self.regressor(decoded) # (B, num_joints, 3)
        return pred_joints


class PointMAEPoseEstimator(nn.Module):
    """
    Sensor Encoder Backbone + 3D Human Pose Estimation Head (Tahap 2).
    Extracts Physical State Representation (Z_t) and regresses 17 3D joint keypoints.
    Supports both MLP Head and Joint-Query Cross-Attention Head, as well as
    3-channel (xyz) and 5-channel (xyz + doppler + snr) radar input.
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
        in_channels: int = 3,
        freeze_encoder: bool = False,
        pose_head_type: str = "cross_attention",
        pose_head_depth: int = 2,
        pose_head_dropout: float = 0.2,
        drop_path_rate: float = 0.1
    ):
        super().__init__()
        self.num_points = num_points
        self.num_groups = num_groups
        self.group_size = group_size
        self.embed_dim = embed_dim
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.pose_head_type = pose_head_type

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Backbone: Point-MAE Transformer Encoder
        self.MAE_encoder = nn.ModuleDict({
            'encoder': MiniPointNetPatchEmbed(in_channel=in_channels, out_dim=embed_dim),
            'pos_embed': nn.Sequential(
                nn.Linear(3, 128),
                nn.GELU(),
                nn.Linear(128, embed_dim)
            ),
            'blocks': nn.ModuleDict({
                'blocks': nn.ModuleList([
                    Block(dim=embed_dim, num_heads=num_heads, qkv_bias=True, drop_path=dpr[i])
                    for i in range(depth)
                ])
            }),
            'norm': nn.LayerNorm(embed_dim)
        })

        # 3D Skeleton Pose Estimation Head
        if pose_head_type == "cross_attention":
            self.pose_head = JointQueryPoseHead(
                embed_dim=embed_dim,
                num_joints=num_joints,
                num_heads=num_heads,
                depth=pose_head_depth,
                dropout=pose_head_dropout
            )
        else:
            self.pose_head = nn.Sequential(
                nn.Linear(embed_dim, 512),
                nn.GELU(),
                nn.LayerNorm(512),
                nn.Dropout(pose_head_dropout),
                nn.Linear(512, num_joints * 3)
            )

        if freeze_encoder:
            self.freeze_backbone(True)

    def freeze_backbone(self, freeze: bool = True):
        """Freezes/unfreezes the Point-MAE backbone encoder parameters."""
        for param in self.MAE_encoder.parameters():
            param.requires_grad = not freeze
        print(f"[PointMAEPoseEstimator] Backbone encoder frozen = {freeze}")

    def extract_features(self, pts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts Physical State Representation Z_t.
        pts: (B, N, C), C >= 3
        Returns:
            z_t: (B, embed_dim) Global Physical State Vector
            tokens: (B, G, embed_dim) Local token representations
        """
        xyz = pts[:, :, :3]
        extra_features = pts[:, :, 3:] if pts.shape[-1] > 3 else None

        # 1. FPS Center Sampling (spatial in 3D)
        center_idx = farthest_point_sampling(xyz, self.num_groups)
        B, N, _ = xyz.shape
        centers = torch.gather(xyz, 1, center_idx.unsqueeze(-1).expand(-1, -1, 3))

        # 2. kNN Grouping
        patches_norm, _ = knn_group(self.group_size, xyz, centers, extra_features=extra_features)
        B, G, K, C = patches_norm.shape

        # 3. Patch Embedding
        patches_flat = patches_norm.view(B * G, K, C)
        tokens = self.MAE_encoder['encoder'](patches_flat).view(B, G, self.embed_dim)

        # 4. Positional Embedding (based on 3D spatial center coordinates)
        pos = self.MAE_encoder['pos_embed'](centers)
        x = tokens + pos

        # 5. Transformer Encoder
        for blk in self.MAE_encoder['blocks']['blocks']:
            x = blk(x)
        x = self.MAE_encoder['norm'](x) # (B, G, embed_dim)

        # 6. Global Physical State Representation Z_t (mean pool across patch tokens)
        z_t = torch.mean(x, dim=1) # (B, embed_dim)
        return z_t, x

    def forward(self, pts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass predicting 3D human pose keypoints.
        pts: (B, num_points, C), C >= 3
        Returns:
            pred_joints: (B, num_joints, 3) in meters
            z_t: (B, embed_dim) Physical State Representation
        """
        z_t, patch_tokens = self.extract_features(pts)
        if self.pose_head_type == "cross_attention":
            pred_joints = self.pose_head(patch_tokens)
        else:
            pred_flat = self.pose_head(z_t)
            pred_joints = pred_flat.view(-1, self.num_joints, 3)
        return pred_joints, z_t
