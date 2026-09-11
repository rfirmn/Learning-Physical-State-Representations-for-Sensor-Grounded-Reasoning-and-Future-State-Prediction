import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

def farthest_point_sampling(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Pure PyTorch Batched Farthest Point Sampling (FPS).
    xyz: (B, N, 3) point coordinates
    npoint: number of sampled centers
    Returns: (B, npoint) indices
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def knn_group(nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch k-Nearest Neighbors grouping.
    xyz: (B, N, 3) original points
    new_xyz: (B, G, 3) center points
    Returns:
        grouped_xyz: (B, G, nsample, 3) normalized local patches
        grouped_idx: (B, G, nsample)
    """
    # Pairwise distance: (B, G, N)
    dist = torch.cdist(new_xyz, xyz)
    _, idx = torch.topk(dist, k=nsample, dim=-1, largest=False, sorted=True) # (B, G, nsample)

    B, G, K = idx.shape
    idx_expanded = idx.view(B, G * K, 1).expand(-1, -1, 3)
    grouped_xyz = torch.gather(xyz, 1, idx_expanded).view(B, G, K, 3)

    # Normalize patch relative to center
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)
    return grouped_xyz_norm, idx


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 6, qkv_bias: bool = False, qk_scale: Optional[float] = None, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4., qkv_bias: bool = False, qk_scale: Optional[float] = None, drop: float = 0., attn_drop: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MiniPointNetPatchEmbed(nn.Module):
    """
    Mini-PointNet Patch Embedding matching module.MAE_encoder.encoder in pretrain.pth.
    Maps local patch (B*G, 3, K) to token embedding (B*G, embed_dim).
    """
    def __init__(self, in_channel: int = 3, out_dim: int = 384):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv1d(in_channel, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, out_dim, 1)
        )

    def forward(self, patch_xyz: torch.Tensor) -> torch.Tensor:
        # patch_xyz: (B*G, K, 3)
        x = patch_xyz.transpose(1, 2) # (B*G, 3, K)
        feat1 = self.first_conv(x)    # (B*G, 256, K)
        feat1_max = torch.max(feat1, 2, keepdim=True)[0].repeat(1, 1, feat1.shape[2]) # (B*G, 256, K)
        feat_cat = torch.cat([feat1, feat1_max], dim=1) # (B*G, 512, K)
        feat2 = self.second_conv(feat_cat) # (B*G, out_dim, K)
        token = torch.max(feat2, 2)[0]     # (B*G, out_dim)
        return token


class PointMAE(nn.Module):
    """
    Pure PyTorch implementation of Point-MAE (ECCV 2022).
    Fully compatible with ShapeNet pretrained checkpoint (models/Point-MAE/pretrain.pth).
    """
    def __init__(
        self,
        num_points: int = 128,
        num_groups: int = 16,
        group_size: int = 16,
        mask_ratio: float = 0.6,
        embed_dim: int = 384,
        depth: int = 12,
        decoder_depth: int = 4,
        num_heads: int = 6
    ):
        super().__init__()
        self.num_points = num_points
        self.num_groups = num_groups
        self.group_size = group_size
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim

        # Patch Embedding
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

        # Decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )
        self.MAE_decoder = nn.ModuleDict({
            'blocks': nn.ModuleList([
                Block(dim=embed_dim, num_heads=num_heads, qkv_bias=True)
                for _ in range(decoder_depth)
            ]),
            'norm': nn.LayerNorm(embed_dim)
        })
        self.increase_dim = nn.Sequential(
            nn.Linear(embed_dim, group_size * 3)
        )

        nn.init.normal_(self.mask_token, std=.02)

    def group_and_sample(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # FPS Center Sampling
        center_idx = farthest_point_sampling(xyz, self.num_groups) # (B, G)
        B, N, _ = xyz.shape
        centers = torch.gather(xyz, 1, center_idx.unsqueeze(-1).expand(-1, -1, 3)) # (B, G, 3)

        # kNN Grouping
        patches_norm, _ = knn_group(self.group_size, xyz, centers) # (B, G, K, 3)
        return centers, patches_norm, center_idx

    def random_masking(self, centers: torch.Tensor, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Random masking of patches.
        """
        B, G, C = tokens.shape
        len_keep = int(G * (1 - self.mask_ratio))
        noise = torch.rand(B, G, device=tokens.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        tokens_vis = torch.gather(tokens, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, C))
        centers_vis = torch.gather(centers, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, 3))

        return tokens_vis, centers_vis, ids_restore, ids_keep

    def forward_encoder(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Grouping and Sampling
        centers, patches_norm, _ = self.group_and_sample(xyz)
        B, G, K, _ = patches_norm.shape

        # 2. Patch Embedding
        patches_flat = patches_norm.view(B * G, K, 3)
        tokens = self.MAE_encoder['encoder'](patches_flat).view(B, G, self.embed_dim)

        # 3. Masking
        tokens_vis, centers_vis, ids_restore, ids_keep = self.random_masking(centers, tokens)

        # 4. Positional Embedding
        pos_vis = self.MAE_encoder['pos_embed'](centers_vis)
        x = tokens_vis + pos_vis

        # 5. Transformer Encoder
        for blk in self.MAE_encoder['blocks']['blocks']:
            x = blk(x)
        x = self.MAE_encoder['norm'](x)

        return x, centers, centers_vis, patches_norm, ids_restore

    def forward_decoder(self, x_vis: torch.Tensor, centers: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        B, G, _ = centers.shape
        len_vis = x_vis.shape[1]
        len_mask = G - len_vis

        # Append mask tokens
        mask_tokens = self.mask_token.expand(B, len_mask, -1)
        x_all = torch.cat([x_vis, mask_tokens], dim=1)
        x_all = torch.gather(x_all, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, self.embed_dim))

        # Add decoder positional embedding
        dec_pos = self.decoder_pos_embed(centers)
        x = x_all + dec_pos

        for blk in self.MAE_decoder['blocks']:
            x = blk(x)
        x = self.MAE_decoder['norm'](x)

        # Reconstruct coordinates
        pred_patches = self.increase_dim(x) # (B, G, K * 3)
        pred_patches = pred_patches.view(B, G, self.group_size, 3)
        return pred_patches

    def forward(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_vis, centers, centers_vis, target_patches, ids_restore = self.forward_encoder(xyz)
        pred_patches = self.forward_decoder(x_vis, centers, ids_restore)
        return pred_patches, target_patches
