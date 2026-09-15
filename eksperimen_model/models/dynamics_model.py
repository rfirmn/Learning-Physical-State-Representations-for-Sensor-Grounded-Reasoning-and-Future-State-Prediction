import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class TemporalSelfAttentionBlock(nn.Module):
    """
    Pre-LayerNorm Transformer Block with Temporal Multi-Head Attention and DropPath.
    """
    def __init__(self, dim: int = 384, num_heads: int = 6, dim_feedforward: int = 1024, dropout: float = 0.1, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(dropout)
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN Self-Attention
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + self.drop_path1(attn_out)

        # Pre-LN MLP
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class HorizonQueryDecoderBlock(nn.Module):
    """
    Cross-Attention Decoder Block: Future horizon queries attend to past encoder memory.
    """
    def __init__(self, dim: int = 384, num_heads: int = 6, dim_feedforward: int = 1024, dropout: float = 0.1, drop_path: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.drop_path_self = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm_cross_q = nn.LayerNorm(dim)
        self.norm_cross_mem = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.drop_path_cross = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(dropout)
        )
        self.drop_path_mlp = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # 1. Self-Attention among horizon queries
        q_norm = self.norm_q(tgt)
        self_out, _ = self.self_attn(q_norm, q_norm, q_norm)
        tgt = tgt + self.drop_path_self(self_out)

        # 2. Cross-Attention: queries attend to historical memory
        cross_out, _ = self.cross_attn(query=self.norm_cross_q(tgt),
                                       key=self.norm_cross_mem(memory),
                                       value=self.norm_cross_mem(memory))
        tgt = tgt + self.drop_path_cross(cross_out)

        # 3. Feedforward
        tgt = tgt + self.drop_path_mlp(self.mlp(self.norm_mlp(tgt)))
        return tgt


class TemporalTransformerDynamics(nn.Module):
    """
    State-of-the-Art Latent Dynamics Predictor:
    Takes (B, T_in, embed_dim) historical physical latents and predicts (B, T_out, embed_dim) future latents
    using an Encoder-Decoder Horizon-Query architecture.
    """
    def __init__(
        self,
        embed_dim: int = 384,
        t_in: int = 16,
        t_out: int = 8,
        num_layers: int = 4,
        num_heads: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.t_in = t_in
        self.t_out = t_out

        # Learnable Temporal Positional Embeddings
        self.pos_embed_in = nn.Parameter(torch.randn(1, t_in, embed_dim) * 0.02)
        self.pos_embed_out = nn.Parameter(torch.randn(1, t_out, embed_dim) * 0.02)

        # Learnable Horizon Queries (representing future time-steps t+1 .. t+h)
        self.horizon_queries = nn.Parameter(torch.randn(1, t_out, embed_dim) * 0.02)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]

        # Temporal Encoder Blocks
        self.encoder_blocks = nn.ModuleList([
            TemporalSelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                drop_path=dpr[i]
            ) for i in range(num_layers)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # Horizon Query Decoder Blocks (2 layers)
        self.decoder_blocks = nn.ModuleList([
            HorizonQueryDecoderBlock(
                dim=embed_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                drop_path=drop_path_rate
            ) for _ in range(2)
        ])
        self.decoder_norm = nn.LayerNorm(embed_dim)

        # Final projection head with residual connection to the last observed state
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, hist_z: torch.Tensor) -> torch.Tensor:
        """
        hist_z: (B, T_in, embed_dim)
        Returns:
            pred_future_z: (B, T_out, embed_dim)
        """
        B = hist_z.shape[0]

        # 1. Add temporal positional embedding to historical sequence
        x = hist_z + self.pos_embed_in  # (B, T_in, embed_dim)

        # 2. Encode past history via Temporal Self-Attention
        for blk in self.encoder_blocks:
            x = blk(x)
        memory = self.encoder_norm(x)  # (B, T_in, embed_dim)

        # 3. Initialize future horizon queries
        queries = self.horizon_queries.expand(B, -1, -1) + self.pos_embed_out  # (B, T_out, embed_dim)

        # 4. Decode future horizons via Cross-Attention
        out = queries
        for blk in self.decoder_blocks:
            out = blk(out, memory)
        out = self.decoder_norm(out)

        # 5. Output projection + anchor residual from last observed state Z_t
        last_observed = hist_z[:, -1:, :]  # (B, 1, embed_dim)
        pred_delta = self.out_proj(out)
        pred_future_z = last_observed + pred_delta  # Residual velocity formulation

        return pred_future_z


class ResidualGRUDynamics(nn.Module):
    """
    Baseline Multi-Layer Gated Recurrent Unit (GRU) with residual anchor connection.
    """
    def __init__(
        self,
        embed_dim: int = 384,
        t_in: int = 16,
        t_out: int = 8,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.t_in = t_in
        self.t_out = t_out

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward := 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, t_out * embed_dim)
        )

    def forward(self, hist_z: torch.Tensor) -> torch.Tensor:
        B = hist_z.shape[0]
        _, h_n = self.gru(hist_z)  # h_n: (num_layers, B, hidden_dim)
        last_hidden = h_n[-1]      # (B, hidden_dim)

        pred_flat = self.out_proj(last_hidden)  # (B, t_out * embed_dim)
        pred_delta = pred_flat.view(B, self.t_out, self.embed_dim)

        last_observed = hist_z[:, -1:, :]  # (B, 1, embed_dim)
        return last_observed + pred_delta


class CompositeDynamicsLoss(nn.Module):
    """
    Composite Latent Dynamics Loss:
    L_total = L_MSE + lambda_cos * L_cos + lambda_vel * L_vel
    """
    def __init__(self, loss_cos_weight: float = 0.2, loss_vel_weight: float = 0.1):
        super().__init__()
        self.loss_cos_weight = loss_cos_weight
        self.loss_vel_weight = loss_vel_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. Latent Mean Squared Error (per-dimension MSE)
        mse_loss = F.mse_loss(pred, target)

        # 2. Cosine Directional Alignment Loss
        cos_sim = F.cosine_similarity(pred, target, dim=-1)  # (B, T_out)
        cos_loss = (1.0 - cos_sim).mean()

        # 3. Temporal Velocity Continuity Loss
        diff_pred = pred[:, 1:] - pred[:, :-1]          # (B, T_out - 1, D)
        diff_target = target[:, 1:] - target[:, :-1]    # (B, T_out - 1, D)
        vel_loss = F.mse_loss(diff_pred, diff_target) if pred.shape[1] > 1 else torch.tensor(0.0, device=pred.device)

        total_loss = mse_loss + self.loss_cos_weight * cos_loss + self.loss_vel_weight * vel_loss

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_mse": float(mse_loss.item()),
            "loss_cos": float(cos_loss.item()),
            "loss_vel": float(vel_loss.item()),
            "cos_sim": float(cos_sim.mean().item())
        }

        return total_loss, metrics
