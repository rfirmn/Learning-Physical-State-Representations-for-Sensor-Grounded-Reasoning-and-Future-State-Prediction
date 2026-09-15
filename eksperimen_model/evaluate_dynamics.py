import os
import sys

# Ensure UTF-8 output in Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
import yaml
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Any, Tuple

sys.path.insert(0, os.path.abspath("."))

from eksperimen_model.models import TemporalTransformerDynamics, ResidualGRUDynamics, PointMAEPoseEstimator
from eksperimen_model.datasets import TemporalPhysicalDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Held-Out Benchmark Evaluation for Latent Dynamics Model")
    parser.add_argument("--checkpoint", type=str, default="eksperimen_model/checkpoints/dynamics/best_dynamics_model.pth",
                        help="Path to dynamics model checkpoint")
    parser.add_argument("--config", type=str, default="eksperimen_model/configs/mmfi_dynamics_best_tuned.yaml",
                        help="Path to configuration YAML")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Evaluation batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Computation device")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Output JSON path for detailed benchmark metrics")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory to save plots and reports")
    return parser.parse_args()


def compute_procrustes_alignment(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Orthogonal Procrustes alignment via SVD. pred, gt: (17, 3)
    """
    mu_pred = np.mean(pred, axis=0)
    mu_gt = np.mean(gt, axis=0)
    pred_c = pred - mu_pred
    gt_c = gt - mu_gt

    h = pred_c.T @ gt_c
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T

    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T

    scale = np.sum(gt_c * (pred_c @ r)) / (np.sum(pred_c ** 2) + 1e-8)
    aligned_pred = scale * (pred_c @ r) + mu_gt
    return aligned_pred


def decode_latent_to_skeleton(pose_head: nn.Module, z_denorm: torch.Tensor) -> torch.Tensor:
    """
    Decodes latent vector Z (B, T_out, 384) into 3D skeleton (B, T_out, 17, 3)
    using the frozen Pose Head from Model Av2.
    """
    B, T, D = z_denorm.shape
    z_flat = z_denorm.view(B * T, D)

    # Expand single global vector to 16 memory slots for Cross-Attention Pose Head
    z_mem = z_flat.unsqueeze(1).expand(-1, 16, -1)  # (B*T, 16, 384)
    with torch.no_grad():
        pred_joints_flat = pose_head(z_mem)  # (B*T, 17, 3)

    return pred_joints_flat.view(B, T, 17, 3)


def plot_horizon_error_progression(horizon_metrics: Dict[str, list], output_path: str):
    """
    Generates a line plot showing error progression across future horizons t+1 .. t+8.
    """
    horizons = [f"t+{k}" for k in range(1, len(horizon_metrics["pred_mpjpe_mm"]) + 1)]

    plt.figure(figsize=(10, 5))
    plt.plot(horizons, horizon_metrics["pred_mpjpe_mm"], marker='o', linewidth=2.2, color='#d62728', label='Dynamics Forecast MPJPE')
    plt.plot(horizons, horizon_metrics["oracle_mpjpe_mm"], marker='s', linewidth=2.0, linestyle='--', color='#2ca02c', label='Oracle Av2 Baseline (Upper Bound)')
    plt.plot(horizons, horizon_metrics["root_rel_mpjpe_mm"], marker='^', linewidth=1.8, color='#1f77b4', label='Root-Relative MPJPE (Articulation)')

    plt.title("Future Horizon Error Progression (t+1 to t+8)", fontsize=13, fontweight='bold')
    plt.xlabel("Forecast Horizon Step (10 Hz / 100ms per step)", fontsize=11)
    plt.ylabel("Mean Per-Joint Position Error (mm)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[Done] Saved horizon error plot -> {output_path}")


def evaluate_dynamics_benchmark(
    dynamics_model: nn.Module,
    pose_estimator: PointMAEPoseEstimator,
    dataset: TemporalPhysicalDataset,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    dynamics_model.eval()
    pose_estimator.eval()
    pose_head = pose_estimator.pose_head

    all_mse_per_horizon = []
    all_cos_per_horizon = []
    all_pred_mpjpe_per_horizon = []
    all_oracle_mpjpe_per_horizon = []
    all_root_rel_mpjpe_per_horizon = []
    all_pelvis_err_per_horizon = []

    subject_records = {}

    print(f"\n[Evaluating Benchmark] Iterating {len(dataset):,} held-out test sequences...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Test Set"):
            hist_z = batch["hist_z"].to(device)                 # (B, T_in, 384)
            target_z = batch["target_z"].to(device)             # (B, T_out, 384)
            target_gt_skel = batch["target_gt_skel"].numpy()    # (B, T_out, 17, 3) (Meters)
            target_pred_skel = batch["target_pred_skel"].numpy() # (B, T_out, 17, 3) (Oracle from Av2)
            subs = batch["sub"]
            acts = batch["act"]

            # 1. Forward pass dynamics model
            pred_z = dynamics_model(hist_z)                     # (B, T_out, 384)

            # 2. Latent metrics (in standardized space)
            mse_h = torch.mean((pred_z - target_z) ** 2, dim=-1).cpu().numpy()  # (B, T_out)
            cos_h = torch.cosine_similarity(pred_z, target_z, dim=-1).cpu().numpy()  # (B, T_out)

            # 3. Denormalize for physical Pose Head probe
            pred_z_denorm = dataset.denormalize_z(pred_z)
            pred_skel = decode_latent_to_skeleton(pose_head, pred_z_denorm).cpu().numpy()  # (B, T_out, 17, 3)

            B, T_out, num_joints, _ = pred_skel.shape

            # Calculate physical joint errors in millimeters
            pred_joint_diff = np.linalg.norm(pred_skel - target_gt_skel, axis=-1) * 1000.0   # (B, T_out, 17) in mm
            oracle_joint_diff = np.linalg.norm(target_pred_skel - target_gt_skel, axis=-1) * 1000.0 # (B, T_out, 17) in mm

            pred_mpjpe_h = np.mean(pred_joint_diff, axis=-1)       # (B, T_out)
            oracle_mpjpe_h = np.mean(oracle_joint_diff, axis=-1)   # (B, T_out)

            # Root-relative joint error (relative to pelvis j=0)
            pred_rel = pred_skel - pred_skel[:, :, 0:1, :]
            gt_rel = target_gt_skel - target_gt_skel[:, :, 0:1, :]
            root_rel_h = np.mean(np.linalg.norm(pred_rel - gt_rel, axis=-1) * 1000.0, axis=-1) # (B, T_out)

            # Pelvis trajectory displacement error (j=0)
            pelvis_err_h = np.linalg.norm(pred_skel[:, :, 0, :] - target_gt_skel[:, :, 0, :], axis=-1) * 1000.0 # (B, T_out)

            all_mse_per_horizon.append(mse_h)
            all_cos_per_horizon.append(cos_h)
            all_pred_mpjpe_per_horizon.append(pred_mpjpe_h)
            all_oracle_mpjpe_per_horizon.append(oracle_mpjpe_h)
            all_root_rel_mpjpe_per_horizon.append(root_rel_h)
            all_pelvis_err_per_horizon.append(pelvis_err_h)

            # Subject-level tracking
            for b in range(B):
                sub_id = subs[b]
                if sub_id not in subject_records:
                    subject_records[sub_id] = {"pred_mpjpe": [], "oracle_mpjpe": [], "root_rel": []}
                subject_records[sub_id]["pred_mpjpe"].append(np.mean(pred_mpjpe_h[b]))
                subject_records[sub_id]["oracle_mpjpe"].append(np.mean(oracle_mpjpe_h[b]))
                subject_records[sub_id]["root_rel"].append(np.mean(root_rel_h[b]))

    all_mse = np.concatenate(all_mse_per_horizon, axis=0)         # (N, T_out)
    all_cos = np.concatenate(all_cos_per_horizon, axis=0)         # (N, T_out)
    all_pred = np.concatenate(all_pred_mpjpe_per_horizon, axis=0) # (N, T_out)
    all_oracle = np.concatenate(all_oracle_mpjpe_per_horizon, axis=0) # (N, T_out)
    all_root_rel = np.concatenate(all_root_rel_mpjpe_per_horizon, axis=0) # (N, T_out)
    all_pelvis = np.concatenate(all_pelvis_err_per_horizon, axis=0) # (N, T_out)

    T_out = all_pred.shape[1]
    horizon_metrics = {
        "pred_mpjpe_mm": [float(np.mean(all_pred[:, k])) for k in range(T_out)],
        "oracle_mpjpe_mm": [float(np.mean(all_oracle[:, k])) for k in range(T_out)],
        "root_rel_mpjpe_mm": [float(np.mean(all_root_rel[:, k])) for k in range(T_out)],
        "pelvis_trajectory_err_mm": [float(np.mean(all_pelvis[:, k])) for k in range(T_out)],
        "latent_mse": [float(np.mean(all_mse[:, k])) for k in range(T_out)],
        "latent_cos_sim": [float(np.mean(all_cos[:, k])) for k in range(T_out)]
    }

    # Overall Summary
    overall_pred_mpjpe = float(np.mean(all_pred))
    overall_oracle_mpjpe = float(np.mean(all_oracle))
    net_dynamics_degradation = overall_pred_mpjpe - overall_oracle_mpjpe
    overall_root_rel_mpjpe = float(np.mean(all_root_rel))
    overall_pelvis_err = float(np.mean(all_pelvis))
    overall_latent_mse = float(np.mean(all_mse))
    overall_latent_cos = float(np.mean(all_cos))

    # Subject-level aggregation (Mean +- Std across subjects)
    sub_pred_means = [float(np.mean(vals["pred_mpjpe"])) for vals in subject_records.values()]
    sub_oracle_means = [float(np.mean(vals["oracle_mpjpe"])) for vals in subject_records.values()]
    sub_root_means = [float(np.mean(vals["root_rel"])) for vals in subject_records.values()]

    sub_pred_mean = float(np.mean(sub_pred_means))
    sub_pred_std = float(np.std(sub_pred_means))

    results = {
        "overall_summary": {
            "pred_mpjpe_mm": overall_pred_mpjpe,
            "oracle_mpjpe_mm": overall_oracle_mpjpe,
            "net_dynamics_degradation_mm": net_dynamics_degradation,
            "root_relative_mpjpe_mm": overall_root_rel_mpjpe,
            "pelvis_trajectory_err_mm": overall_pelvis_err,
            "latent_mse": overall_latent_mse,
            "latent_cos_sim": overall_latent_cos,
            "subject_level_pred_mpjpe": f"{sub_pred_mean:.2f} +- {sub_pred_std:.2f} mm"
        },
        "per_horizon": horizon_metrics,
        "per_subject": {sub: {
            "pred_mpjpe_mm": float(np.mean(v["pred_mpjpe"])),
            "oracle_mpjpe_mm": float(np.mean(v["oracle_mpjpe"])),
            "net_degradation_mm": float(np.mean(v["pred_mpjpe"])) - float(np.mean(v["oracle_mpjpe"]))
        } for sub, v in sorted(subject_records.items())}
    }

    # Print publication-grade summary table
    print("\n" + "=" * 82)
    print("      SCIENTIFIC BENCHMARK RESULTS: HELD-OUT TEST SPLIT (8 UNSEEN SUBJECTS)")
    print("=" * 82)
    print(f" Total Evaluated Sequences   : {len(all_pred):,}")
    print(f" Latent Forecast MSE         : {overall_latent_mse:.4f} (per latent dimension)")
    print(f" Latent Directional Cosine   : {overall_latent_cos:.4f}")
    print("-" * 82)
    print(f" [1] Physical 3D Skeleton Probing (via Frozen Model Av2 Pose Head):")
    print(f"     * Dynamics Forecast MPJPE     : {overall_pred_mpjpe:.2f} mm  [Subject Mean: {sub_pred_mean:.1f} +- {sub_pred_std:.1f} mm]")
    print(f"     * Oracle Av2 Upper Bound      : {overall_oracle_mpjpe:.2f} mm")
    print(f"     * Net Dynamics Degradation (Delta): {net_dynamics_degradation:+.2f} mm")
    print(f"     * Root-Relative MPJPE (Pose)  : {overall_root_rel_mpjpe:.2f} mm")
    print(f"     * Pelvis Global Drift Error   : {overall_pelvis_err:.2f} mm")
    print("-" * 82)
    print(" [2] Horizon Progression Breakdown (t+1 to t+8):")
    print(f"     {'Horizon':<8} | {'Forecast MPJPE':<16} | {'Oracle MPJPE':<14} | {'Net Deg (Delta)':<14} | {'Cosine Sim':<10}")
    print("     " + "-" * 70)
    for k in range(T_out):
        p_m = horizon_metrics["pred_mpjpe_mm"][k]
        o_m = horizon_metrics["oracle_mpjpe_mm"][k]
        deg = p_m - o_m
        c_s = horizon_metrics["latent_cos_sim"][k]
        print(f"     t+{k+1:<5} | {p_m:<16.2f} | {o_m:<14.2f} | {deg:<+14.2f} | {c_s:<10.4f}")
    print("-" * 82)
    print(" [3] Held-Out Subject Breakdown:")
    for sub, stats in results["per_subject"].items():
        print(f"     * {sub}: Forecast = {stats['pred_mpjpe_mm']:.1f} mm | Oracle = {stats['oracle_mpjpe_mm']:.1f} mm (Delta = {stats['net_degradation_mm']:+.1f} mm)")
    print("=" * 82)

    return results


def build_dynamics_model(dyn_cfg: Dict[str, Any], t_in: int = 16, t_out: int = 8) -> nn.Module:
    arch_type = dyn_cfg.get("type", "temporal_transformer")
    embed_dim = dyn_cfg.get("d_model", 384)

    if arch_type == "temporal_transformer":
        model = TemporalTransformerDynamics(
            embed_dim=embed_dim,
            t_in=t_in,
            t_out=t_out,
            num_layers=dyn_cfg.get("num_layers", 4),
            num_heads=dyn_cfg.get("nhead", 6),
            dim_feedforward=dyn_cfg.get("dim_feedforward", 1024),
            dropout=0.0,
            drop_path_rate=0.0
        )
    else:
        model = ResidualGRUDynamics(
            embed_dim=embed_dim,
            t_in=t_in,
            t_out=t_out,
            hidden_dim=dyn_cfg.get("hidden_dim", 512),
            num_layers=dyn_cfg.get("num_layers", 3),
            dropout=0.0
        )
    return model


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Resolve config
    config_path = args.config
    if not os.path.exists(config_path):
        config_path = "eksperimen_model/configs/mmfi_dynamics_v3.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg["dataset"]
    temp_cfg = cfg.get("temporal", {})
    dyn_cfg = cfg.get("dynamics_model", {})
    t_in = temp_cfg.get("t_in", 16)
    t_out = temp_cfg.get("t_out", 8)
    features_dir = ds_cfg["features_output_dir"]

    # 1. Load Test Dataset
    test_ds = TemporalPhysicalDataset(
        features_dir=features_dir,
        split="test",
        t_in=t_in,
        t_out=t_out,
        stride=temp_cfg.get("eval_stride", 4),
        normalize_z=temp_cfg.get("normalize_z", True),
        add_noise=False,
        preload_ram=True
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    # 2. Load Dynamics Model Checkpoint
    dynamics_model = build_dynamics_model(dyn_cfg, t_in=t_in, t_out=t_out).to(device)

    if os.path.exists(args.checkpoint):
        print(f"[Model] Loading dynamics checkpoint from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        dynamics_model.load_state_dict(state_dict)
    else:
        print(f"[Warning] Checkpoint {args.checkpoint} not found! Testing with randomly initialized model.")

    # 3. Load Frozen Model Av2 for Pose Head Probing
    av2_path = ds_cfg.get("encoder_checkpoint", "eksperimen_model/checkpoints/pose_estimation_v2/model_av2.pth")
    pose_estimator = PointMAEPoseEstimator(
        num_points=128,
        num_groups=16,
        group_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_joints=17,
        in_channels=5,
        pose_head_type="cross_attention",
        pose_head_depth=2,
        pose_head_dropout=0.0,
        drop_path_rate=0.0
    ).to(device)

    if os.path.exists(av2_path):
        ckpt_av2 = torch.load(av2_path, map_location=device, weights_only=False)
        st_av2 = ckpt_av2["model_state_dict"] if "model_state_dict" in ckpt_av2 else ckpt_av2
        pose_estimator.load_state_dict(st_av2)
        print(f"[Model] Loaded frozen Model Av2 weights for Oracle Probing from: {av2_path}")
    else:
        print(f"[Warning] Model Av2 checkpoint not found: {av2_path}")

    # 4. Run Evaluation
    results = evaluate_dynamics_benchmark(dynamics_model, pose_estimator, test_ds, test_loader, device)

    # 5. Save Artifacts
    output_dir = args.output_dir or "docs/report_training"
    os.makedirs(output_dir, exist_ok=True)

    json_path = args.output_json or os.path.join(output_dir, "dynamics_benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[Done] Saved benchmark metrics JSON -> {json_path}")

    plot_path = os.path.join(output_dir, "dynamics_horizon_error_progression.png")
    plot_horizon_error_progression(results["per_horizon"], plot_path)


if __name__ == "__main__":
    main()
