import os
import sys
import argparse
import yaml
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.abspath("."))

from eksperimen_model.datasets import MMFiDataset
from eksperimen_model.models import PointMAEPoseEstimator

def run_point_dropout_experiment(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    point_counts: List[int] = [128, 96, 64, 32, 16],
    max_batches: Optional[int] = None
) -> Dict[int, float]:
    """Evaluates MPJPE when random points are dropped (simulating occlusion / weak radar return)."""
    results = {}
    print("\n[Robustness Test 1/2] Point Cloud Dropout Evaluation...")

    total_steps = min(len(dataloader), max_batches) if max_batches else len(dataloader)
    for k in point_counts:
        errs = []
        for b_idx, batch in enumerate(tqdm(dataloader, total=total_steps, desc=f"Dropout -> {k} pts")):
            if max_batches and b_idx >= max_batches:
                break
            pts = batch['points'].to(device) # (B, 128, 3)
            gt = batch['skeleton'].to(device) # (B, 17, 3)

            # Randomly sub-select k points and pad back to 128 (via duplication)
            b_size, n_pts, d = pts.shape
            if k < n_pts:
                idx = torch.randperm(n_pts)[:k]
                sub_pts = pts[:, idx, :] # (B, k, 3)
                # Duplicate points back to 128
                repeat_times = (n_pts // k) + 1
                expanded_pts = sub_pts.repeat(1, repeat_times, 1)[:, :n_pts, :]
            else:
                expanded_pts = pts

            with torch.no_grad():
                pred, _ = model(expanded_pts)
                diff = pred - gt
                dist = torch.norm(diff, p=2, dim=-1).mean(dim=-1).cpu().numpy()
                errs.extend(dist)

        avg_mm = float(np.mean(errs) * 1000.0)
        results[k] = avg_mm
        print(f" >>> Points: {k:<3} | MPJPE: {avg_mm:.2f} mm ({avg_mm/1000:.4f} m)")

    return results

def run_gaussian_noise_experiment(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    noise_stds: List[float] = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10],
    max_batches: Optional[int] = None
) -> Dict[float, float]:
    """Evaluates MPJPE under synthetic spatial Gaussian jitter (simulating radar phase noise / multipath)."""
    results = {}
    print("\n[Robustness Test 2/2] Gaussian Spatial Noise Injection Evaluation...")

    total_steps = min(len(dataloader), max_batches) if max_batches else len(dataloader)
    for sigma in noise_stds:
        errs = []
        for b_idx, batch in enumerate(tqdm(dataloader, total=total_steps, desc=f"Noise sigma = {sigma*1000:.0f} mm")):
            if max_batches and b_idx >= max_batches:
                break
            pts = batch['points'].to(device) # (B, 128, 3)
            gt = batch['skeleton'].to(device) # (B, 17, 3)

            if sigma > 0:
                noise = torch.randn_like(pts) * sigma
                noisy_pts = pts + noise
            else:
                noisy_pts = pts

            with torch.no_grad():
                pred, _ = model(noisy_pts)
                diff = pred - gt
                dist = torch.norm(diff, p=2, dim=-1).mean(dim=-1).cpu().numpy()
                errs.extend(dist)

        avg_mm = float(np.mean(errs) * 1000.0)
        results[sigma] = avg_mm
        print(f" >>> Sigma: {sigma*1000:>3.0f} mm | MPJPE: {avg_mm:.2f} mm ({avg_mm/1000:.4f} m)")

    return results

def plot_robustness_curves(dropout_results: Dict[int, float], noise_results: Dict[float, float], output_dir: str):
    """Plots publication-grade degradation curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Curve 1: Point Dropout
    pts = sorted(list(dropout_results.keys()))
    d_errs = [dropout_results[k] for k in pts]
    ax1.plot(pts, d_errs, 'o-', color='#e74c3c', linewidth=2.5, markersize=7)
    ax1.set_title('Robustness to Point Cloud Sparsity / Occlusion', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Number of Available Radar Points', fontsize=11)
    ax1.set_ylabel('MPJPE (mm)', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    for x, y in zip(pts, d_errs):
        ax1.annotate(f'{y:.1f}mm', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9, fontweight='bold')

    # Curve 2: Gaussian Noise
    sigmas = sorted(list(noise_results.keys()))
    sigmas_mm = [s * 1000.0 for s in sigmas]
    n_errs = [noise_results[s] for s in sigmas]
    ax2.plot(sigmas_mm, n_errs, 's-', color='#2980b9', linewidth=2.5, markersize=7)
    ax2.set_title('Robustness to Radar Spatial Phase Noise (Multipath)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Injected Noise Standard Deviation σ (mm)', fontsize=11)
    ax2.set_ylabel('MPJPE (mm)', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    for x, y in zip(sigmas_mm, n_errs):
        ax2.annotate(f'{y:.1f}mm', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plot_file = os.path.join(output_dir, "robustness_curves.png")
    fig.savefig(plot_file)
    plt.close(fig)
    print(f"\n[Robustness] Curves saved to: {plot_file}")
    return plot_file

def main():
    parser = argparse.ArgumentParser(description="Evaluate Point-MAE Robustness to Point Dropout and Noise")
    parser.add_argument("--checkpoint", type=str, default="eksperimen_model/checkpoints/pose_estimation/best_model.pth")
    parser.add_argument("--config", type=str, default="eksperimen_model/configs/mmfi_pose_finetune.yaml")
    parser.add_argument("--output_dir", type=str, default="docs/report_training/robustness_analysis")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=100, help="Evaluate on subset of batches for speed")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    val_subjects = cfg["dataset"]["split"].get("val_subjects", ["S05", "S10", "S18", "S20", "S24", "S28", "S34", "S35"])
    dataset = MMFiDataset(
        root_dir=cfg["dataset"]["root_dir"],
        subjects=val_subjects,
        environments=cfg["dataset"].get("environments", ["E01", "E02", "E03", "E04"]),
        num_points=cfg["model"]["num_points"],
        augment=False,
        normalize=True
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = PointMAEPoseEstimator(
        num_points=cfg["model"]["num_points"],
        num_groups=cfg["model"]["num_groups"],
        group_size=cfg["model"]["group_size"],
        embed_dim=cfg["model"]["embed_dim"],
        depth=cfg["model"]["depth"],
        num_heads=cfg["model"]["num_heads"],
        num_joints=cfg["model"]["num_joints"]
    ).to(device)

    if os.path.exists(args.checkpoint):
        print(f"Loading weights from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    else:
        print(f"[Warning] Checkpoint '{args.checkpoint}' not found! Evaluating baseline model.")

    dropout_res = run_point_dropout_experiment(model, dataloader, device, max_batches=args.max_batches)
    noise_res = run_gaussian_noise_experiment(model, dataloader, device, max_batches=args.max_batches)
    plot_file = plot_robustness_curves(dropout_res, noise_res, args.output_dir)

    summary = {
        'point_dropout_mpjpe_mm': dropout_res,
        'gaussian_noise_mpjpe_mm': {str(k): v for k, v in noise_res.items()},
        'plot_path': plot_file
    }

    json_path = os.path.join(args.output_dir, "robustness_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Done] Robustness report saved to: {json_path}")

if __name__ == "__main__":
    main()
