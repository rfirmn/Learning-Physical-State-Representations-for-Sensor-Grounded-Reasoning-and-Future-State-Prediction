import os
import sys
import argparse
import yaml
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional

sys.path.insert(0, os.path.abspath("."))

from eksperimen_model.datasets import MMFiDataset
from eksperimen_model.models import PointMAEPoseEstimator

JOINT_NAMES = [
    "Pelvis", "R_Hip", "R_Knee", "R_Ankle", "L_Hip", "L_Knee", "L_Ankle",
    "Spine", "Thorax", "Neck/Nose", "Head", "L_Shoulder", "L_Elbow", "L_Wrist",
    "R_Shoulder", "R_Elbow", "R_Wrist"
]

ANATOMICAL_GROUPS = {
    "Central Body Axis": [0, 7, 8, 9, 10],      # Pelvis, Spine, Thorax, Neck, Head
    "Lower Limbs (Legs)": [1, 2, 3, 4, 5, 6],    # Hips, Knees, Ankles
    "Upper Limbs (Arms)": [11, 12, 13, 14, 15, 16] # Shoulders, Elbows, Wrists
}

ACTION_NAMES = {
    0: "A01 (Chest expansion)", 1: "A02 (Side twist)", 2: "A03 (Arm extensions)",
    3: "A04 (Left extension)", 4: "A05 (Right extension)", 5: "A06 (High arm punch)",
    6: "A07 (Low arm punch)", 7: "A08 (Horizontal kick)", 8: "A09 (Squat)",
    9: "A10 (Front lunge)", 10: "A11 (Left lunge)", 11: "A12 (Right lunge)",
    12: "A13 (Diagonal punch)", 13: "A14 (Two-handed punch)",
    14: "A15 (Rehab: Shoulder abduct)", 15: "A16 (Rehab: Shoulder flex)",
    16: "A17 (Rehab: Elbow flex)", 17: "A18 (Rehab: Wrist flex)",
    18: "A19 (Rehab: Hip flex)", 19: "A20 (Rehab: Knee flex)",
    20: "A21 (Rehab: Ankle flex)", 21: "A22 (Rehab: Side leg raise)",
    22: "A23 (Rehab: Back leg raise)", 23: "A24 (Rehab: Torso twist)",
    24: "A25 (Rehab: Arm circle)", 25: "A26 (Rehab: Leg circle)",
    26: "A27 (Rehab: Balance stance)"
}

def compute_procrustes_alignment(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Computes optimal rotation + translation alignment via orthogonal Procrustes (SVD).
    pred, gt: (17, 3)
    Returns: aligned_pred (17, 3)
    """
    mu_pred = np.mean(pred, axis=0)
    mu_gt = np.mean(gt, axis=0)
    pred_c = pred - mu_pred
    gt_c = gt - mu_gt

    # SVD of covariance matrix
    h = pred_c.T @ gt_c
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T

    # Reflection correction
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T

    return pred_c @ r + mu_gt

def compute_bootstrap_ci(data: np.ndarray, n_boot: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Computes non-parametric bootstrap confidence interval for the mean."""
    if len(data) == 0:
        return 0.0, 0.0
    indices = np.random.randint(0, len(data), size=(n_boot, len(data)))
    boot_means = np.mean(data[indices], axis=1)
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, alpha * 100.0))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100.0))
    return low, high

def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()

    all_raw_errors = []        # (N, 17) meters
    all_pa_errors = []         # (N, 17) meters
    all_torso_lengths = []     # (N,) meters
    env_labels = []            # (N,) str
    sub_labels = []            # (N,) str
    action_labels = []         # (N,) int

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Benchmarks"):
            pts = batch['points'].to(device)
            gt = batch['skeleton'].to(device)
            envs = batch.get('env', ['E01'] * len(pts))
            subs = batch.get('sub', ['S01'] * len(pts))
            acts = batch.get('action', torch.zeros(len(pts), dtype=torch.long)).cpu().numpy()

            pred, _ = model(pts)

            pred_np = pred.cpu().numpy() # (B, 17, 3)
            gt_np = gt.cpu().numpy()     # (B, 17, 3)

            # Raw MPJPE distance
            diff = pred_np - gt_np
            raw_dist = np.linalg.norm(diff, axis=-1) # (B, 17)
            all_raw_errors.append(raw_dist)

            # Procrustes Aligned MPJPE (PA-MPJPE)
            b_size = len(pred_np)
            pa_dist_batch = np.zeros((b_size, 17), dtype=np.float32)
            torso_batch = np.zeros(b_size, dtype=np.float32)

            for i in range(b_size):
                p_aligned = compute_procrustes_alignment(pred_np[i], gt_np[i])
                pa_dist_batch[i] = np.linalg.norm(p_aligned - gt_np[i], axis=-1)
                # Torso length: distance between Pelvis (0) and Neck (9) or Thorax (8)
                torso_len = np.linalg.norm(gt_np[i, 9] - gt_np[i, 0])
                torso_batch[i] = max(0.2, torso_len) # Clamp minimum 20cm

            all_pa_errors.append(pa_dist_batch)
            all_torso_lengths.append(torso_batch)
            env_labels.extend(envs)
            sub_labels.extend(subs)
            action_labels.extend(acts)

    all_raw_errors = np.concatenate(all_raw_errors, axis=0) # (N, 17)
    all_pa_errors = np.concatenate(all_pa_errors, axis=0)   # (N, 17)
    all_torso_lengths = np.concatenate(all_torso_lengths, axis=0) # (N,)
    sample_mpjpe = np.mean(all_raw_errors, axis=1)          # (N,) in meters
    sample_pa_mpjpe = np.mean(all_pa_errors, axis=1)        # (N,) in meters

    # 1. Primary Metrics
    overall_mpjpe_m = float(np.mean(sample_mpjpe))
    overall_pa_mpjpe_m = float(np.mean(sample_pa_mpjpe))
    overall_mpjpe_mm = overall_mpjpe_m * 1000.0
    overall_pa_mpjpe_mm = overall_pa_mpjpe_m * 1000.0

    # 95% Confidence Interval
    ci_low, ci_high = compute_bootstrap_ci(sample_mpjpe * 1000.0, n_boot=500)

    # 2. PCK @ thresholds (in meters: 30mm=0.03m, 50mm=0.05m, 100mm=0.10m, 150mm=0.15m)
    pck_30 = float(np.mean(all_raw_errors < 0.030) * 100.0)
    pck_50 = float(np.mean(all_raw_errors < 0.050) * 100.0)
    pck_100 = float(np.mean(all_raw_errors < 0.100) * 100.0)
    pck_150 = float(np.mean(all_raw_errors < 0.150) * 100.0)

    # 3. Normalized MPJPE (N-MPJPE)
    normalized_errors = sample_mpjpe / all_torso_lengths
    n_mpjpe = float(np.mean(normalized_errors))

    # 4. Per-Joint Error
    mean_per_joint = np.mean(all_raw_errors, axis=0) # (17,)

    # 5. Anatomical Groups
    group_metrics = {}
    for g_name, j_indices in ANATOMICAL_GROUPS.items():
        group_errors = all_raw_errors[:, j_indices]
        group_metrics[g_name] = {
            'mpjpe_mm': float(np.mean(group_errors) * 1000.0),
            'mpjpe_m': float(np.mean(group_errors))
        }

    # 6. Per-Environment Metrics
    env_metrics = {}
    env_labels_arr = np.array(env_labels)
    unique_envs = sorted(list(set(env_labels)))

    for env_id in unique_envs:
        mask = (env_labels_arr == env_id)
        if np.any(mask):
            env_errs = sample_mpjpe[mask] * 1000.0
            env_pa_errs = sample_pa_mpjpe[mask] * 1000.0
            env_metrics[env_id] = {
                'samples': int(np.sum(mask)),
                'mpjpe_mm': float(np.mean(env_errs)),
                'pa_mpjpe_mm': float(np.mean(env_pa_errs)),
                'pck_50': float(np.mean(all_raw_errors[mask] < 0.050) * 100.0),
                'pck_100': float(np.mean(all_raw_errors[mask] < 0.100) * 100.0)
            }

    # Environment Robustness Score (ERS)
    env_mpjpes = [v['mpjpe_mm'] for v in env_metrics.values()]
    if len(env_mpjpes) > 1:
        ers = float(max(0.0, 1.0 - (np.std(env_mpjpes) / np.mean(env_mpjpes))))
    else:
        ers = 1.0

    # 7. Action Category Metrics (Daily A01-A14 vs Rehab A15-A27)
    action_labels_arr = np.array(action_labels)
    daily_mask = action_labels_arr < 14
    rehab_mask = action_labels_arr >= 14

    category_metrics = {
        'Daily Activities (A01-A14)': {
            'samples': int(np.sum(daily_mask)),
            'mpjpe_mm': float(np.mean(sample_mpjpe[daily_mask]) * 1000.0) if np.any(daily_mask) else 0.0,
            'pa_mpjpe_mm': float(np.mean(sample_pa_mpjpe[daily_mask]) * 100.0) if np.any(daily_mask) else 0.0
        },
        'Rehabilitation Movements (A15-A27)': {
            'samples': int(np.sum(rehab_mask)),
            'mpjpe_mm': float(np.mean(sample_mpjpe[rehab_mask]) * 1000.0) if np.any(rehab_mask) else 0.0,
            'pa_mpjpe_mm': float(np.mean(sample_pa_mpjpe[rehab_mask]) * 100.0) if np.any(rehab_mask) else 0.0
        }
    }

    # Top-3 Best and Worst Actions
    unique_acts = np.unique(action_labels_arr)
    act_avg_errs = []
    for act_id in unique_acts:
        mask = (action_labels_arr == act_id)
        act_avg_errs.append((act_id, float(np.mean(sample_mpjpe[mask]) * 1000.0), int(np.sum(mask))))

    act_avg_errs.sort(key=lambda x: x[1])
    best_actions = act_avg_errs[:3]
    worst_actions = act_avg_errs[-3:][::-1]

    # Console Output (Publication-Grade Table)
    print("\n" + "=" * 78)
    print("        SCIENTIFIC EVALUATION BENCHMARK: 3D SKELETON ESTIMATION (MM-Fi)")
    print("=" * 78)
    print(f" Total Evaluated Frames: {len(sample_mpjpe):,} frames")
    print(f" Overall MPJPE          : {overall_mpjpe_mm:.2f} mm ({overall_mpjpe_m:.4f} m) [95% CI: {ci_low:.1f} - {ci_high:.1f} mm]")
    print(f" Procrustes PA-MPJPE    : {overall_pa_mpjpe_mm:.2f} mm ({overall_pa_mpjpe_m:.4f} m)")
    print(f" Normalized MPJPE (N)   : {n_mpjpe:.4f} (relative to trunk length)")
    print(f" Keypoint PCK Accuracies: PCK@30mm={pck_30:.1f}% | PCK@50mm={pck_50:.1f}% | PCK@100mm={pck_100:.1f}% | PCK@150mm={pck_150:.1f}%")
    print(f" Env Robustness Score   : {ers:.3f} (1.0 = invariant across environments)")
    print("-" * 78)
    print(" [1] Cross-Environment Generalization Breakdown:")
    print(f"     {'Env':<6} | {'Samples':<8} | {'MPJPE (mm)':<12} | {'PA-MPJPE (mm)':<14} | {'PCK@50mm':<10} | {'PCK@100mm':<10}")
    print("     " + "-" * 70)
    for env_id, stats in env_metrics.items():
        print(f"     {env_id:<6} | {stats['samples']:<8} | {stats['mpjpe_mm']:<12.1f} | {stats['pa_mpjpe_mm']:<14.1f} | {stats['pck_50']:<10.1f}% | {stats['pck_100']:<10.1f}%")

    print("-" * 78)
    print(" [2] Anatomical Group Breakdown:")
    for g_name, stats in group_metrics.items():
        print(f"     * {g_name:<20}: {stats['mpjpe_mm']:.1f} mm ({stats['mpjpe_m']:.4f} m)")

    print("-" * 78)
    print(" [3] Action Domain Breakdown:")
    for cat_name, stats in category_metrics.items():
        print(f"     * {cat_name:<34}: {stats['mpjpe_mm']:.1f} mm (Frames: {stats['samples']:,})")
    print(f"     Best 3 Actions  : " + ", ".join([f"{ACTION_NAMES.get(a[0], f'A{a[0]+1:02d}')}: {a[1]:.1f}mm" for a in best_actions]))
    print(f"     Worst 3 Actions : " + ", ".join([f"{ACTION_NAMES.get(a[0], f'A{a[0]+1:02d}')}: {a[1]:.1f}mm" for a in worst_actions]))

    print("-" * 78)
    print(" [4] Per-Joint Error Table (17 Keypoints):")
    print(f"     {'Idx':<4} | {'Joint Name':<16} | {'Group':<18} | {'Error (meters)':<14} | {'Error (mm)':<10}")
    print("     " + "-" * 70)
    for j_idx, (j_name, err) in enumerate(zip(JOINT_NAMES, mean_per_joint)):
        grp = "Central" if j_idx in ANATOMICAL_GROUPS["Central Body Axis"] else ("Lower Limb" if j_idx in ANATOMICAL_GROUPS["Lower Limbs (Legs)"] else "Upper Limb")
        print(f"     {j_idx:<4} | {j_name:<16} | {grp:<18} | {err:<14.4f} | {err*1000:<10.1f}")
    print("=" * 78 + "\n")

    return {
        'total_frames': len(sample_mpjpe),
        'mpjpe_mm': overall_mpjpe_mm,
        'mpjpe_m': overall_mpjpe_m,
        'pa_mpjpe_mm': overall_pa_mpjpe_mm,
        'pa_mpjpe_m': overall_pa_mpjpe_m,
        'ci_95_mm': [ci_low, ci_high],
        'pck': {'pck_30': pck_30, 'pck_50': pck_50, 'pck_100': pck_100, 'pck_150': pck_150},
        'n_mpjpe': n_mpjpe,
        'ers': ers,
        'env_metrics': env_metrics,
        'group_metrics': group_metrics,
        'category_metrics': category_metrics,
        'per_joint_mm': (mean_per_joint * 1000.0).tolist(),
        'per_joint_m': mean_per_joint.tolist()
    }

def main():
    parser = argparse.ArgumentParser(description="Publication-Grade Evaluation for MM-Fi 3D Skeleton Estimation")
    parser.add_argument("--checkpoint", type=str, default="eksperimen_model/checkpoints/pose_estimation/best_model.pth")
    parser.add_argument("--config", type=str, default="eksperimen_model/configs/mmfi_pose_finetune.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train", "all", "custom"],
                        help="Data split to evaluate: 'test' (unseen test subjects), 'val' (unseen val subjects), 'all', or 'custom'")
    parser.add_argument("--subjects", type=str, nargs="+", default=None, help="Explicit list of subjects (e.g. S04 S07 S13)")
    parser.add_argument("--env", type=str, default=None, help="Filter specific environment (e.g. E01)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to save full evaluation JSON")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine subjects based on split mode
    split_cfg = cfg["dataset"].get("split", {})
    if args.subjects is not None:
        target_subjects = args.subjects
    elif args.split == "test":
        target_subjects = split_cfg.get("test_subjects", ["S04", "S07", "S13", "S17", "S22", "S25", "S36", "S40"])
    elif args.split == "val":
        target_subjects = split_cfg.get("val_subjects", ["S05", "S10", "S18", "S20", "S24", "S28", "S34", "S35"])
    elif args.split == "train":
        target_subjects = split_cfg.get("train_subjects", [])
    elif args.split == "all":
        target_subjects = None # None means all 40 subjects in MMFiDataset
    else:
        target_subjects = split_cfg.get("val_subjects", [])

    target_envs = [args.env] if args.env else cfg["dataset"].get("environments", ["E01", "E02", "E03", "E04"])

    print("=" * 78)
    print(f" EVALUATION SETUP: Split='{args.split}' | Subjects={target_subjects if target_subjects else 'ALL (S01-S40)'}")
    print(f" ENVIRONMENTS: {target_envs} | CHECKPOINT: {args.checkpoint}")
    print("=" * 78)

    num_points = cfg["model"]["num_points"]
    dataset = MMFiDataset(
        root_dir=cfg["dataset"]["root_dir"],
        subjects=target_subjects,
        environments=target_envs,
        num_points=num_points,
        augment=False,
        normalize=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    model = PointMAEPoseEstimator(
        num_points=num_points,
        num_groups=cfg["model"]["num_groups"],
        group_size=cfg["model"]["group_size"],
        embed_dim=cfg["model"]["embed_dim"],
        depth=cfg["model"]["depth"],
        num_heads=cfg["model"]["num_heads"],
        num_joints=cfg["model"]["num_joints"]
    ).to(device)

    if os.path.exists(args.checkpoint):
        print(f" Loading checkpoint weights from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    else:
        print(f" [Warning] Checkpoint '{args.checkpoint}' not found! Running evaluation on randomly initialized model.")

    results = evaluate(model, dataloader, device)

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[Done] Detailed evaluation results saved to: {args.output_json}")

if __name__ == "__main__":
    main()
