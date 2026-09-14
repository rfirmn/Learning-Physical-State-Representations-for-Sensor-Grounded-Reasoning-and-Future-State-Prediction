import os
import sys
import copy
import yaml
import json
import time
import random
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.abspath("."))

from eksperimen_model.datasets import MMFiDataset
from eksperimen_model.models import PointMAEPoseEstimator, CompositePoseLoss
from eksperimen_model.utils import load_pretrained_point_mae
from eksperimen_model.train_pose_v2 import get_layer_wise_lr_groups, EMAModel


# Hyperparameter search space
SEARCH_SPACE = {
    "lr": [1e-4, 2e-4, 3e-4, 5e-4],
    "weight_decay": [0.01, 0.05, 0.1],
    "loss_bone_weight": [0.2, 0.4, 0.6, 0.8],
    "lr_layer_decay": [0.65, 0.75, 0.85],
    "pose_head_depth": [1, 2, 3],
    "pose_head_dropout": [0.1, 0.2, 0.3],
    "drop_path_rate": [0.05, 0.1, 0.15]
}


def sample_hyperparameters(search_space: Dict[str, list]) -> Dict[str, Any]:
    """Randomly samples a configuration from the search space."""
    return {k: random.choice(v) for k, v in search_space.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Automated Hyperparameter Tuning for Phase 2 3D Pose Estimation")
    parser.add_argument("--base_config", type=str, default="eksperimen_model/configs/mmfi_pose_v2.yaml", help="Base config")
    parser.add_argument("--n_trials", type=int, default=5, help="Number of tuning trials to evaluate")
    parser.add_argument("--epochs_per_trial", type=int, default=8, help="Number of epochs to train each candidate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size per trial")
    parser.add_argument("--env", type=str, nargs="+", default=None, help="Target environments (default: all from base config)")
    parser.add_argument("--train_sub", type=str, nargs="+", default=None, help="Subset of train subjects for fast tuning")
    parser.add_argument("--val_sub", type=str, nargs="+", default=None, help="Subset of val subjects for fast tuning")
    parser.add_argument("--prune_threshold_mm", type=float, default=450.0, help="Prune trial if MPJPE exceeds this after 3 epochs")
    parser.add_argument("--output_config", type=str, default="eksperimen_model/configs/mmfi_pose_best_tuned.yaml", help="Path to save best config")
    parser.add_argument("--auto_train_best", action="store_true", help="Launch full training automatically after tuning completes")
    parser.add_argument("--max_train_batches", type=int, default=None, help="Limiter for train batches per epoch (testing)")
    parser.add_argument("--max_val_batches", type=int, default=None, help="Limiter for val batches per epoch (testing)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def evaluate_trial(
    trial_id: int,
    params: Dict[str, Any],
    base_cfg: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader
) -> Dict[str, Any]:
    """Runs a single hyperparameter evaluation trial."""
    print("\n" + "=" * 70)
    print(f" >>> STARTING TRIAL {trial_id + 1}/{args.n_trials}")
    print(f"     Candidate Parameters: {params}")
    print("=" * 70)

    # Instantiate model with candidate architecture parameters
    model = PointMAEPoseEstimator(
        num_points=base_cfg["model"]["num_points"],
        num_groups=base_cfg["model"]["num_groups"],
        group_size=base_cfg["model"]["group_size"],
        embed_dim=base_cfg["model"]["embed_dim"],
        depth=base_cfg["model"]["depth"],
        num_heads=base_cfg["model"]["num_heads"],
        num_joints=base_cfg["model"]["num_joints"],
        in_channels=base_cfg["model"].get("in_channels", 5),
        freeze_encoder=False,
        pose_head_type=base_cfg["model"].get("pose_head_type", "cross_attention"),
        pose_head_depth=params["pose_head_depth"],
        pose_head_dropout=params["pose_head_dropout"],
        drop_path_rate=params["drop_path_rate"]
    ).to(device)

    # Load ShapeNet pretrained weights
    load_pretrained_point_mae(model, base_cfg["model"]["pretrained_weights"])

    # Composite Loss with candidate bone weight
    criterion = CompositePoseLoss(
        alpha=1.0,
        beta=params["loss_bone_weight"],
        use_weighted=True
    ).to(device)

    # Optimizer with candidate LLRD decay & weight decay
    param_groups = get_layer_wise_lr_groups(
        model,
        base_lr=params["lr"],
        weight_decay=params["weight_decay"],
        lr_decay=params["lr_layer_decay"],
        head_lr_mult=1.5
    )
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=base_cfg["training"].get("lr_restart_period", 20),
        T_mult=1,
        eta_min=1e-6
    )

    ema = EMAModel(model, decay=0.999)

    best_trial_mpjpe = float("inf")
    epoch_records = []
    pruned = False

    grad_accum_steps = base_cfg["training"].get("gradient_accumulation_steps", 2)

    for ep in range(1, args.epochs_per_trial + 1):
        model.train()
        train_loss_total = 0.0
        train_mpjpe_mm_total = 0.0
        train_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Trial {trial_id+1} Ep {ep}/{args.epochs_per_trial} [Train]",
            leave=False,
            dynamic_ncols=True
        )

        optimizer.zero_grad()
        for b_idx, batch in enumerate(pbar):
            if args.max_train_batches and b_idx >= args.max_train_batches:
                break
            pts = batch['points'].to(device, non_blocking=True)
            gt_pose = batch['skeleton'].to(device, non_blocking=True)

            pred_pose, _ = model(pts)
            loss, loss_dict = criterion(pred_pose, gt_pose)

            (loss / grad_accum_steps).backward()

            if (b_idx + 1) % grad_accum_steps == 0 or (b_idx + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)

            train_loss_total += loss.item()
            train_mpjpe_mm_total += loss_dict['mpjpe_mm']
            train_batches += 1

        scheduler.step()
        train_avg_mpjpe_mm = train_mpjpe_mm_total / max(1, train_batches)

        # Validation with EMA weights
        backup = ema.apply_shadow(model)
        model.eval()
        val_mpjpe_mm_total = 0.0
        val_batches = 0

        with torch.no_grad():
            for b_idx, batch in enumerate(val_loader):
                if args.max_val_batches and b_idx >= args.max_val_batches:
                    break
                pts = batch['points'].to(device, non_blocking=True)
                gt_pose = batch['skeleton'].to(device, non_blocking=True)
                pred_pose, _ = model(pts)
                _, loss_dict = criterion(pred_pose, gt_pose)
                val_mpjpe_mm_total += loss_dict['mpjpe_mm']
                val_batches += 1

        ema.restore(model, backup)

        val_avg_mpjpe_mm = val_mpjpe_mm_total / max(1, val_batches)
        if val_avg_mpjpe_mm < best_trial_mpjpe:
            best_trial_mpjpe = val_avg_mpjpe_mm

        epoch_records.append({
            "epoch": ep,
            "train_mpjpe_mm": train_avg_mpjpe_mm,
            "val_mpjpe_mm": val_avg_mpjpe_mm
        })

        print(f"     [Trial {trial_id+1} Ep {ep}/{args.epochs_per_trial}] Train: {train_avg_mpjpe_mm:.1f} mm | Val: {val_avg_mpjpe_mm:.1f} mm")

        # Early trial pruning if divergence detected
        if ep >= 3 and val_avg_mpjpe_mm > args.prune_threshold_mm:
            print(f"     >>> [PRUNED] Trial {trial_id+1} pruned early (Val MPJPE {val_avg_mpjpe_mm:.1f}mm > {args.prune_threshold_mm}mm).")
            pruned = True
            break

    del model, optimizer, scheduler, criterion
    torch.cuda.empty_cache()

    return {
        "trial_id": trial_id + 1,
        "parameters": params,
        "best_val_mpjpe_mm": best_trial_mpjpe,
        "pruned": pruned,
        "history": epoch_records
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.base_config, "r") as f:
        base_cfg = yaml.safe_load(f)

    environments = args.env or base_cfg["dataset"]["environments"]
    train_subjects = args.train_sub or base_cfg["dataset"]["split"]["train_subjects"]
    val_subjects = args.val_sub or base_cfg["dataset"]["split"]["val_subjects"]

    print("=" * 70)
    print(" POINT-MAE 3D POSE ESTIMATION: HYPERPARAMETER TUNING")
    print("=" * 70)
    print(f" DEVICE              : {device}")
    print(f" BASE CONFIG         : {args.base_config}")
    print(f" TOTAL CANDIDATES    : {args.n_trials} trials")
    print(f" EPOCHS PER TRIAL    : {args.epochs_per_trial}")
    print(f" BATCH SIZE          : {args.batch_size}")
    print(f" TRAIN / VAL SUBJECTS: {len(train_subjects)} train / {len(val_subjects)} val")
    print("=" * 70)

    # Prepare datasets once
    dataset_root = base_cfg["dataset"]["root_dir"]
    num_points = base_cfg["model"]["num_points"]
    use_extra_features = base_cfg["dataset"].get("use_extra_features", True)
    aug_cfg = base_cfg.get("augmentation", {})

    print("[Tuning] Loading dataset splits...")
    train_dataset = MMFiDataset(
        root_dir=dataset_root,
        subjects=train_subjects,
        environments=environments,
        num_points=num_points,
        augment=True,
        normalize=True,
        use_extra_features=use_extra_features,
        augment_config=aug_cfg
    )
    val_dataset = MMFiDataset(
        root_dir=dataset_root,
        subjects=val_subjects,
        environments=environments,
        num_points=num_points,
        augment=False,
        normalize=True,
        use_extra_features=use_extra_features
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=base_cfg["training"].get("num_workers", 4),
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=base_cfg["training"].get("num_workers", 4),
        pin_memory=(device.type == "cuda")
    )

    # Generate distinct trial configurations
    trial_configs = []
    # Candidate 0 is always the baseline config
    trial_configs.append({
        "lr": float(base_cfg["training"]["lr"]),
        "weight_decay": float(base_cfg["training"].get("weight_decay", 0.05)),
        "loss_bone_weight": float(base_cfg["training"].get("loss_bone_weight", 0.5)),
        "lr_layer_decay": float(base_cfg["training"].get("lr_layer_decay", 0.75)),
        "pose_head_depth": int(base_cfg["model"].get("pose_head_depth", 2)),
        "pose_head_dropout": float(base_cfg["model"].get("pose_head_dropout", 0.2)),
        "drop_path_rate": float(base_cfg["model"].get("drop_path_rate", 0.1))
    })

    while len(trial_configs) < args.n_trials:
        cand = sample_hyperparameters(SEARCH_SPACE)
        if cand not in trial_configs:
            trial_configs.append(cand)

    all_results = []
    start_time = time.time()

    for idx, params in enumerate(trial_configs):
        res = evaluate_trial(
            trial_id=idx,
            params=params,
            base_cfg=base_cfg,
            args=args,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader
        )
        all_results.append(res)

    total_tuning_time = time.time() - start_time

    # Sort results by best validation MPJPE
    all_results.sort(key=lambda x: x["best_val_mpjpe_mm"])
    best_result = all_results[0]

    print("\n" + "=" * 70)
    print(" HYPERPARAMETER TUNING COMPLETED!")
    print(f" Total Elapsed Time: {total_tuning_time / 60:.1f} minutes")
    print("=" * 70)
    print(" SUMMARY LEADERBOARD:")
    print(f" {'Rank':<5} | {'Trial':<6} | {'Val MPJPE':<12} | {'LR':<8} | {'Bone W':<7} | {'LLRD':<6} | {'Head Depth':<10} | {'Pruned'}")
    print("-" * 75)
    for rank, r in enumerate(all_results, 1):
        p = r["parameters"]
        pruned_str = "YES" if r["pruned"] else "NO"
        print(f" {rank:<5} | {r['trial_id']:<6} | {r['best_val_mpjpe_mm']:.1f} mm      | {p['lr']:<8.1e} | {p['loss_bone_weight']:<7.2f} | {p['lr_layer_decay']:<6.2f} | {p['pose_head_depth']:<10} | {pruned_str}")
    print("=" * 70)
    print(f"\n >>> WINNING CONFIGURATION (Trial {best_result['trial_id']}):")
    print(f"     Val MPJPE: {best_result['best_val_mpjpe_mm']:.1f} mm")
    print(f"     Parameters: {json.dumps(best_result['parameters'], indent=4)}")

    # Update base config with winning parameters and write to output_config
    best_cfg = copy.deepcopy(base_cfg)
    best_params = best_result["parameters"]
    best_cfg["training"]["lr"] = best_params["lr"]
    best_cfg["training"]["weight_decay"] = best_params["weight_decay"]
    best_cfg["training"]["loss_bone_weight"] = best_params["loss_bone_weight"]
    best_cfg["training"]["lr_layer_decay"] = best_params["lr_layer_decay"]
    best_cfg["model"]["pose_head_depth"] = best_params["pose_head_depth"]
    best_cfg["model"]["pose_head_dropout"] = best_params["pose_head_dropout"]
    best_cfg["model"]["drop_path_rate"] = best_params["drop_path_rate"]
    best_cfg["experiment_name"] = f"point_mae_mmfi_pose_best_tuned_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    os.makedirs(os.path.dirname(args.output_config), exist_ok=True)
    with open(args.output_config, "w") as f:
        yaml.safe_dump(best_cfg, f, sort_keys=False)
    print(f"\n[Saved] Winning configuration saved to: {args.output_config}")

    # Save Markdown and JSON Tuning Reports
    report_dir = "docs/report_training"
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(report_dir, f"tuning_results_{timestamp}.json")
    md_path = os.path.join(report_dir, f"tuning_results_{timestamp}.md")

    with open(json_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "total_time_seconds": total_tuning_time,
            "n_trials": args.n_trials,
            "epochs_per_trial": args.epochs_per_trial,
            "best_trial": best_result,
            "leaderboard": all_results
        }, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Hasil Hyperparameter Tuning Tahap 2\n\n")
        f.write(f"- **Waktu:** {timestamp}\n")
        f.write(f"- **Total Durasi:** {total_tuning_time / 60:.1f} menit\n")
        f.write(f"- **Jumlah Trial:** {args.n_trials} (setiap trial {args.epochs_per_trial} epoch)\n")
        f.write(f"- **Best Val MPJPE:** **{best_result['best_val_mpjpe_mm']:.1f} mm** (Trial {best_result['trial_id']})\n\n")
        f.write(f"## Tabel Peringkat (*Leaderboard*)\n\n")
        f.write("| Rank | Trial | Val MPJPE (mm) | LR | Bone Weight | LLRD Decay | Head Depth | Head Dropout | DropPath | Status |\n")
        f.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for rank, r in enumerate(all_results, 1):
            p = r["parameters"]
            st = "Pruned" if r["pruned"] else "Completed"
            f.write(f"| {rank} | #{r['trial_id']} | **{r['best_val_mpjpe_mm']:.1f}** | {p['lr']:.1e} | {p['loss_bone_weight']} | {p['lr_layer_decay']} | {p['pose_head_depth']} | {p['pose_head_dropout']} | {p['drop_path_rate']} | {st} |\n")
        f.write(f"\n## Konfigurasi Terbaik Yang Diekspor\n\n")
        f.write(f"Konfigurasi telah disimpan ke `{args.output_config}`.\n\n")

    print(f"[Saved] Tuning reports generated at:\n       {json_path}\n       {md_path}")

    # Optional auto-launch full training
    if args.auto_train_best:
        print("\n" + "=" * 70)
        print(" [Auto-Train] Launching full training with best tuned configuration...")
        print("=" * 70)
        cmd = f'& "{sys.executable}" eksperimen_model/train_pose_v2.py --config "{args.output_config}"'
        os.system(cmd)


if __name__ == "__main__":
    main()
