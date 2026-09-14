import os
import sys
import yaml
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, os.path.abspath("."))

from eksperimen_model.datasets import MMFiDataset
from eksperimen_model.models import (
    PointMAEPoseEstimator,
    CompositePoseLoss,
    MPJPELoss
)
from eksperimen_model.utils import (
    load_pretrained_point_mae,
    save_checkpoint,
    ExperimentLogger,
    StallWatchdog,
    ExperimentReporter
)


class EMAModel:
    """
    Exponential Moving Average (EMA) for smoother model evaluation and weights consolidation.
    Maintains shadow weights: theta_ema = decay * theta_ema + (1 - decay) * theta
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Loads shadow weights into model, returning backup of current weights."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for k, v in self.shadow.items():
            if k in backup:
                backup_tensor = model.state_dict()[k]
                backup_tensor.copy_(v)
        return backup

    def restore(self, model: nn.Module, backup: Dict[str, torch.Tensor]):
        """Restores model weights from backup."""
        model.load_state_dict(backup)


def get_layer_wise_lr_groups(
    model: PointMAEPoseEstimator,
    base_lr: float,
    weight_decay: float = 0.05,
    lr_decay: float = 0.75,
    head_lr_mult: float = 1.5
) -> List[Dict[str, Any]]:
    """
    Layer-wise Learning Rate Decay (LLRD).
    Lower layers (spatial geometric primitives) learn with smaller LR to preserve ShapeNet weights.
    Upper layers and Pose Head learn with higher LR for task-specific adaptation.
    """
    num_blocks = 12
    param_groups = []

    def get_decay_scale(param_name: str) -> float:
        if "pose_head" in param_name:
            return head_lr_mult
        elif "MAE_encoder.blocks.blocks." in param_name:
            # Extract block index
            parts = param_name.split("MAE_encoder.blocks.blocks.")[1].split(".")
            block_idx = int(parts[0])
            # block 11 has decay^0, block 0 has decay^11
            return lr_decay ** (num_blocks - 1 - block_idx)
        elif "MAE_encoder.norm" in param_name:
            return 1.0
        else:
            # First conv, patch embed, pos embed (deepest layer)
            return lr_decay ** num_blocks

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        scale = get_decay_scale(name)
        # Biases and 1D norms do not receive weight decay
        is_decay = param.ndim > 1 and "norm" not in name and "bias" not in name
        param_groups.append({
            'params': [param],
            'lr': base_lr * scale,
            'weight_decay': weight_decay if is_decay else 0.0,
            'name': name
        })

    return param_groups


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2 (v2): Point-MAE 3D Human Pose Estimation (Optimized)")
    parser.add_argument("--config", type=str, default="eksperimen_model/configs/mmfi_pose_v2.yaml", help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--env", type=str, nargs="+", default=None, help="Target environments (e.g. E01 E02)")
    parser.add_argument("--train_sub", type=str, nargs="+", default=None, help="Target train subjects")
    parser.add_argument("--val_sub", type=str, nargs="+", default=None, help="Target val subjects")
    parser.add_argument("--stall_timeout", type=float, default=120.0, help="Watchdog timeout in seconds")
    parser.add_argument("--max_train_batches", type=int, default=None, help="Debug limiter for train batches per epoch")
    parser.add_argument("--max_val_batches", type=int, default=None, help="Debug limiter for val batches per epoch")
    parser.add_argument("--no_ema", action="store_true", help="Disable Exponential Moving Average")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load Configuration
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Overrides from CLI
    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    lr = args.lr or cfg["training"]["lr"]
    environments = args.env or cfg["dataset"]["environments"]
    train_subjects = args.train_sub or cfg["dataset"]["split"]["train_subjects"]
    val_subjects = args.val_sub or cfg["dataset"]["split"]["val_subjects"]
    test_subjects = cfg["dataset"]["split"].get("test_subjects", [])
    num_workers = cfg["training"].get("num_workers", 4)
    output_dir = cfg.get("output_dir", "eksperimen_model/checkpoints/pose_estimation_v2")
    os.makedirs(output_dir, exist_ok=True)

    # Gradient accumulation
    grad_accum_steps = cfg["training"].get("gradient_accumulation_steps", 2)
    effective_bs = batch_size * grad_accum_steps

    # Set Reproducibility Seed
    seed = cfg["training"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(" POINT-MAE 3D SKELETON POSE ESTIMATION (v2 OPTIMIZED)")
    print("=" * 70)
    print(f" TARGET DEVICE   : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" CONFIG PATH     : {args.config}")
    print(f" INPUT CHANNELS  : {cfg['model'].get('in_channels', 5)} (Doppler + SNR enabled: {cfg['dataset'].get('use_extra_features', True)})")
    print(f" POSE HEAD TYPE  : {cfg['model'].get('pose_head_type', 'cross_attention')} (Depth: {cfg['model'].get('pose_head_depth', 2)})")
    print(f" LOSS FUNCTION   : CompositePoseLoss (MPJPE + BoneLength prior beta={cfg['training'].get('loss_bone_weight', 0.5)})")
    print(f" BATCH / EFFECTIVE: {batch_size} / {effective_bs} (accum steps={grad_accum_steps})")
    print(f" EPOCHS / LR     : {epochs} | base LR={lr:.2e} with Layer-wise LR Decay (rate={cfg['training'].get('lr_layer_decay', 0.75)})")
    print(f" ENVIRONMENTS    : {environments}")
    print(f" TRAIN SUBJECTS  : {len(train_subjects)} subjects")
    print(f" VAL SUBJECTS    : {len(val_subjects)} subjects")
    print("=" * 70)

    # 1. Datasets & DataLoaders
    dataset_root = cfg["dataset"]["root_dir"]
    num_points = cfg["model"]["num_points"]
    use_extra_features = cfg["dataset"].get("use_extra_features", True)
    aug_cfg = cfg.get("augmentation", {})

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
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    # 2. Build Model
    model = PointMAEPoseEstimator(
        num_points=num_points,
        num_groups=cfg["model"]["num_groups"],
        group_size=cfg["model"]["group_size"],
        embed_dim=cfg["model"]["embed_dim"],
        depth=cfg["model"]["depth"],
        num_heads=cfg["model"]["num_heads"],
        num_joints=cfg["model"]["num_joints"],
        in_channels=cfg["model"].get("in_channels", 5),
        freeze_encoder=cfg["model"].get("freeze_encoder", False),
        pose_head_type=cfg["model"].get("pose_head_type", "cross_attention"),
        pose_head_depth=cfg["model"].get("pose_head_depth", 2),
        pose_head_dropout=cfg["model"].get("pose_head_dropout", 0.2),
        drop_path_rate=cfg["model"].get("drop_path_rate", 0.1)
    ).to(device)

    # Load Foundation Weights (ShapeNet Point-MAE with channel expansion)
    ckpt_path = cfg["model"]["pretrained_weights"]
    if os.path.exists(ckpt_path):
        load_pretrained_point_mae(model, ckpt_path)
    else:
        print(f"[Warning] Pretrained weights not found at: {ckpt_path}. Training from random initialization.")

    # 3. Loss & Optimizer with LLRD
    loss_alpha = cfg["training"].get("loss_alpha", 1.0)
    loss_bone_weight = cfg["training"].get("loss_bone_weight", 0.5)
    criterion = CompositePoseLoss(alpha=loss_alpha, beta=loss_bone_weight, use_weighted=True).to(device)

    param_groups = get_layer_wise_lr_groups(
        model,
        base_lr=lr,
        weight_decay=cfg["training"].get("weight_decay", 0.05),
        lr_decay=cfg["training"].get("lr_layer_decay", 0.75),
        head_lr_mult=1.5
    )
    optimizer = torch.optim.AdamW(param_groups)

    # Learning Rate Scheduler: CosineAnnealingWarmRestarts with minimum LR
    min_lr = cfg["training"].get("min_lr", 1e-6)
    restart_period = cfg["training"].get("lr_restart_period", 20)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=restart_period,
        T_mult=1,
        eta_min=min_lr
    )

    # Exponential Moving Average (EMA)
    ema = EMAModel(model, decay=cfg["training"].get("ema_decay", 0.999)) if not args.no_ema else None

    # Watchdog & Logger
    watchdog = StallWatchdog(timeout=args.stall_timeout, check_interval=5.0)
    watchdog.start()
    logger = ExperimentLogger(output_dir, cfg["experiment_name"])

    best_val_mpjpe = float("inf")
    patience = cfg["training"].get("early_stopping_patience", 25)
    patience_counter = 0

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss_total = 0.0
            train_mpjpe_mm_total = 0.0
            train_batches = 0

            total_train_steps = min(len(train_loader), args.max_train_batches) if args.max_train_batches else len(train_loader)
            pbar = tqdm(
                enumerate(train_loader),
                total=total_train_steps,
                desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]",
                bar_format="{l_bar}{bar:25}{r_bar}",
                dynamic_ncols=True
            )

            optimizer.zero_grad()

            for batch_idx, batch in pbar:
                if args.max_train_batches and batch_idx >= args.max_train_batches:
                    break
                watchdog.tick(f"Train Epoch {epoch:02d}/{epochs:02d} | Batch {batch_idx + 1}/{total_train_steps}")

                pts = batch['points'].to(device, non_blocking=True)
                gt_pose = batch['skeleton'].to(device, non_blocking=True)

                pred_pose, _ = model(pts)
                loss, loss_dict = criterion(pred_pose, gt_pose)

                # Scale loss for gradient accumulation
                loss_scaled = loss / grad_accum_steps
                loss_scaled.backward()

                # Step optimizer every grad_accum_steps
                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == total_train_steps:
                    if cfg["training"].get("gradient_clip_val"):
                        nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip_val"])
                    optimizer.step()
                    optimizer.zero_grad()

                    if ema is not None:
                        ema.update(model)

                train_loss_total += loss.item()
                train_mpjpe_mm_total += loss_dict['mpjpe_mm']
                train_batches += 1

                gpu_mem_mb = torch.cuda.memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0
                current_head_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f"{loss.item():.3f}",
                    'mpjpe': f"{loss_dict['mpjpe_mm']:.1f}mm",
                    'avg': f"{(train_mpjpe_mm_total / train_batches):.1f}mm",
                    'lr': f"{current_head_lr:.1e}",
                    'vram': f"{gpu_mem_mb:.0f}MB"
                })

            scheduler.step()
            train_avg_loss = train_loss_total / max(1, train_batches)
            train_avg_mpjpe_mm = train_mpjpe_mm_total / max(1, train_batches)

            # -------------------------------------------------------------
            # Validation Loop (Using EMA shadow weights if enabled)
            # -------------------------------------------------------------
            backup_weights = None
            if ema is not None:
                backup_weights = ema.apply_shadow(model)

            model.eval()
            val_loss_total = 0.0
            val_mpjpe_mm_total = 0.0
            val_batches = 0
            env_val_errors = {"E01": [], "E02": [], "E03": [], "E04": []}

            total_val_steps = min(len(val_loader), args.max_val_batches) if args.max_val_batches else len(val_loader)
            pbar_val = tqdm(
                enumerate(val_loader),
                total=total_val_steps,
                desc=f"Epoch {epoch:02d}/{epochs:02d} [Val]  ",
                bar_format="{l_bar}{bar:25}{r_bar}",
                dynamic_ncols=True
            )

            with torch.no_grad():
                for batch_idx, batch in pbar_val:
                    if args.max_val_batches and batch_idx >= args.max_val_batches:
                        break
                    watchdog.tick(f"Val Epoch {epoch:02d}/{epochs:02d} | Batch {batch_idx + 1}/{total_val_steps}")

                    pts = batch['points'].to(device, non_blocking=True)
                    gt_pose = batch['skeleton'].to(device, non_blocking=True)

                    pred_pose, _ = model(pts)
                    loss, loss_dict = criterion(pred_pose, gt_pose)

                    val_loss_total += loss.item()
                    val_mpjpe_mm_total += loss_dict['mpjpe_mm']
                    val_batches += 1

                    envs_in_batch = batch.get('env', [])
                    diff = pred_pose - gt_pose
                    sample_dists = torch.norm(diff, p=2, dim=-1).mean(dim=-1).cpu().numpy()

                    for env_name, dist_m in zip(envs_in_batch, sample_dists):
                        if env_name in env_val_errors:
                            env_val_errors[env_name].append(dist_m)

                    pbar_val.set_postfix({
                        'loss': f"{loss.item():.3f}",
                        'mpjpe': f"{loss_dict['mpjpe_mm']:.1f}mm",
                        'avg': f"{(val_mpjpe_mm_total / val_batches):.1f}mm"
                    })

            val_avg_loss = val_loss_total / max(1, val_batches)
            val_avg_mpjpe_mm = val_mpjpe_mm_total / max(1, val_batches)
            val_avg_mpjpe_m = val_avg_mpjpe_mm / 1000.0

            epoch_env_summary = {}
            for e_name, errs in env_val_errors.items():
                if len(errs) > 0:
                    epoch_env_summary[e_name] = float(np.mean(errs))

            env_str = " | ".join([f"{k}: {v*1000:.1f}mm" for k, v in epoch_env_summary.items()])
            current_lr = optimizer.param_groups[0]['lr']

            print(f" >>> [Epoch {epoch:02d}/{epochs:02d}] Train MPJPE: {train_avg_mpjpe_mm:.1f} mm | "
                  f"Val MPJPE: {val_avg_mpjpe_mm:.1f} mm ({val_avg_mpjpe_m:.4f} m) | LR: {current_lr:.2e}")
            if env_str:
                print(f"     Environment Breakdown -> {env_str}")

            # Log metrics
            logger.log_epoch(
                epoch=epoch,
                train_metrics={'loss': train_avg_loss, 'mpjpe_mm': train_avg_mpjpe_mm},
                val_metrics={'mpjpe': val_avg_mpjpe_m, 'mpjpe_mm': val_avg_mpjpe_mm, 'env_mpjpe': epoch_env_summary},
                lr=current_lr
            )

            # Check for best validation score
            if val_avg_mpjpe_m < best_val_mpjpe:
                best_val_mpjpe = val_avg_mpjpe_m
                patience_counter = 0
                best_path = os.path.join(output_dir, "best_model.pth")
                av2_path = os.path.join(output_dir, "model_av2.pth")
                save_checkpoint(model, optimizer, epoch, {'mpjpe': val_avg_mpjpe_m, 'mpjpe_mm': val_avg_mpjpe_mm, 'env_mpjpe': epoch_env_summary}, best_path)
                save_checkpoint(model, optimizer, epoch, {'mpjpe': val_avg_mpjpe_m, 'mpjpe_mm': val_avg_mpjpe_mm, 'env_mpjpe': epoch_env_summary}, av2_path)
                print(f"     >>> [BEST MODEL SAVED] Val MPJPE: {val_avg_mpjpe_mm:.1f} mm (Model Av2 updated)")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n[Early Stopping] No improvement for {patience} epochs. Stopping early at epoch {epoch}.")
                    if backup_weights is not None:
                        ema.restore(model, backup_weights)
                    break

            # Save Latest Checkpoint
            latest_path = os.path.join(output_dir, "latest_checkpoint.pth")
            save_checkpoint(model, optimizer, epoch, {'mpjpe': val_avg_mpjpe_m, 'env_mpjpe': epoch_env_summary}, latest_path)

            # Restore original weights if EMA was applied for validation
            if backup_weights is not None:
                ema.restore(model, backup_weights)

    finally:
        watchdog.stop()

    print("\n" + "=" * 70)
    print(f" TRAINING FINISHED! Best Validation MPJPE: {best_val_mpjpe:.4f} m ({best_val_mpjpe*1000:.1f} mm)")
    print(f" Checkpoint saved as 'Model Av2': {os.path.join(output_dir, 'model_av2.pth')}")
    print("=" * 70)

    # -------------------------------------------------------------
    # 6. Automatic Reporting Pipeline Generation (docs/report_training/)
    # -------------------------------------------------------------
    print("\n[Reporter] Generating comprehensive multi-environment evaluation assets...")
    reporter = ExperimentReporter(
        exp_name=cfg["experiment_name"],
        config=cfg,
        base_dir="docs/report_training"
    )

    best_path = os.path.join(output_dir, "best_model.pth")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

    model.eval()
    all_joint_errors = []
    all_sample_errors = []
    env_joint_accum = {"E01": [], "E02": [], "E03": [], "E04": []}
    pred_sample = None
    gt_sample = None

    total_eval_steps = min(len(val_loader), args.max_val_batches) if args.max_val_batches else len(val_loader)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, total=total_eval_steps, desc="Final Validation Analysis")):
            if args.max_val_batches and batch_idx >= args.max_val_batches:
                break
            pts = batch['points'].to(device)
            gt_pose = batch['skeleton'].to(device)
            envs_in_batch = batch.get('env', [])

            pred_pose, _ = model(pts)
            diff = pred_pose - gt_pose
            dist_per_joint = torch.norm(diff, p=2, dim=-1).cpu().numpy() # (B, 17)
            dist_per_sample = np.mean(dist_per_joint, axis=-1) # (B,)

            all_joint_errors.append(dist_per_joint)
            all_sample_errors.append(dist_per_sample)

            for env_name, j_dist in zip(envs_in_batch, dist_per_joint):
                if env_name in env_joint_accum:
                    env_joint_accum[env_name].append(j_dist)

            if pred_sample is None and len(pred_pose) > 0:
                pred_sample = pred_pose[0].cpu().numpy()
                gt_sample = gt_pose[0].cpu().numpy()

    all_joint_errors = np.concatenate(all_joint_errors, axis=0) # (N, 17)
    all_sample_errors = np.concatenate(all_sample_errors, axis=0) # (N,)
    per_joint_errors = np.mean(all_joint_errors, axis=0).tolist()

    per_env_mpjpe = {}
    env_keys = ["E01", "E02", "E03", "E04"]
    joint_env_matrix = np.zeros((17, 4), dtype=np.float32)

    for j, env_k in enumerate(env_keys):
        if len(env_joint_accum[env_k]) > 0:
            env_arr = np.array(env_joint_accum[env_k])
            per_env_mpjpe[env_k] = float(np.mean(env_arr))
            joint_env_matrix[:, j] = np.mean(env_arr, axis=0)
        else:
            per_env_mpjpe[env_k] = 0.0

    valid_env_vals = [v for v in per_env_mpjpe.values() if v > 0]
    if len(valid_env_vals) > 1:
        cv = np.std(valid_env_vals) / np.mean(valid_env_vals)
        env_robustness_score = float(max(0.0, 1.0 - cv))
    else:
        env_robustness_score = 1.0

    final_eval = {
        'best_val_mpjpe': best_val_mpjpe,
        'per_joint_errors': per_joint_errors,
        'per_env_mpjpe': per_env_mpjpe,
        'env_robustness_score': env_robustness_score,
        'sample_errors': all_sample_errors.tolist(),
        'joint_env_matrix': joint_env_matrix.tolist()
    }

    hw_info = {
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'cpu': 'AMD Ryzen 7 7700 8-Core Processor'
    }

    report_file = reporter.finalize_report(
        history=logger.history,
        final_eval=final_eval,
        pred_sample=pred_sample,
        gt_sample=gt_sample,
        hardware_info=hw_info,
        sample_errors=all_sample_errors,
        joint_env_matrix=joint_env_matrix
    )

    print(f"\n[Done] Training report and publication assets successfully generated at:\n       {report_file}")


if __name__ == "__main__":
    main()
