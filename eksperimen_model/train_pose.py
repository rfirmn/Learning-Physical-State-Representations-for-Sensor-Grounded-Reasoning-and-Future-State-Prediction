import os
import sys
import time
import threading
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Any, Optional

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath("."))

from eksperimen_model.datasets import MMFiDataset
from eksperimen_model.models import PointMAEPoseEstimator, MPJPELoss
from eksperimen_model.utils import load_pretrained_point_mae, save_checkpoint, ExperimentLogger, ExperimentReporter

class StallWatchdog:
    """
    Non-intrusive background watchdog that monitors loop heartbeat.
    If no batch finishes within `timeout` seconds, an alert banner is printed
    with diagnostic instructions without crashing the process.
    """
    def __init__(self, timeout: float = 30.0, check_interval: float = 5.0):
        self.timeout = timeout
        self.check_interval = check_interval
        self.last_tick = time.time()
        self.current_context = ""
        self.is_running = False
        self.stalled_alerted = False
        self.thread = None
        self._lock = threading.Lock()

    def start(self):
        self.is_running = True
        self.last_tick = time.time()
        self.stalled_alerted = False
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def tick(self, context_msg: str = ""):
        with self._lock:
            now = time.time()
            elapsed = now - self.last_tick
            if self.stalled_alerted and elapsed > 0.1:
                tqdm.write(f"\n>>> [STALL RECOVERED] Loop resumed activity after {elapsed:.1f}s delay. Continuing normal execution...\n")
            self.last_tick = now
            self.current_context = context_msg
            self.stalled_alerted = False

    def pause(self):
        with self._lock:
            self.last_tick = time.time()

    def _monitor(self):
        while self.is_running:
            time.sleep(self.check_interval)
            with self._lock:
                elapsed = time.time() - self.last_tick
                if elapsed > self.timeout and not self.stalled_alerted:
                    self.stalled_alerted = True
                    notice = (
                        f"\n"
                        f"+==========================================================================================+\n"
                        f"| [!] STALL NOTICE: Training loop has not completed a batch for {elapsed:.1f}s (> {self.timeout:.0f}s threshold)   |\n"
                        f"+------------------------------------------------------------------------------------------+\n"
                        f"| Current Step: {self.current_context:<74} |\n"
                        f"| Possible Causes & Diagnostics:                                                           |\n"
                        f"|  1. MM-Fi Disk I/O: Reading radar .bin files may be throttled by disk read speed.       |\n"
                        f"|  2. Windows DataLoader Workers: Multi-worker IPC can occasionally serialize or pause.   |\n"
                        f"|     -> If this persists across batches, rerun with `--num_workers 0`.                    |\n"
                        f"|  3. GPU Kernel / VRAM: PyTorch CUDA execution or memory paging might be saturated.       |\n"
                        f"| Action: Process is NOT terminated. Still waiting for batch completion...                 |\n"
                        f"+==========================================================================================+\n"
                    )
                    tqdm.write(notice)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Point-MAE for 3D Pose Estimation on MM-Fi (Tahap 2)")
    parser.add_argument("--config", type=str, default="eksperimen_model/configs/mmfi_pose_finetune.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Target device (cuda or cpu)")
    parser.add_argument("--env", type=str, default=None, help="Filter specific environment (e.g. E01)")
    parser.add_argument("--train_sub", type=str, nargs="+", default=None, help="Train subjects e.g. S01 S02 ... S08")
    parser.add_argument("--val_sub", type=str, nargs="+", default=None, help="Val subjects e.g. S09 S10")
    parser.add_argument("--num_workers", type=int, default=None, help="Override DataLoader num_workers")
    parser.add_argument("--stall_timeout", type=float, default=35.0, help="Seconds before stall detection notice fires")
    parser.add_argument("--max_train_batches", type=int, default=None, help="Limit number of train batches for fast testing")
    parser.add_argument("--max_val_batches", type=int, default=None, help="Limit number of val batches for fast testing")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load YAML configuration
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Command line overrides
    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    lr = args.lr or cfg["training"]["lr"]
    num_workers = args.num_workers if args.num_workers is not None else cfg["training"].get("num_workers", 4)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    environments = [args.env] if args.env else cfg["dataset"].get("environments", ["E01", "E02", "E03", "E04"])
    train_subjects = args.train_sub or cfg["dataset"]["split"].get("train_subjects", [])
    val_subjects = args.val_sub or cfg["dataset"]["split"].get("val_subjects", [])
    test_subjects = cfg["dataset"]["split"].get("test_subjects", [])

    # Split Integrity & Data Leakage Guard
    overlap_train_test = set(train_subjects).intersection(set(test_subjects))
    overlap_val_test = set(val_subjects).intersection(set(test_subjects))
    overlap_train_val = set(train_subjects).intersection(set(val_subjects))

    if overlap_train_test:
        raise ValueError(f"DATA LEAKAGE DETECTED! Train subjects contain test subjects: {overlap_train_test}")
    if overlap_val_test:
        raise ValueError(f"DATA LEAKAGE DETECTED! Val subjects contain test subjects: {overlap_val_test}")
    if overlap_train_val:
        raise ValueError(f"DATA LEAKAGE DETECTED! Train and Val subjects overlap: {overlap_train_val}")

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    logger = ExperimentLogger(output_dir, cfg["experiment_name"])

    print("=" * 70)
    print(f" EXPERIMENT: {cfg['experiment_name']} (Tahap 2: 3D Pose Estimation)")
    print(f" DEVICE: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f" ENVIRONMENTS: {environments}")
    print(f" TRAIN SUBJECTS ({len(train_subjects)}): {train_subjects}")
    print(f" VAL SUBJECTS   ({len(val_subjects)}): {val_subjects}")
    print(f" TEST SUBJECTS  ({len(test_subjects)}): {test_subjects} (Strictly Isolated)")
    print(f" HYPERPARAMETERS: Batch Size={batch_size} | Epochs={epochs} | LR={lr} | Workers={num_workers}")
    print(f" DATA SPLIT GUARD: Passed (0% Subject Leakage)")
    print("=" * 70)

    # 1. Datasets & DataLoaders
    dataset_root = cfg["dataset"]["root_dir"]
    num_points = cfg["model"]["num_points"]

    train_dataset = MMFiDataset(
        root_dir=dataset_root,
        subjects=train_subjects,
        environments=environments,
        num_points=num_points,
        augment=True,
        normalize=True
    )
    val_dataset = MMFiDataset(
        root_dir=dataset_root,
        subjects=val_subjects,
        environments=environments,
        num_points=num_points,
        augment=False,
        normalize=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
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
        freeze_encoder=cfg["model"].get("freeze_encoder", False)
    ).to(device)

    # Load Foundation Weights (ShapeNet Point-MAE)
    ckpt_path = cfg["model"]["pretrained_weights"]
    if os.path.exists(ckpt_path):
        load_pretrained_point_mae(model, ckpt_path)
    else:
        print(f"[Warning] Pretrained weights not found at: {ckpt_path}. Training from random initialization.")

    # 3. Loss & Optimizer
    criterion = MPJPELoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=cfg["training"].get("weight_decay", 0.05)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=cfg["training"].get("min_lr", 1e-5)
    )

    # 4. Initialize Watchdog Monitor
    watchdog = StallWatchdog(timeout=args.stall_timeout, check_interval=5.0)
    watchdog.start()

    # 5. Training Loop
    best_val_mpjpe = float("inf")

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss_total = 0.0
            train_batches = 0

            total_train_steps = min(len(train_loader), args.max_train_batches) if args.max_train_batches else len(train_loader)
            pbar = tqdm(
                enumerate(train_loader),
                total=total_train_steps,
                desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]",
                bar_format="{l_bar}{bar:25}{r_bar}",
                dynamic_ncols=True
            )

            for batch_idx, batch in pbar:
                if args.max_train_batches and batch_idx >= args.max_train_batches:
                    break
                watchdog.tick(f"Train Epoch {epoch:02d}/{epochs:02d} | Batch {batch_idx + 1}/{total_train_steps}")

                pts = batch['points'].to(device, non_blocking=True)
                gt_pose = batch['skeleton'].to(device, non_blocking=True)

                optimizer.zero_grad()
                pred_pose, z_t = model(pts)
                loss = criterion(pred_pose, gt_pose)

                loss.backward()
                if cfg["training"].get("gradient_clip_val"):
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip_val"])
                optimizer.step()

                train_loss_total += loss.item()
                train_batches += 1

                gpu_mem_mb = torch.cuda.memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}m",
                    'mpjpe': f"{loss.item()*1000:.1f}mm",
                    'avg': f"{(train_loss_total/train_batches)*1000:.1f}mm",
                    'lr': f"{optimizer.param_groups[0]['lr']:.1e}",
                    'vram': f"{gpu_mem_mb:.0f}MB"
                })

            scheduler.step()
            train_mpjpe = train_loss_total / max(1, train_batches)

            # Validation Loop with per-environment tracking
            model.eval()
            val_loss_total = 0.0
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
                    loss = criterion(pred_pose, gt_pose)

                    val_loss_total += loss.item()
                    val_batches += 1

                    # Accumulate per-environment errors
                    envs_in_batch = batch.get('env', [])
                    diff = pred_pose - gt_pose
                    sample_dists = torch.norm(diff, p=2, dim=-1).mean(dim=-1).cpu().numpy() # (B,) in meters

                    for env_name, dist_m in zip(envs_in_batch, sample_dists):
                        if env_name in env_val_errors:
                            env_val_errors[env_name].append(dist_m)

                    pbar_val.set_postfix({
                        'mpjpe': f"{loss.item():.4f}m",
                        'avg': f"{(val_loss_total/val_batches)*1000:.1f}mm"
                    })

            val_mpjpe = val_loss_total / max(1, val_batches)
            current_lr = optimizer.param_groups[0]['lr']

            # Summarize per-env errors for this epoch
            epoch_env_summary = {}
            for e_name, errs in env_val_errors.items():
                if len(errs) > 0:
                    epoch_env_summary[e_name] = float(np.mean(errs))

            env_str = " | ".join([f"{k}: {v*1000:.1f}mm" for k, v in epoch_env_summary.items()])

            # Pretty status printout
            print(f" >>> [Epoch {epoch:02d}/{epochs:02d}] Train: {train_mpjpe*1000:.1f}mm ({train_mpjpe:.4f}m) | "
                  f"Val: {val_mpjpe*1000:.1f}mm ({val_mpjpe:.4f}m) | LR: {current_lr:.2e}")
            if env_str:
                print(f"     Environment Breakdown -> {env_str}")

            # Log metrics
            logger.log_epoch(
                epoch=epoch,
                train_metrics={'loss': train_mpjpe, 'mpjpe_mm': train_mpjpe * 1000},
                val_metrics={'mpjpe': val_mpjpe, 'mpjpe_mm': val_mpjpe * 1000, 'env_mpjpe': epoch_env_summary},
                lr=current_lr
            )

            # Save Best Model (Model Av2)
            if val_mpjpe < best_val_mpjpe:
                best_val_mpjpe = val_mpjpe
                best_path = os.path.join(output_dir, "best_model.pth")
                av2_path = os.path.join(output_dir, "model_av2.pth")
                save_checkpoint(model, optimizer, epoch, {'mpjpe': val_mpjpe, 'env_mpjpe': epoch_env_summary}, best_path)
                save_checkpoint(model, optimizer, epoch, {'mpjpe': val_mpjpe, 'env_mpjpe': epoch_env_summary}, av2_path)
                print(f"     >>> [BEST MODEL SAVED] Val MPJPE: {val_mpjpe:.4f} m ({val_mpjpe*1000:.1f} mm)")

            # Save Latest Checkpoint
            latest_path = os.path.join(output_dir, "latest_checkpoint.pth")
            save_checkpoint(model, optimizer, epoch, {'mpjpe': val_mpjpe, 'env_mpjpe': epoch_env_summary}, latest_path)

    finally:
        watchdog.stop()

    print("\n" + "=" * 70)
    print(f" TRAINING FINISHED! Best Validation MPJPE: {best_val_mpjpe:.4f} m ({best_val_mpjpe*1000:.1f} mm)")
    print(f" Checkpoint saved as 'Model Av2': {os.path.join(output_dir, 'model_av2.pth')}")
    print("=" * 70)

    # 6. Automatic Reporting Pipeline Generation (docs/report_training/)
    print("\n[Reporter] Generating comprehensive multi-environment evaluation assets...")
    reporter = ExperimentReporter(
        exp_name=cfg["experiment_name"],
        config=cfg,
        base_dir="docs/report_training"
    )

    # Load best model for evaluation & visual sample extraction
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

    # Per-environment MPJPE and 17x4 matrix
    per_env_mpjpe = {}
    env_keys = ["E01", "E02", "E03", "E04"]
    joint_env_matrix = np.zeros((17, 4), dtype=np.float32)

    for j, env_k in enumerate(env_keys):
        if len(env_joint_accum[env_k]) > 0:
            env_arr = np.array(env_joint_accum[env_k]) # (N_env, 17)
            per_env_mpjpe[env_k] = float(np.mean(env_arr))
            joint_env_matrix[:, j] = np.mean(env_arr, axis=0)
        else:
            per_env_mpjpe[env_k] = 0.0

    # Calculate Environment Robustness Score (ERS = 1 - CV)
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
