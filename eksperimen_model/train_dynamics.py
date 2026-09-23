import os
import sys
import copy
import argparse
import hashlib
import yaml
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.abspath("."))

from eksperimen_model.models import TemporalTransformerDynamics, ResidualGRUDynamics, CompositeDynamicsLoss
from eksperimen_model.datasets import TemporalPhysicalDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train Latent Dynamics Model with EMA and Warm Restarts")
    parser.add_argument("--config", type=str, default="eksperimen_model/configs/mmfi_dynamics_best_tuned.yaml",
                        help="Path to configuration YAML (defaults to best tuned)")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Total training epochs (default: 150 for marathon)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Computation device")
    parser.add_argument("--output_dir", type=str, default="eksperimen_model/checkpoints/dynamics_stage4_recovery",
                        help="Directory to save dynamics checkpoints")
    parser.add_argument("--features_dir", type=str, default=None,
                        help="Override dataset features_output_dir without modifying the source config")
    parser.add_argument("--dry_run", action="store_true",
                        help="Dry run mode: 2 epochs only")
    return parser.parse_args()


class ModelEMA:
    """
    Exponential Moving Average (EMA) of model parameters for smooth generalization.
    theta_EMA = decay * theta_EMA + (1 - decay) * theta_current
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        model_params = dict(model.named_parameters())
        for name, param in self.shadow.named_parameters():
            if name in model_params:
                param.data.mul_(self.decay).add_(model_params[name].data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()


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
            dropout=dyn_cfg.get("dropout", 0.1),
            drop_path_rate=dyn_cfg.get("drop_path_rate", 0.1)
        )
    else:
        model = ResidualGRUDynamics(
            embed_dim=embed_dim,
            t_in=t_in,
            t_out=t_out,
            hidden_dim=dyn_cfg.get("hidden_dim", 512),
            num_layers=dyn_cfg.get("num_layers", 3),
            dropout=dyn_cfg.get("dropout", 0.1)
        )
    return model


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_training_provenance(
    train_ds: TemporalPhysicalDataset,
    val_ds: TemporalPhysicalDataset,
    features_dir: str,
    config_path: str,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Fingerprint the exact Stage 3 inputs and reject mixed feature lineage."""
    if not train_ds.file_paths or not train_ds.samples or not val_ds.file_paths or not val_ds.samples:
        raise ValueError("Stage 3 requires nonempty train and validation feature windows")
    expected = None
    train_file_hashes = []
    train_frames = 0
    for split, dataset in (("train", train_ds), ("val", val_ds)):
        if not dataset.preload_ram or len(dataset.file_paths) != len(dataset.loaded_data):
            raise ValueError(f"{split} features were not fully preloaded")
        for path, data in zip(dataset.file_paths, dataset.loaded_data):
            if not isinstance(data, dict):
                raise ValueError(f"Unreadable {split} feature file: {path}")
            lineage = data.get("provenance")
            if not isinstance(lineage, dict) or any(
                lineage.get(key) is None for key in
                ("encoder_sha256", "config_sha256", "preprocessing", "sampling_seed")
            ):
                raise ValueError(f"Missing extraction provenance in {path}")
            if expected is None:
                expected = lineage
            elif lineage != expected:
                raise ValueError(f"Mixed feature extraction lineage in {split}: {path}")
            source_hash = data.get("source_manifest_sha256")
            if not isinstance(source_hash, str) or len(source_hash) != 64:
                raise ValueError(f"Missing source manifest hash in {path}")
            latent = data.get("latent_z")
            if not isinstance(latent, torch.Tensor) or latent.ndim != 2 or not torch.isfinite(latent).all():
                raise ValueError(f"Invalid latent state in {path}")
            if split == "train":
                train_file_hashes.append((os.path.basename(path), sha256_file(path)))
                train_frames += len(latent)

    normalization_path = os.path.join(features_dir, "normalization_stats.pt")
    normalization_hash = None
    # Match extract_physical_features.compute_normalization_stats exactly.
    source_digest = hashlib.sha256(
        json.dumps(train_file_hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if cfg["temporal"].get("normalize_z", False):
        if not os.path.isfile(normalization_path):
            raise FileNotFoundError(f"Stage 3 normalization stats missing: {normalization_path}")
        stats = torch.load(normalization_path, map_location="cpu", weights_only=True)
        if (
            stats.get("feature_provenance") != expected
            or stats.get("provenance") != expected
            or stats.get("source_manifest_sha256") != source_digest
            or stats.get("train_files_count") != len(train_file_hashes)
            or stats.get("total_frames") != train_frames
        ):
            raise ValueError("Normalization stats do not match the exact training feature set")
        mean_z, std_z = stats.get("mean_z"), stats.get("std_z")
        if (
            not isinstance(mean_z, torch.Tensor) or not isinstance(std_z, torch.Tensor)
            or mean_z.shape != (cfg["dynamics_model"]["d_model"],)
            or std_z.shape != mean_z.shape
            or not torch.isfinite(mean_z).all() or not torch.isfinite(std_z).all()
            or (std_z <= 0).any()
        ):
            raise ValueError("Invalid Stage 3 normalization statistics")
        normalization_hash = sha256_file(normalization_path)

    return {
        "config_sha256": sha256_file(config_path),
        "effective_config_sha256": _json_digest(cfg),
        "normalization_stats_sha256": normalization_hash,
        "feature_provenance": copy.deepcopy(expected),
        "train_feature_manifest_sha256": source_digest,
        "train_feature_file_count": len(train_file_hashes),
    }


def plot_dynamics_curves(history: Dict[str, list], output_path: str):
    """
    Plots training loss, validation loss, validation MSE, and cosine similarity.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Total Loss
    axes[0].plot(history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=1.8)
    axes[0].plot(history["val_loss"], label="Val Loss (EMA)", color="#ff7f0e", linewidth=1.8)
    axes[0].set_title("Composite Loss Curve (MSE + Cos + Vel)", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Latent MSE
    axes[1].plot(history["val_mse"], label="Val Latent MSE", color="#2ca02c", linewidth=1.8)
    axes[1].set_title("Validation Latent MSE (per dimension)", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Cosine Similarity
    axes[2].plot(history["val_cos_sim"], label="Val Cosine Alignment", color="#d62728", linewidth=1.8)
    axes[2].set_title("Validation Directional Cosine Similarity", fontsize=13)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Cosine Similarity (higher is better)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[Done] Saved dynamics training curve -> {output_path}")


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Fallback to base config if best_tuned does not exist yet
    config_path = args.config
    if not os.path.exists(config_path):
        fallback = "eksperimen_model/configs/mmfi_dynamics_v3.yaml"
        print(f"[Warning] Config {config_path} not found. Falling back to base: {fallback}")
        config_path = fallback

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.features_dir:
        cfg["dataset"]["features_output_dir"] = args.features_dir

    # Resolve parameters
    ds_cfg = cfg["dataset"]
    temp_cfg = cfg.get("temporal", {})
    dyn_cfg = cfg.get("dynamics_model", {})
    train_cfg = cfg.get("training", {})

    t_in = temp_cfg.get("t_in", 16)
    t_out = temp_cfg.get("t_out", 8)
    features_dir = ds_cfg["features_output_dir"]

    epochs = 2 if args.dry_run else (args.epochs or train_cfg.get("epochs", 150))
    batch_size = args.batch_size or train_cfg.get("batch_size", 64)
    lr = args.lr or train_cfg.get("lr", 3.5e-4)
    weight_decay = train_cfg.get("weight_decay", 5e-5)
    t_0 = train_cfg.get("t_0", 35)
    ema_decay = train_cfg.get("ema_decay", 0.999)
    patience = 5 if args.dry_run else train_cfg.get("early_stopping_patience", 35)

    loss_cos_w = dyn_cfg.get("loss_cos_weight", 0.2)
    loss_vel_w = dyn_cfg.get("loss_vel_weight", 0.1)

    print("=" * 82)
    print("      FULL SCALE TRAINING: FROZEN-REPRESENTATION LATENT DYNAMICS MODEL")
    print("=" * 82)
    print(f" Config      : {config_path}")
    print(f" Device      : {device}")
    print(f" Model Type  : {dyn_cfg.get('type', 'temporal_transformer').upper()}")
    print(f" Epochs      : {epochs} | Batch Size: {batch_size} | LR: {lr:.2e}")
    print(f" Cosine W    : {loss_cos_w} | Velocity W: {loss_vel_w} | EMA Decay: {ema_decay}")
    print(f" Output Dir  : {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Datasets
    print(f"\n[DataLoader] Loading temporal dataset from {features_dir}...")
    train_ds = TemporalPhysicalDataset(
        features_dir=features_dir,
        split="train",
        t_in=t_in,
        t_out=t_out,
        stride=temp_cfg.get("train_stride", 2),
        normalize_z=temp_cfg.get("normalize_z", True),
        add_noise=temp_cfg.get("add_latent_noise", True),
        noise_std=temp_cfg.get("latent_noise_std", 0.01),
        preload_ram=True
    )
    val_ds = TemporalPhysicalDataset(
        features_dir=features_dir,
        split="val",
        t_in=t_in,
        t_out=t_out,
        stride=temp_cfg.get("eval_stride", 4),
        normalize_z=temp_cfg.get("normalize_z", True),
        add_noise=False,
        preload_ram=True
    )
    checkpoint_provenance = build_training_provenance(train_ds, val_ds, features_dir, config_path, cfg)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Train: {len(train_ds):,} sequences | Val: {len(val_ds):,} sequences")

    # 2. Build Model & Loss
    model = build_dynamics_model(dyn_cfg, t_in=t_in, t_out=t_out).to(device)
    loss_fn = CompositeDynamicsLoss(loss_cos_weight=loss_cos_w, loss_vel_weight=loss_vel_w).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable Parameters: {total_params:,} (~{total_params / 1e6:.2f}M)")

    # 3. Setup Optimizer, LR Schedule, & EMA
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=t_0, T_mult=1, eta_min=1e-6)
    ema = ModelEMA(model, decay=ema_decay)

    # 4. Training Loop
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mse": [],
        "val_cos_sim": [],
        "lr": []
    }

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    best_model_path = os.path.join(args.output_dir, "best_dynamics_model.pth")
    latest_model_path = os.path.join(args.output_dir, "latest_dynamics_checkpoint.pth")
    history_json_path = os.path.join(args.output_dir, "dynamics_training_history.json")
    loss_plot_path = os.path.join(args.output_dir, "dynamics_loss_curve.png")

    print("\n[Start Training] Initiating training loop...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{epochs:03d}", leave=False)
        for batch in pbar:
            hist_z = batch["hist_z"].to(device)
            target_z = batch["target_z"].to(device)

            optimizer.zero_grad()
            pred_z = model(hist_z)
            loss, metrics = loss_fn(pred_z, target_z)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update EMA shadow weights
            ema.update(model)

            train_losses.append(metrics["loss_total"])
            pbar.set_postfix({"loss": f"{metrics['loss_total']:.4f}", "mse": f"{metrics['loss_mse']:.4f}"})

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Validation with EMA weights
        val_losses = []
        val_mses = []
        val_coss = []

        with torch.no_grad():
            for batch in val_loader:
                hist_z = batch["hist_z"].to(device)
                target_z = batch["target_z"].to(device)
                pred_z = ema.shadow(hist_z)
                loss, metrics = loss_fn(pred_z, target_z)
                val_losses.append(metrics["loss_total"])
                val_mses.append(metrics["loss_mse"])
                val_coss.append(metrics["cos_sim"])

        train_loss_epoch = float(np.mean(train_losses))
        val_loss_epoch = float(np.mean(val_losses))
        val_mse_epoch = float(np.mean(val_mses))
        val_cos_epoch = float(np.mean(val_coss))
        epoch_dur = time.time() - epoch_start

        history["train_loss"].append(train_loss_epoch)
        history["val_loss"].append(val_loss_epoch)
        history["val_mse"].append(val_mse_epoch)
        history["val_cos_sim"].append(val_cos_epoch)
        history["lr"].append(current_lr)

        is_best = val_loss_epoch < best_val_loss
        if is_best:
            best_val_loss = val_loss_epoch
            best_epoch = epoch
            patience_counter = 0

            # Save best checkpoint (saving EMA weights as the primary model)
            torch.save({
                "epoch": epoch,
                "model_state_dict": ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss_epoch,
                "val_mse": val_mse_epoch,
                "val_cos_sim": val_cos_epoch,
                "config": cfg,
                **checkpoint_provenance,
            }, best_model_path)
        else:
            patience_counter += 1

        print(f" Epoch {epoch:03d}/{epochs:03d} [{epoch_dur:.1f}s] | Train: {train_loss_epoch:.4f} | Val: {val_loss_epoch:.4f} (MSE: {val_mse_epoch:.4f}, Cos: {val_cos_epoch:.4f}) | LR: {current_lr:.2e} {'[BEST]' if is_best else ''}")

        # Save latest checkpoint every 5 epochs
        if epoch % 5 == 0 or epoch == epochs:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "val_loss": val_loss_epoch,
                "config": cfg,
                **checkpoint_provenance,
            }, latest_model_path)

            with open(history_json_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

        # Early Stopping Check
        if patience_counter >= patience:
            print(f"\n[Early Stopping] No validation improvement for {patience} epochs. Stopping at Epoch {epoch}.")
            break

    total_time = time.time() - start_time
    print("\n" + "=" * 82)
    print(f" [TRAINING COMPLETED] Best Val Loss: {best_val_loss:.4f} achieved at Epoch {best_epoch}")
    print(f" Total Elapsed Time: {total_time / 60:.2f} minutes")
    print(f" Best Official Model Checkpoint: {best_model_path}")
    print("=" * 82)

    # Plot final curves
    plot_dynamics_curves(history, loss_plot_path)


if __name__ == "__main__":
    main()
