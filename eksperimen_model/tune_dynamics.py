import os
import sys
import copy
import random
import argparse
import yaml
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.abspath("."))

from eksperimen_model.models import TemporalTransformerDynamics, ResidualGRUDynamics, CompositeDynamicsLoss
from eksperimen_model.datasets import TemporalPhysicalDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Trial Hyperparameter Tuning for Latent Dynamics Model")
    parser.add_argument("--config", type=str, default="eksperimen_model/configs/mmfi_dynamics_v3.yaml",
                        help="Path to base configuration YAML")
    parser.add_argument("--output_config", type=str, default="eksperimen_model/configs/mmfi_dynamics_best_tuned.yaml",
                        help="Path to save best tuned configuration YAML")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of exploration trials (default: 20 for marathon mode)")
    parser.add_argument("--epochs_per_trial", type=int, default=12,
                        help="Number of epochs per exploration trial")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for tuning")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Computation device")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_hyperparameters(trial_idx: int) -> Dict[str, Any]:
    # Trial 0 is standard baseline
    if trial_idx == 0:
        return {
            "arch_type": "temporal_transformer",
            "num_layers": 4,
            "num_heads": 6,
            "dim_feedforward": 1024,
            "dropout": 0.1,
            "drop_path_rate": 0.1,
            "lr": 4e-4,
            "weight_decay": 5e-5,
            "loss_cos_weight": 0.2,
            "loss_vel_weight": 0.1,
            "latent_noise_std": 0.01
        }

    # Archetype selection: 75% transformer, 25% GRU
    arch_type = random.choice(["temporal_transformer", "temporal_transformer", "temporal_transformer", "residual_gru"])

    if arch_type == "temporal_transformer":
        num_layers = random.choice([4, 6, 8])
        num_heads = random.choice([6, 8, 12])
        dim_feedforward = random.choice([1024, 1536])
        dropout = random.choice([0.1, 0.15, 0.2])
        drop_path_rate = random.choice([0.05, 0.1, 0.15])
        hidden_dim = None
    else:
        num_layers = random.choice([2, 3, 4])
        hidden_dim = random.choice([384, 512])
        num_heads = None
        dim_feedforward = None
        dropout = random.choice([0.1, 0.15])
        drop_path_rate = None

    lr = float(10 ** random.uniform(np.log10(1.5e-4), np.log10(7.0e-4)))
    weight_decay = float(10 ** random.uniform(np.log10(1e-5), np.log10(3e-4)))
    loss_cos_weight = round(random.uniform(0.1, 0.35), 3)
    loss_vel_weight = round(random.uniform(0.05, 0.25), 3)
    latent_noise_std = random.choice([0.0, 0.005, 0.01, 0.02])

    return {
        "arch_type": arch_type,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "dim_feedforward": dim_feedforward,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "drop_path_rate": drop_path_rate,
        "lr": lr,
        "weight_decay": weight_decay,
        "loss_cos_weight": loss_cos_weight,
        "loss_vel_weight": loss_vel_weight,
        "latent_noise_std": latent_noise_std
    }


def train_and_eval_trial(
    hparams: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Trains model for a set number of epochs and returns (best_val_loss, best_val_mse, best_val_cos).
    """
    if hparams["arch_type"] == "temporal_transformer":
        model = TemporalTransformerDynamics(
            embed_dim=384,
            t_in=16,
            t_out=8,
            num_layers=hparams["num_layers"],
            num_heads=hparams["num_heads"],
            dim_feedforward=hparams["dim_feedforward"],
            dropout=hparams["dropout"],
            drop_path_rate=hparams["drop_path_rate"]
        ).to(device)
    else:
        model = ResidualGRUDynamics(
            embed_dim=384,
            t_in=16,
            t_out=8,
            hidden_dim=hparams["hidden_dim"],
            num_layers=hparams["num_layers"],
            dropout=hparams["dropout"]
        ).to(device)

    loss_fn = CompositeDynamicsLoss(
        loss_cos_weight=hparams["loss_cos_weight"],
        loss_vel_weight=hparams["loss_vel_weight"]
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    best_val_mse = float("inf")
    best_val_cos = 0.0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            hist_z = batch["hist_z"].to(device)
            target_z = batch["target_z"].to(device)

            optimizer.zero_grad()
            pred_z = model(hist_z)
            loss, _ = loss_fn(pred_z, target_z)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # Evaluate on validation split
        model.eval()
        val_losses = []
        val_mses = []
        val_coss = []

        with torch.no_grad():
            for batch in val_loader:
                hist_z = batch["hist_z"].to(device)
                target_z = batch["target_z"].to(device)
                pred_z = model(hist_z)
                loss, metrics = loss_fn(pred_z, target_z)
                val_losses.append(metrics["loss_total"])
                val_mses.append(metrics["loss_mse"])
                val_coss.append(metrics["cos_sim"])

        epoch_val_loss = float(np.mean(val_losses))
        epoch_val_mse = float(np.mean(val_mses))
        epoch_val_cos = float(np.mean(val_coss))

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_mse = epoch_val_mse
            best_val_cos = epoch_val_cos

    return best_val_loss, best_val_mse, best_val_cos


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    print("=" * 82)
    print("      DEEP HYPERPARAMETER TUNING: DYNAMICS MODEL (TAHAP 3 - MARATHON MODE)")
    print("=" * 82)
    print(f" Config: {args.config} | Device: {device} | Total Trials: {args.n_trials}")
    print(f" Epochs/Trial: {args.epochs_per_trial} | Batch Size: {args.batch_size}")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg["dataset"]
    features_dir = ds_cfg["features_output_dir"]
    temp_cfg = cfg.get("temporal", {})

    print(f"\n[Loading Data] Preloading temporal dataset from {features_dir}...")
    train_ds = TemporalPhysicalDataset(
        features_dir=features_dir,
        split="train",
        t_in=temp_cfg.get("t_in", 16),
        t_out=temp_cfg.get("t_out", 8),
        stride=temp_cfg.get("train_stride", 2),
        normalize_z=temp_cfg.get("normalize_z", True),
        add_noise=True,
        noise_std=temp_cfg.get("latent_noise_std", 0.01),
        preload_ram=True
    )
    val_ds = TemporalPhysicalDataset(
        features_dir=features_dir,
        split="val",
        t_in=temp_cfg.get("t_in", 16),
        t_out=temp_cfg.get("t_out", 8),
        stride=temp_cfg.get("eval_stride", 4),
        normalize_z=temp_cfg.get("normalize_z", True),
        add_noise=False,
        preload_ram=True
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"  Train: {len(train_ds):,} sequences | Val: {len(val_ds):,} sequences")

    trial_results: List[Dict[str, Any]] = []
    best_overall_score = float("inf")
    best_hparams: Dict[str, Any] = {}

    start_tune_time = time.time()

    for trial_idx in range(args.n_trials):
        hparams = sample_hyperparameters(trial_idx)
        print(f"\n" + "-" * 82)
        print(f"  TRIAL {trial_idx + 1}/{args.n_trials} [{hparams['arch_type'].upper()}]")
        print(f"  Layers={hparams['num_layers']} | Heads={hparams.get('num_heads', 'N/A')} | LR={hparams['lr']:.2e} | CosW={hparams['loss_cos_weight']}")
        print("-" * 82)

        trial_start = time.time()
        val_loss, val_mse, val_cos = train_and_eval_trial(
            hparams=hparams,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs_per_trial,
            device=device
        )
        trial_time = time.time() - trial_start

        is_best = val_loss < best_overall_score
        if is_best:
            best_overall_score = val_loss
            best_hparams = copy.deepcopy(hparams)

        print(f"  -> Result: Val Loss = {val_loss:.4f} | MSE = {val_mse:.4f} | CosSim = {val_cos:.4f} | Time = {trial_time:.1f}s {'[NEW BEST!]' if is_best else ''}")

        record = {
            "trial": trial_idx + 1,
            "val_loss": val_loss,
            "val_mse": val_mse,
            "val_cos_sim": val_cos,
            "time_sec": round(trial_time, 2),
            "hparams": hparams
        }
        trial_results.append(record)

    total_tune_time = time.time() - start_tune_time

    # Sort trials by validation loss
    trial_results.sort(key=lambda x: x["val_loss"])

    print("\n" + "=" * 82)
    print("                    HYPERPARAMETER TUNING RANKING (TOP 5)")
    print("=" * 82)
    print(f" {'Rank':<5} | {'Trial':<6} | {'Arch':<18} | {'Val Loss':<10} | {'Val MSE':<10} | {'Cos Sim':<10}")
    print(" " + "-" * 78)
    for rank, res in enumerate(trial_results[:5], 1):
        hp = res["hparams"]
        arch_label = f"{hp['arch_type'][:12]}-L{hp['num_layers']}"
        print(f" #{rank:<4} | Trial {res['trial']:<2} | {arch_label:<18} | {res['val_loss']:<10.4f} | {res['val_mse']:<10.4f} | {res['val_cos_sim']:<10.4f}")

    print(f"\n[Winner] Best Trial #{trial_results[0]['trial']} with Val Loss = {trial_results[0]['val_loss']:.4f}")
    print(f"  Hyperparameters: {json.dumps(best_hparams, indent=2)}")

    # Update and export best tuned YAML configuration
    tuned_cfg = copy.deepcopy(cfg)
    tuned_cfg["dynamics_model"]["type"] = best_hparams["arch_type"]
    tuned_cfg["dynamics_model"]["num_layers"] = best_hparams["num_layers"]
    if best_hparams.get("num_heads"):
        tuned_cfg["dynamics_model"]["nhead"] = best_hparams["num_heads"]
    if best_hparams.get("dim_feedforward"):
        tuned_cfg["dynamics_model"]["dim_feedforward"] = best_hparams["dim_feedforward"]
    if best_hparams.get("hidden_dim"):
        tuned_cfg["dynamics_model"]["hidden_dim"] = best_hparams["hidden_dim"]
    tuned_cfg["dynamics_model"]["dropout"] = best_hparams["dropout"]
    if best_hparams.get("drop_path_rate") is not None:
        tuned_cfg["dynamics_model"]["drop_path_rate"] = best_hparams["drop_path_rate"]
    tuned_cfg["dynamics_model"]["loss_cos_weight"] = best_hparams["loss_cos_weight"]
    tuned_cfg["dynamics_model"]["loss_vel_weight"] = best_hparams["loss_vel_weight"]

    # Training parameters
    tuned_cfg["training"] = {
        "lr": best_hparams["lr"],
        "weight_decay": best_hparams["weight_decay"],
        "batch_size": args.batch_size,
        "epochs": 150,
        "early_stopping_patience": 35,
        "warmup_epochs": 5,
        "t_0": 35,
        "ema_decay": 0.999
    }
    tuned_cfg["temporal"]["latent_noise_std"] = best_hparams["latent_noise_std"]

    os.makedirs(os.path.dirname(os.path.abspath(args.output_config)), exist_ok=True)
    with open(args.output_config, "w", encoding="utf-8") as f:
        yaml.dump(tuned_cfg, f, sort_keys=False)

    print(f"[Done] Exported winning configuration -> {args.output_config}")
    print(f"Total Tuning Duration: {total_tune_time / 60:.2f} minutes.")


if __name__ == "__main__":
    main()
