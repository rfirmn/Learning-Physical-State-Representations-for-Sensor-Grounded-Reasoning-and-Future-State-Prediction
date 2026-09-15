"""
train_probe.py
Baseline B2: Direct Linear / Multi-Layer Perceptron (MLP) Probe.
Trains lightweight direct task probes directly from the physical latent representation z_t (384d)
to benchmark raw classification and regression performance against the generative SLM.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Dict, List, Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class ProbeDataset(Dataset):
    """Loads QA records and pairs physical latent z_t with numerical/categorical targets."""
    def __init__(self, jsonl_path: str, features_dir: str, split: str = "train"):
        self.records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))

        self.split_dir = os.path.join(features_dir, split)
        self.feature_cache = {}
        for fname in os.listdir(self.split_dir):
            if fname.endswith(".pt"):
                data = torch.load(os.path.join(self.split_dir, fname), weights_only=True)
                self.feature_cache[fname] = data["latent_z"]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        fname = rec["feature_file"]
        w_start = rec["window_start"]
        t_curr = min(w_start + 15, len(self.feature_cache[fname]) - 1)
        z_t = self.feature_cache[fname][t_curr].float() # (384,)

        t_struct = rec["target_structured"]

        # 1. Posture class: standing=0, lunging=1, squatting=2
        posture_map = {"standing": 0, "lunging": 1, "squatting": 2}
        posture_cls = posture_map.get(t_struct.get("posture", "standing"), 0)

        # 2. Lateral class: left=0, center=1, right=2
        lateral_map = {"left": 0, "center": 1, "right": 2}
        lateral_cls = lateral_map.get(t_struct.get("lateral_position", "center"), 1)

        # 3. Radial direction: approaching=0, receding=1, stationary=2
        dir_map = {"approaching": 0, "receding": 1, "stationary": 2}
        radial_dir = dir_map.get(t_struct.get("radial_direction", "stationary"), 2)

        # 4. Metric values (depth, velocity, wrist distance)
        depth_m = float(t_struct.get("depth_m", 0.0))
        velocity = float(t_struct.get("radial_velocity_mps", 0.0))
        wrist_dist = float(t_struct.get("wrist_distance_m", 0.0))

        return {
            "z_t": z_t,
            "posture_cls": torch.tensor(posture_cls, dtype=torch.long),
            "lateral_cls": torch.tensor(lateral_cls, dtype=torch.long),
            "radial_dir": torch.tensor(radial_dir, dtype=torch.long),
            "depth_m": torch.tensor(depth_m, dtype=torch.float32),
            "velocity": torch.tensor(velocity, dtype=torch.float32),
            "wrist_dist": torch.tensor(wrist_dist, dtype=torch.float32),
            "task_name": rec["task_name"]
        }


class MultiTaskPhysicalProbe(nn.Module):
    """
    Direct Multi-Task MLP Probe: z_t (384d) -> Tasks
    """
    def __init__(self, in_dim: int = 384, hidden_dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.posture_head = nn.Linear(hidden_dim, 3)     # standing, lunging, squatting
        self.lateral_head = nn.Linear(hidden_dim, 3)     # left, center, right
        self.direction_head = nn.Linear(hidden_dim, 3)   # approaching, receding, stationary
        self.depth_head = nn.Linear(hidden_dim, 1)       # depth in meters
        self.velocity_head = nn.Linear(hidden_dim, 1)    # radial velocity in m/s
        self.wrist_head = nn.Linear(hidden_dim, 1)       # wrist distance in meters

    def forward(self, z: torch.Tensor):
        h = self.shared(z)
        return {
            "posture_logits": self.posture_head(h),
            "lateral_logits": self.lateral_head(h),
            "direction_logits": self.direction_head(h),
            "depth_pred": self.depth_head(h).squeeze(-1),
            "velocity_pred": self.velocity_head(h).squeeze(-1),
            "wrist_pred": self.wrist_head(h).squeeze(-1)
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_dir", type=str, default="datasets/MM-Fi_grounded_qa")
    parser.add_argument("--features_dir", type=str, default="datasets/MM-Fi_features_v2")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="eksperimen_model/checkpoints/probe")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Baseline B2 (Direct Probe) on {device}...")

    train_ds = ProbeDataset(os.path.join(args.qa_dir, "mmfi_grounded_qa_train.jsonl"), args.features_dir, "train")
    val_ds = ProbeDataset(os.path.join(args.qa_dir, "mmfi_grounded_qa_val.jsonl"), args.features_dir, "val")
    test_ds = ProbeDataset(os.path.join(args.qa_dir, "mmfi_grounded_qa_test.jsonl"), args.features_dir, "test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = MultiTaskPhysicalProbe().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for b in train_loader:
            z = b["z_t"].to(device)
            preds = model(z)

            l_pos = F.cross_entropy(preds["posture_logits"], b["posture_cls"].to(device))
            l_lat = F.cross_entropy(preds["lateral_logits"], b["lateral_cls"].to(device))
            l_dir = F.cross_entropy(preds["direction_logits"], b["radial_dir"].to(device))

            mask_depth = (b["depth_m"] > 0).to(device)
            l_depth = F.l1_loss(preds["depth_pred"][mask_depth], b["depth_m"].to(device)[mask_depth]) if mask_depth.any() else 0.0

            l_tot = l_pos + l_lat + l_dir + 2.0 * l_depth

            optimizer.zero_grad()
            l_tot.backward()
            optimizer.step()
            total_loss += l_tot.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b in val_loader:
                z = b["z_t"].to(device)
                preds = model(z)
                l_pos = F.cross_entropy(preds["posture_logits"], b["posture_cls"].to(device))
                l_lat = F.cross_entropy(preds["lateral_logits"], b["lateral_cls"].to(device))
                l_dir = F.cross_entropy(preds["direction_logits"], b["radial_dir"].to(device))
                mask_depth = (b["depth_m"] > 0).to(device)
                l_depth = F.l1_loss(preds["depth_pred"][mask_depth], b["depth_m"].to(device)[mask_depth]) if mask_depth.any() else 0.0
                val_loss += (l_pos + l_lat + l_dir + 2.0 * l_depth).item()

        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_probe_model.pth"))

        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

    # Evaluate Test Set (Held-Out Benchmark)
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_probe_model.pth")))
    model.eval()
    correct_pos, correct_lat, correct_dir, total_samples = 0, 0, 0, 0
    depth_errors = []

    with torch.no_grad():
        for b in test_loader:
            z = b["z_t"].to(device)
            preds = model(z)
            p_pos = preds["posture_logits"].argmax(dim=-1).cpu()
            p_lat = preds["lateral_logits"].argmax(dim=-1).cpu()
            p_dir = preds["direction_logits"].argmax(dim=-1).cpu()

            correct_pos += (p_pos == b["posture_cls"]).sum().item()
            correct_lat += (p_lat == b["lateral_cls"]).sum().item()
            correct_dir += (p_dir == b["radial_dir"]).sum().item()
            total_samples += len(z)

            mask = (b["depth_m"] > 0)
            if mask.any():
                err = torch.abs(preds["depth_pred"].cpu()[mask] - b["depth_m"][mask])
                depth_errors.extend(err.tolist())

    metrics = {
        "baseline": "B2_Direct_Probe",
        "test_posture_accuracy": correct_pos / total_samples,
        "test_lateral_accuracy": correct_lat / total_samples,
        "test_direction_accuracy": correct_dir / total_samples,
        "test_depth_mae_meters": float(np.mean(depth_errors)) if depth_errors else 0.0
    }
    print("\n" + "=" * 60)
    print(" BASELINE B2 (DIRECT PROBE) TEST RESULTS ")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    with open(os.path.join(args.output_dir, "probe_benchmark_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[+] Saved metrics to {os.path.join(args.output_dir, 'probe_benchmark_metrics.json')}")


if __name__ == "__main__":
    main()
