import os
import glob
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any


class TemporalPhysicalDataset(Dataset):
    """
    Temporal Physical State Sequence Dataset for Phase 3 (Dynamics Modeling).
    Loads pre-extracted latent physical representations (Z_t) and applies boundary-aware
    sliding windows strictly within each individual action recording (anti-data leakage).
    """
    def __init__(
        self,
        features_dir: str,
        split: str = "train",
        t_in: int = 16,
        t_out: int = 8,
        stride: int = 2,
        normalize_z: bool = False,
        add_noise: bool = False,
        noise_std: float = 0.01,
        preload_ram: bool = True
    ):
        self.features_dir = features_dir
        self.split = split
        self.t_in = t_in
        self.t_out = t_out
        self.window_size = t_in + t_out
        self.stride = stride
        self.normalize_z = normalize_z
        self.add_noise = (add_noise and split == "train")
        self.noise_std = noise_std
        self.preload_ram = preload_ram

        self.split_dir = os.path.join(features_dir, split)
        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        # Load normalization stats if requested
        self.mean_z = None
        self.std_z = None
        if self.normalize_z:
            stats_path = os.path.join(features_dir, "normalization_stats.pt")
            if os.path.exists(stats_path):
                stats = torch.load(stats_path, weights_only=True)
                self.mean_z = stats["mean_z"]
                self.std_z = stats["std_z"]
            else:
                print(f"[TemporalDataset] Warning: {stats_path} not found. Proceeding without Z normalization.")

        # Discover all .pt files in the split directory
        self.file_paths = sorted(glob.glob(os.path.join(self.split_dir, "*.pt")))
        self.loaded_data: List[Dict[str, Any]] = []
        self.samples: List[Tuple[int, int]] = [] # (file_idx, start_frame_offset)

        self._build_index()

    def denormalize_z(self, z_tensor: torch.Tensor) -> torch.Tensor:
        """
        Inverts z-score standardization: Z_denorm = Z_norm * std_z + mean_z.
        Crucial step: must be applied to predicted latents before feeding into the frozen Pose Head probe!
        """
        if self.mean_z is not None and self.std_z is not None:
            device = z_tensor.device
            return z_tensor * self.std_z.to(device) + self.mean_z.to(device)
        return z_tensor

    def _build_index(self):
        print(f"[TemporalDataset: {self.split.upper()}] Indexing sequences from {len(self.file_paths)} action files (window={self.window_size}, stride={self.stride})...")
        skipped_non_contiguous = 0

        for file_idx, fpath in enumerate(self.file_paths):
            try:
                data = torch.load(fpath, weights_only=True)
            except Exception as e:
                print(f"  [Warning] Failed loading {fpath}: {e}")
                continue

            num_frames = data["latent_z"].shape[0]
            # Use original MM-Fi source frame numbers to detect any physical missing frames
            if "source_frame_ids" in data:
                raw_frame_ids = data["source_frame_ids"].numpy()
            else:
                raw_frame_ids = data["frame_ids"].numpy()

            if self.preload_ram:
                self.loaded_data.append(data)

            # Generate sliding windows strictly bounded inside this file
            if num_frames < self.window_size:
                continue

            for start in range(0, num_frames - self.window_size + 1, self.stride):
                # Strict continuity assertion: check delta between successive original frames == 1
                window_frame_ids = raw_frame_ids[start : start + self.window_size]
                diffs = np.diff(window_frame_ids)
                if not np.all(diffs == 1):
                    skipped_non_contiguous += 1
                    continue

                self.samples.append((file_idx, start))

        print(f"[TemporalDataset: {self.split.upper()}] Generated {len(self.samples):,} valid sequence samples. (Skipped non-contiguous windows: {skipped_non_contiguous})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Single-sample extraction.
        Returned Shapes:
            hist_z           : (T_in, 384)
            target_z         : (T_out, 384)
            hist_gt_skel     : (T_in, 17, 3)
            target_gt_skel   : (T_out, 17, 3)
            hist_pred_skel   : (T_in, 17, 3)
            target_pred_skel : (T_out, 17, 3)
            velocity         : (T_in - 1, 17, 3)
        (When wrapped in PyTorch DataLoader, each tensor will have leading batch dimension B: (B, T_in, 384), etc.)
        """
        file_idx, start = self.samples[idx]

        if self.preload_ram:
            data = self.loaded_data[file_idx]
        else:
            data = torch.load(self.file_paths[file_idx], weights_only=True)

        # Slice temporal window
        z_seq = data["latent_z"][start : start + self.window_size]               # (T_in + T_out, 384)
        gt_skel_seq = data["gt_skeleton"][start : start + self.window_size]       # (T_in + T_out, 17, 3)
        pred_skel_seq = data["pred_skeleton"][start : start + self.window_size]   # (T_in + T_out, 17, 3)

        # Partition into historical observation and future prediction
        hist_z = z_seq[:self.t_in].clone()              # (T_in, 384)
        target_z = z_seq[self.t_in:].clone()            # (T_out, 384)

        hist_gt_skel = gt_skel_seq[:self.t_in]          # (T_in, 17, 3)
        target_gt_skel = gt_skel_seq[self.t_in:]        # (T_out, 17, 3)

        hist_pred_skel = pred_skel_seq[:self.t_in]      # (T_in, 17, 3)
        target_pred_skel = pred_skel_seq[self.t_in:]    # (T_out, 17, 3)

        # 1. Optional z-score normalization (fitted exclusively on train set)
        if self.normalize_z and self.mean_z is not None:
            hist_z = (hist_z - self.mean_z) / self.std_z
            target_z = (target_z - self.mean_z) / self.std_z

        # 2. Training-time latent perturbation noise (in standardized latent space)
        if self.add_noise:
            noise = torch.randn_like(hist_z) * self.noise_std
            hist_z = hist_z + noise

        # Velocity / Displacement differential (T_in - 1, 17, 3)
        velocity = hist_gt_skel[1:] - hist_gt_skel[:-1]

        start_frame_id = int(data["source_frame_ids"][start]) if "source_frame_ids" in data else int(data["frame_ids"][start])

        return {
            "hist_z": hist_z,                                               # (T_in, 384)
            "target_z": target_z,                                           # (T_out, 384)
            "hist_gt_skel": hist_gt_skel,                                   # (T_in, 17, 3)
            "target_gt_skel": target_gt_skel,                               # (T_out, 17, 3) (Passive probe target)
            "hist_pred_skel": hist_pred_skel,                               # (T_in, 17, 3)
            "target_pred_skel": target_pred_skel,                           # (T_out, 17, 3)
            "velocity": velocity,                                           # (T_in - 1, 17, 3)
            "action": torch.tensor(data["action_idx"], dtype=torch.long),
            "env": data["metadata"]["env"],
            "sub": data["metadata"]["sub"],
            "act": data["metadata"]["act"],
            "start_frame": start_frame_id
        }


def create_temporal_dataloaders(
    config_path: str,
    batch_size: int = 64,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Factory function to create Train, Val, and Test DataLoaders from configuration.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg["dataset"]
    temp_cfg = cfg.get("temporal", {})

    features_dir = ds_cfg["features_output_dir"]
    t_in = temp_cfg.get("t_in", 16)
    t_out = temp_cfg.get("t_out", 8)
    train_stride = temp_cfg.get("train_stride", 2)
    eval_stride = temp_cfg.get("eval_stride", 4)
    add_noise = temp_cfg.get("add_latent_noise", True)
    noise_std = temp_cfg.get("latent_noise_std", 0.01)

    train_ds = TemporalPhysicalDataset(
        features_dir=features_dir,
        split="train",
        t_in=t_in,
        t_out=t_out,
        stride=train_stride,
        add_noise=add_noise,
        noise_std=noise_std
    )

    val_ds = TemporalPhysicalDataset(
        features_dir=features_dir,
        split="val",
        t_in=t_in,
        t_out=t_out,
        stride=eval_stride,
        add_noise=False
    )

    test_ds = TemporalPhysicalDataset(
        features_dir=features_dir,
        split="test",
        t_in=t_in,
        t_out=t_out,
        stride=eval_stride,
        add_noise=False
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
