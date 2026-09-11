import os
import glob
import json
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Tuple, Dict
from .transforms import sample_or_pad_points, PointCloudNormalize, PointCloudAugment

class MMFiDataset(Dataset):
    """
    MM-Fi mmWave Radar Dataset Loader for 3D Pose Estimation & Point-MAE Pretraining.
    Matches each radar frame (.bin) with its 3D Human Pose keypoints (ground_truth.npy).
    """
    def __init__(
        self,
        root_dir: str,
        subjects: Optional[List[str]] = None,
        environments: Optional[List[str]] = None,
        actions: Optional[List[str]] = None,
        num_points: int = 128,
        augment: bool = False,
        normalize: bool = False,
        cache_index: bool = True
    ):
        self.root_dir = root_dir
        self.num_points = num_points
        self.augment = PointCloudAugment() if augment else None
        self.normalize = PointCloudNormalize(unit_sphere=True) if normalize else None

        self.samples = [] # List of tuples: (bin_path, gt_npy_path, frame_idx, action_idx, env, sub, act)
        self._build_index(subjects, environments, actions, cache_index)

    def _build_index(
        self,
        subjects: Optional[List[str]],
        environments: Optional[List[str]],
        actions: Optional[List[str]],
        cache_index: bool
    ):
        cache_file = os.path.join(self.root_dir, "_index_cache.json")

        env_list = environments or ['E01', 'E02', 'E03', 'E04']
        subject_set = set(subjects) if subjects else None
        action_set = set(actions) if actions else None

        # Check in-memory index creation
        print(f"[MMFiDataset] Scanning dataset root: {self.root_dir}...")
        for env in env_list:
            env_path = os.path.join(self.root_dir, env)
            if not os.path.isdir(env_path):
                continue
            for sub in sorted(os.listdir(env_path)):
                if subject_set and sub not in subject_set:
                    continue
                sub_path = os.path.join(env_path, sub)
                if not os.path.isdir(sub_path):
                    continue
                for act in sorted(os.listdir(sub_path)):
                    if action_set and act not in action_set:
                        continue
                    act_path = os.path.join(sub_path, act)
                    if not os.path.isdir(act_path):
                        continue

                    gt_path = os.path.join(act_path, "ground_truth.npy")
                    has_gt = os.path.exists(gt_path)

                    bin_files = glob.glob(os.path.join(act_path, "frame*.bin"))
                    action_idx = int(act.replace("A", "")) - 1

                    for bf in bin_files:
                        fname = os.path.basename(bf)
                        match = re.search(r"frame(\d+)\.bin", fname)
                        if match:
                            frame_num = int(match.group(1))
                            frame_idx = frame_num - 1
                        else:
                            frame_idx = 0

                        self.samples.append({
                            'bin': bf,
                            'gt': gt_path if has_gt else None,
                            'frame_idx': frame_idx,
                            'action_idx': action_idx,
                            'env': env,
                            'sub': sub,
                            'act': act
                        })

        print(f"[MMFiDataset] Indexed {len(self.samples)} valid frames across subjects {subjects if subjects else 'ALL'}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # 1. Load mmWave Radar Point Cloud (.bin)
        # Format: float32 array of shape (N, 5): x, y, z, doppler, snr
        raw_data = np.fromfile(sample['bin'], dtype=np.float32)
        if len(raw_data) % 5 == 0 and len(raw_data) > 0:
            pc = raw_data.reshape(-1, 5)
            # Filter out invalid / sentinel radar buffer points (e.g. ±3.689e+19)
            # Realistic indoor physical radar bounds: |x| < 10.0m, |y| < 15.0m, |z| < 5.0m
            valid_mask = np.isfinite(pc[:, 0]) & np.isfinite(pc[:, 1]) & np.isfinite(pc[:, 2]) & \
                         (np.abs(pc[:, 0]) < 10.0) & (np.abs(pc[:, 1]) < 15.0) & (np.abs(pc[:, 2]) < 5.0)
            if np.any(valid_mask):
                pc = pc[valid_mask]
            else:
                pc = np.zeros((self.num_points, 5), dtype=np.float32)
        else:
            pc = np.zeros((self.num_points, 5), dtype=np.float32)

        # 2. Subsample / Pad points to fixed num_points
        pc = sample_or_pad_points(pc, num_points=self.num_points)

        # Extract 3D coordinates (x, y, z)
        xyz = pc[:, :3]

        if self.augment:
            xyz = self.augment(xyz)
        if self.normalize:
            xyz = self.normalize(xyz)

        # 3. Load Ground Truth 3D Skeleton Keypoints (17 joints x 3)
        if sample['gt'] and os.path.exists(sample['gt']):
            # Lazy load or memory-mapped access
            gt_data = np.load(sample['gt'], mmap_mode='r')
            f_idx = sample['frame_idx']
            if f_idx < len(gt_data):
                skeleton = np.array(gt_data[f_idx], dtype=np.float32)
            else:
                skeleton = np.array(gt_data[-1], dtype=np.float32)
        else:
            skeleton = np.zeros((17, 3), dtype=np.float32)

        return {
            'points': torch.from_numpy(xyz).float(),             # Shape: (num_points, 3)
            'skeleton': torch.from_numpy(skeleton).float(),       # Shape: (17, 3)
            'action': torch.tensor(sample['action_idx'], dtype=torch.long),
            'frame_idx': torch.tensor(sample['frame_idx'], dtype=torch.long),
            'env': sample['env'],
            'sub': sample['sub'],
            'act': sample['act']
        }
