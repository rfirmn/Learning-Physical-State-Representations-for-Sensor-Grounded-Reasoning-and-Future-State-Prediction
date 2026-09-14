import numpy as np
import torch

def sample_or_pad_points(pts: np.ndarray, num_points: int = 128) -> np.ndarray:
    """
    Subsamples or pads point cloud to a fixed number of points (num_points).
    pts: (N, C) numpy array where C >= 3.
    Applies strict physical validation to remove any radar buffer sentinel points across all channels.
    """
    if pts is None or len(pts) == 0:
        return np.zeros((num_points, 3), dtype=np.float32)

    C = pts.shape[1] if pts.ndim > 1 else 3
    if pts.ndim == 1:
        pts = pts.reshape(-1, C)

    # Sanitize and remove any sentinel/non-finite coordinates across all channels
    # Physical room boundaries: |x| < 10.0m, |y| < 15.0m, |z| < 10.0m
    spatial_mask = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1]) & np.isfinite(pts[:, 2]) & \
                   (np.abs(pts[:, 0]) < 10.0) & (np.abs(pts[:, 1]) < 15.0) & (np.abs(pts[:, 2]) < 10.0)

    if C >= 5:
        feature_mask = np.isfinite(pts[:, 3]) & np.isfinite(pts[:, 4]) & \
                       (np.abs(pts[:, 3]) < 50.0) & (np.abs(pts[:, 4]) < 1e5)
        valid_mask = spatial_mask & feature_mask
    else:
        valid_mask = spatial_mask

    if np.any(valid_mask):
        clean_pts = pts[valid_mask]
    else:
        clean_pts = np.zeros((1, C), dtype=np.float32)

    n = len(clean_pts)
    if n >= num_points:
        indices = np.random.choice(n, num_points, replace=False)
    else:
        indices = np.random.choice(n, num_points, replace=True)

    return clean_pts[indices].astype(np.float32)


class PointCloudNormalize:
    """
    Centers the point cloud at the origin and scales it into a unit sphere [-1, 1].
    Guarantees numerical stability and prevents overflow.
    If extra channels (e.g. Doppler, SNR) are present, normalizes them robustly.
    """
    def __init__(self, unit_sphere: bool = True):
        self.unit_sphere = unit_sphere

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        # pts: (N, 3) or (N, C)
        xyz = pts[:, :3].copy()

        # Clean clamp coordinates within physical room boundaries [-10.0, 10.0]
        xyz = np.clip(np.nan_to_num(xyz, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid

        if self.unit_sphere:
            dists = np.linalg.norm(xyz, axis=1)
            furthest_distance = float(np.max(dists))
            if furthest_distance > 1e-6 and np.isfinite(furthest_distance):
                xyz = xyz / furthest_distance

        if pts.shape[1] > 3:
            extra = pts[:, 3:].copy()
            # Robust normalization for extra channels (e.g. Doppler and SNR)
            for ch in range(extra.shape[1]):
                col = extra[:, ch]
                mean_val = np.mean(col)
                std_val = np.std(col) + 1e-6
                extra[:, ch] = np.clip((col - mean_val) / std_val, -3.0, 3.0) / 3.0
            return np.concatenate([xyz, extra], axis=1).astype(np.float32)

        return xyz.astype(np.float32)


class PointCloudAugment:
    """
    Applies data augmentations suitable for radar point clouds:
    - Random rotation around the vertical (z) axis.
    - Random tilt around the horizontal (x) axis.
    - Random jitter / Gaussian noise.
    - Random scaling (isotropic or anisotropic).
    - Random point dropout (simulates radar signal dropouts/occlusions).
    - Random translation.
    - If skeleton is provided, spatial transforms (rotation, scale, translation)
      are applied consistently to both point cloud and skeleton.
    """
    def __init__(
        self,
        rotate: bool = True,
        rotate_z_range: float = 30.0, # degrees (+/-)
        rotate_x_range: float = 5.0,  # degrees (+/-)
        jitter: bool = True,
        jitter_sigma: float = 0.015,
        jitter_clip: float = 0.04,
        scale: bool = True,
        scale_range: tuple = (0.92, 1.08),
        anisotropic_scale: bool = True,
        dropout: bool = True,
        dropout_ratio: float = 0.15,
        translate: bool = False,
        translate_range: float = 0.05
    ):
        self.rotate = rotate
        self.rotate_z_range = np.radians(rotate_z_range)
        self.rotate_x_range = np.radians(rotate_x_range)
        self.jitter = jitter
        self.jitter_sigma = jitter_sigma
        self.jitter_clip = jitter_clip
        self.scale = scale
        self.scale_range = scale_range
        self.anisotropic_scale = anisotropic_scale
        self.dropout = dropout
        self.dropout_ratio = dropout_ratio
        self.translate = translate
        self.translate_range = translate_range

    def __call__(self, pts: np.ndarray, skeleton: np.ndarray = None):
        # pts: (N, C), where C >= 3. skeleton: (17, 3) or None
        xyz = pts[:, :3].copy()
        xyz = np.clip(np.nan_to_num(xyz, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
        skel = skeleton.copy() if skeleton is not None else None

        # Point Dropout (only applied to points, simulates radar signal dropout)
        if self.dropout and self.dropout_ratio > 0:
            n_pts = len(xyz)
            drop_num = int(n_pts * np.random.uniform(0.05, self.dropout_ratio))
            if drop_num > 0 and n_pts - drop_num >= 16:
                keep_idx = np.random.choice(n_pts, n_pts - drop_num, replace=False)
                # Resample back to original length
                resample_idx = np.random.choice(keep_idx, n_pts, replace=True)
                xyz = xyz[resample_idx]
                if pts.shape[1] > 3:
                    pts_extra = pts[resample_idx, 3:]
                else:
                    pts_extra = None
            else:
                pts_extra = pts[:, 3:] if pts.shape[1] > 3 else None
        else:
            pts_extra = pts[:, 3:] if pts.shape[1] > 3 else None

        # Centers for in-place body transformations
        centroid_pts = np.mean(xyz, axis=0)
        centroid_skel = np.mean(skel, axis=0) if skel is not None else None

        # 1. Rotation (around body center)
        if self.rotate:
            ang_z = np.random.uniform(-self.rotate_z_range, self.rotate_z_range)
            ang_x = np.random.uniform(-self.rotate_x_range, self.rotate_x_range)

            cos_z, sin_z = np.cos(ang_z), np.sin(ang_z)
            rot_z = np.array([
                [cos_z, -sin_z, 0],
                [sin_z,  cos_z, 0],
                [0,      0,     1]
            ], dtype=np.float32)

            cos_x, sin_x = np.cos(ang_x), np.sin(ang_x)
            rot_x = np.array([
                [1,      0,      0],
                [0,  cos_x, -sin_x],
                [0,  sin_x,  cos_x]
            ], dtype=np.float32)

            rot_matrix = rot_x @ rot_z

            xyz = (xyz - centroid_pts) @ rot_matrix.T + centroid_pts
            if skel is not None:
                skel = (skel - centroid_skel) @ rot_matrix.T + centroid_skel

        # 2. Scaling (around body center)
        if self.scale:
            if self.anisotropic_scale:
                s_factor = np.random.uniform(self.scale_range[0], self.scale_range[1], size=(1, 3)).astype(np.float32)
            else:
                s_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
            xyz = (xyz - centroid_pts) * s_factor + centroid_pts
            if skel is not None:
                skel = (skel - centroid_skel) * s_factor + centroid_skel

        # 3. Translation
        if self.translate and self.translate_range > 0:
            offset = np.random.uniform(-self.translate_range, self.translate_range, size=(1, 3)).astype(np.float32)
            xyz = xyz + offset
            if skel is not None:
                skel = skel + offset

        # 4. Jitter / Noise (sensor noise, added only to point cloud)
        if self.jitter:
            noise = np.clip(
                np.random.normal(0, self.jitter_sigma, size=xyz.shape),
                -self.jitter_clip,
                self.jitter_clip
            ).astype(np.float32)
            xyz = xyz + noise

        # Combine channels
        if pts_extra is not None:
            aug_pts = np.concatenate([xyz, pts_extra], axis=1).astype(np.float32)
        else:
            aug_pts = xyz.astype(np.float32)

        if skeleton is not None:
            return aug_pts, skel.astype(np.float32)
        return aug_pts
