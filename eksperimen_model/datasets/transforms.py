import numpy as np
import torch

def sample_or_pad_points(pts: np.ndarray, num_points: int = 128) -> np.ndarray:
    """
    Subsamples or pads point cloud to a fixed number of points (num_points).
    pts: (N, C) numpy array where C >= 3.
    Applies strict physical validation to remove any radar buffer sentinel points.
    """
    if pts is None or len(pts) == 0:
        return np.zeros((num_points, 3), dtype=np.float32)

    # Sanitize and remove any sentinel/non-finite coordinates
    # Physical room boundaries: |x| < 10.0m, |y| < 15.0m, |z| < 5.0m
    valid_mask = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1]) & np.isfinite(pts[:, 2]) & \
                 (np.abs(pts[:, 0]) < 10.0) & (np.abs(pts[:, 1]) < 15.0) & (np.abs(pts[:, 2]) < 5.0)

    if np.any(valid_mask):
        clean_pts = pts[valid_mask]
    else:
        clean_pts = np.zeros((1, pts.shape[1] if len(pts.shape) > 1 else 3), dtype=np.float32)

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
            # xyz coordinates are within [-20, 20], max squared sum <= 1200 << float32 limit (3.4e38)
            dists = np.linalg.norm(xyz, axis=1)
            furthest_distance = float(np.max(dists))
            if furthest_distance > 1e-6 and np.isfinite(furthest_distance):
                xyz = xyz / furthest_distance

        if pts.shape[1] > 3:
            return np.concatenate([xyz, pts[:, 3:]], axis=1).astype(np.float32)
        return xyz.astype(np.float32)


class PointCloudAugment:
    """
    Applies data augmentations suitable for radar point clouds:
    - Random rotation around the vertical (z) axis.
    - Random jitter / Gaussian noise.
    - Random scaling.
    """
    def __init__(self, rotate: bool = True, jitter: bool = True, scale: bool = True):
        self.rotate = rotate
        self.jitter = jitter
        self.scale = scale

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        xyz = pts[:, :3].copy()
        xyz = np.clip(np.nan_to_num(xyz, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)

        if self.rotate:
            angle = np.random.uniform(-np.pi / 12, np.pi / 12) # +/- 15 degrees
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a,  cos_a, 0],
                [0,      0,     1]
            ], dtype=np.float32)
            xyz = np.dot(xyz, rot_matrix)

        if self.scale:
            scale_factor = np.random.uniform(0.95, 1.05)
            xyz = xyz * scale_factor

        if self.jitter:
            noise = np.clip(np.random.normal(0, 0.01, size=xyz.shape), -0.03, 0.03)
            xyz = xyz + noise

        if pts.shape[1] > 3:
            return np.concatenate([xyz, pts[:, 3:]], axis=1).astype(np.float32)
        return xyz.astype(np.float32)
