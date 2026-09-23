import os
import sys
import glob
import re
import argparse
import hashlib
import json
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath("."))

from eksperimen_model.models import PointMAEPoseEstimator
from eksperimen_model.datasets.transforms import PointCloudNormalize


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Latent Physical State Features (Z_t) via Frozen Model Av2")
    parser.add_argument("--config", type=str, default="eksperimen_model/configs/mmfi_dynamics_v3.yaml",
                        help="Path to phase 3 configuration YAML")
    parser.add_argument("--dry_run", action="store_true",
                        help="Dry run: extract only 2 actions (1 train, 1 test) for rapid verification")
    parser.add_argument("--max_actions", type=int, default=None,
                        help="Maximum total actions to process (for debugging)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Inference batch size for frame feature extraction")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Computation device (cuda or cpu)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing .pt feature files if already extracted")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Feature directory override; use a new directory for recovery runs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for point sampling; saved with each feature artifact")
    return parser.parse_args()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest_sha256(action_dir: str) -> str:
    """Fingerprint exact radar and ground-truth bytes for cache invalidation."""
    frames = sorted(glob.glob(os.path.join(action_dir, "frame*.bin")))
    frame_manifest = []
    for path in frames:
        frame_manifest.append((os.path.basename(path), sha256_file(path)))
    gt_path = os.path.join(action_dir, "ground_truth.npy")
    manifest = {
        "frames": frame_manifest,
        "gt_sha256": sha256_file(gt_path) if os.path.isfile(gt_path) else None,
    }
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()


def load_point_cloud_frame(bin_path: str, num_points: int = 128, use_extra_features: bool = True,
                           sampling_seed: int = 42) -> np.ndarray:
    """
    Reads a single .bin frame, filters radar sentinel values, pads/subsamples, and normalizes.
    """
    normalizer = PointCloudNormalize(unit_sphere=True)
    raw_data = np.fromfile(bin_path, dtype=np.float32)

    if len(raw_data) == 0 or len(raw_data) % 5:
        raise ValueError(f"Malformed or empty radar frame: {bin_path}")
    pc = raw_data.reshape(-1, 5)
    spatial_mask = np.isfinite(pc[:, 0]) & np.isfinite(pc[:, 1]) & np.isfinite(pc[:, 2]) & \
                   (np.abs(pc[:, 0]) < 10.0) & (np.abs(pc[:, 1]) < 15.0) & (np.abs(pc[:, 2]) < 10.0)
    feature_mask = np.isfinite(pc[:, 3]) & np.isfinite(pc[:, 4]) & \
                   (np.abs(pc[:, 3]) < 50.0) & (np.abs(pc[:, 4]) < 1e5)
    pc = pc[spatial_mask & feature_mask]
    if len(pc) == 0:
        raise ValueError(f"No valid radar points: {bin_path}")

    stable_name = "/".join(os.path.normpath(bin_path).split(os.sep)[-4:])
    frame_seed = int.from_bytes(hashlib.sha256(f"{sampling_seed}:{stable_name}".encode()).digest()[:4], "little")
    # Same choice semantics as sample_or_pad_points, without mutating global RNG.
    indices = np.random.RandomState(frame_seed).choice(
        len(pc), num_points, replace=len(pc) < num_points
    )
    pc = pc[indices].astype(np.float32)

    if not use_extra_features:
        pc = pc[:, :3]

    pc = normalizer(pc)
    return pc


def get_subject_split(sub: str, train_subs: set, val_subs: set, test_subs: set) -> Optional[str]:
    if sub in train_subs:
        return "train"
    elif sub in val_subs:
        return "val"
    elif sub in test_subs:
        return "test"
    return None


def extract_action_features(
    action_dir: str,
    model: PointMAEPoseEstimator,
    device: torch.device,
    batch_size: int = 128,
    num_points: int = 128,
    use_extra_features: bool = True,
    sampling_seed: int = 42
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Processes all frames in a single action directory and returns concatenated tensors.
    """
    bin_files = glob.glob(os.path.join(action_dir, "frame*.bin"))
    if not bin_files:
        return None

    # Sort strictly by frame number
    def get_frame_num(fpath: str) -> int:
        m = re.search(r"frame(\d+)\.bin", os.path.basename(fpath))
        return int(m.group(1)) if m else 0

    bin_files = sorted(bin_files, key=get_frame_num)
    frame_nums = [get_frame_num(f) for f in bin_files]
    num_frames = len(bin_files)

    if len(set(frame_nums)) != len(frame_nums) or any(n < 1 for n in frame_nums):
        raise ValueError(f"Duplicate or invalid frame IDs in {action_dir}")

    # Ground truth must align exactly; never synthesize missing labels.
    gt_path = os.path.join(action_dir, "ground_truth.npy")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"Ground truth missing: {gt_path}")
    gt_data = np.load(gt_path, mmap_mode="r", allow_pickle=False)
    if gt_data.ndim != 3 or gt_data.shape[1:] != (17, 3) or max(frame_nums) > len(gt_data):
        raise ValueError(f"Ground truth shape or frame range invalid: {gt_path}")
    gt_skeletons = np.asarray(gt_data[np.asarray(frame_nums) - 1], dtype=np.float32)
    if not np.isfinite(gt_skeletons).all() or np.any(np.all(gt_skeletons == 0, axis=(1, 2))):
        raise ValueError(f"Ground truth contains nonfinite or empty poses: {gt_path}")

    # Process radar frames in batches
    all_z = []
    all_pred_joints = []

    for i in range(0, num_frames, batch_size):
        batch_paths = bin_files[i:i + batch_size]
        batch_pcs = [load_point_cloud_frame(p, num_points=num_points, use_extra_features=use_extra_features,
                                            sampling_seed=sampling_seed) for p in batch_paths]
        batch_tensor = torch.from_numpy(np.stack(batch_pcs, axis=0)).float().to(device) # (B, N, C)

        with torch.no_grad():
            pred_joints, z_t = model(batch_tensor)

        all_z.append(z_t.cpu())
        all_pred_joints.append(pred_joints.cpu())

    latent_z = torch.cat(all_z, dim=0)               # (F, 384)
    pred_skeleton = torch.cat(all_pred_joints, dim=0) # (F, 17, 3)
    gt_skeleton = torch.from_numpy(gt_skeletons).float() # (F, 17, 3)
    source_frame_ids = torch.tensor(frame_nums, dtype=torch.long) # (F,)

    return {
        "latent_z": latent_z,
        "pred_skeleton": pred_skeleton,
        "gt_skeleton": gt_skeleton,
        "source_frame_ids": source_frame_ids,
        "frame_ids": source_frame_ids,
        "original_filenames": [os.path.basename(f) for f in bin_files],
        "num_frames": num_frames
    }


def compute_normalization_stats(train_files: List[str], output_path: str, provenance: dict):
    """
    Computes global mean and std of latent_z vectors ONLY on the training split.
    Guarantees no statistical leakage to val/test sets.
    """
    print(f"\n[Normalization] Computing statistical parameters from {len(train_files)} validated train files...")
    if not train_files:
        raise RuntimeError("No validated train features for normalization statistics")

    all_z_list = []
    total_frames = 0
    source_manifest = []
    for f in sorted(train_files):
        data = torch.load(f, map_location="cpu", weights_only=True)
        latent = data["latent_z"]
        if data.get("provenance") != provenance or latent.ndim != 2 or len(latent) == 0 or not torch.isfinite(latent).all():
            raise ValueError(f"Invalid train feature for normalization: {f}")
        all_z_list.append(latent)
        total_frames += len(latent)
        source_manifest.append((os.path.basename(f), sha256_file(f)))

    all_z = torch.cat(all_z_list, dim=0) # (Total_Frames, 384)
    mean_z = torch.mean(all_z, dim=0)
    std_z = torch.std(all_z, dim=0) + 1e-7

    stats = {
        "mean_z": mean_z,
        "std_z": std_z,
        "total_frames": total_frames,
        "train_files_count": len(train_files),
        "embed_dim": mean_z.shape[0],
        "provenance": provenance,
        "feature_provenance": provenance,
        "source_manifest_sha256": hashlib.sha256(
            json.dumps(source_manifest, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    temporary_path = output_path + ".tmp"
    torch.save(stats, temporary_path)
    os.replace(temporary_path, output_path)
    print(f"  [Done] Saved normalization stats (Files: {len(train_files)}, Total frames: {total_frames:,}) -> {output_path}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # 1. Load Configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    print("=" * 78)
    print("  OFFLINE FEATURE EXTRACTION: LATENT PHYSICAL STATE REPRESENTATION (Z_t)")
    print("=" * 78)
    print(f"  Config File   : {args.config}")
    print(f"  Device        : {device}")
    print(f"  Dry Run Mode  : {args.dry_run}")
    print(f"  Batch Size    : {args.batch_size}")

    ds_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    split_cfg = ds_cfg["split"]

    root_dir = ds_cfg["root_dir"]
    output_dir = args.output_dir or ds_cfg["features_output_dir"]
    ckpt_path = ds_cfg["encoder_checkpoint"]
    provenance = {
        "encoder_sha256": sha256_file(ckpt_path),
        "config_sha256": sha256_file(args.config),
        "preprocessing": "pointcloud_normalize_unit_sphere_v1_strict_frames",
        "sampling_seed": args.seed,
    }

    train_subs = set(split_cfg.get("train_subjects", []))
    val_subs = set(split_cfg.get("val_subjects", []))
    test_subs = set(split_cfg.get("test_subjects", []))

    print(f"  Subjects Split: Train={len(train_subs)} | Val={len(val_subs)} | Test={len(test_subs)}")

    # Ensure output partition directories exist
    for split_name in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split_name), exist_ok=True)

    # 2. Instantiate and Load Frozen Encoder Model Av2
    num_points = model_cfg.get("num_points", 128)
    use_extra_features = ds_cfg.get("use_extra_features", True)
    in_channels = model_cfg.get("in_channels", 5 if use_extra_features else 3)

    print(f"\n[Model] Instantiating PointMAEPoseEstimator (in_channels={in_channels}, embed_dim={model_cfg['embed_dim']})...")
    model = PointMAEPoseEstimator(
        num_points=num_points,
        num_groups=model_cfg["num_groups"],
        group_size=model_cfg["group_size"],
        embed_dim=model_cfg["embed_dim"],
        depth=model_cfg["depth"],
        num_heads=model_cfg["num_heads"],
        num_joints=model_cfg["num_joints"],
        in_channels=in_channels,
        pose_head_type=model_cfg.get("pose_head_type", "cross_attention"),
        pose_head_depth=model_cfg.get("pose_head_depth", 2),
        pose_head_dropout=model_cfg.get("pose_head_dropout", 0.0),
        drop_path_rate=0.0
    ).to(device)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Required encoder checkpoint not found: {ckpt_path}")

    print(f"[Model] Loading frozen weights from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)

    # Freeze entire model
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print("[Model] Weights successfully loaded and 100% frozen (eval mode, zero gradients).")

    # 3. Discover Action Folders
    print(f"\n[Scan] Scanning actions in {root_dir}...")
    action_dirs = []
    environments = ds_cfg.get("environments", ["E01", "E02", "E03", "E04"])

    for env in environments:
        env_path = os.path.join(root_dir, env)
        if not os.path.isdir(env_path):
            continue
        for sub in sorted(os.listdir(env_path)):
            sub_path = os.path.join(env_path, sub)
            if not os.path.isdir(sub_path):
                continue
            split = get_subject_split(sub, train_subs, val_subs, test_subs)
            if not split:
                continue # Subject not in defined split
            for act in sorted(os.listdir(sub_path)):
                act_path = os.path.join(sub_path, act)
                if os.path.isdir(act_path):
                    action_dirs.append({
                        "path": act_path,
                        "env": env,
                        "sub": sub,
                        "act": act,
                        "split": split
                    })

    print(f"[Scan] Found {len(action_dirs)} total action recordings across valid subjects.")

    # In dry run mode, select 1 train and 1 test action
    if args.dry_run:
        train_samples = [d for d in action_dirs if d["split"] == "train"][:1]
        test_samples = [d for d in action_dirs if d["split"] == "test"][:1]
        action_dirs = train_samples + test_samples
        print(f"[Dry Run] Constrained to {len(action_dirs)} actions: {[d['split'] + ':' + d['sub'] + '_' + d['act'] for d in action_dirs]}")
    elif args.max_actions:
        action_dirs = action_dirs[:args.max_actions]
        print(f"[Debug] Constrained to max {len(action_dirs)} actions.")

    # 4. Extract Features
    processed_count = 0
    skipped_count = 0
    total_frames = 0
    rejected_actions = []
    valid_train_files = []

    pbar = tqdm(action_dirs, desc="Extracting Physical Features")
    for act_info in pbar:
        env, sub, act, split = act_info["env"], act_info["sub"], act_info["act"], act_info["split"]
        out_fname = f"{env}_{sub}_{act}.pt"
        out_fpath = os.path.join(output_dir, split, out_fname)
        source_digest = source_manifest_sha256(act_info["path"])

        if os.path.exists(out_fpath) and not args.overwrite:
            existing = torch.load(out_fpath, map_location="cpu", weights_only=True)
            if existing.get("provenance") != provenance or existing.get("source_manifest_sha256") != source_digest:
                raise RuntimeError(f"Stale feature cache {out_fpath}; rerun extraction with --overwrite")
            if split == "train":
                valid_train_files.append(out_fpath)
            skipped_count += 1
            pbar.set_postfix({"skip": skipped_count, "proc": processed_count})
            continue
        if args.overwrite and os.path.exists(out_fpath):
            # A failed re-extraction must not leave a stale .pt for downstream globbing.
            os.remove(out_fpath)

        try:
            feat_dict = extract_action_features(
                action_dir=act_info["path"],
                model=model,
                device=device,
                batch_size=args.batch_size,
                num_points=num_points,
                use_extra_features=use_extra_features,
                sampling_seed=args.seed
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            rejected_actions.append({"recording": f"{env}/{sub}/{act}", "reason": str(exc)})
            continue

        if feat_dict is None:
            continue
        if source_manifest_sha256(act_info["path"]) != source_digest:
            rejected_actions.append({"recording": f"{env}/{sub}/{act}",
                                     "reason": "source_changed_during_extraction"})
            continue

        # Add metadata & save
        act_num = int(re.sub(r"[^\d]", "", act)) if re.search(r"\d+", act) else 0
        save_obj = {
            "latent_z": feat_dict["latent_z"],                 # (F, 384)
            "pred_skeleton": feat_dict["pred_skeleton"],       # (F, 17, 3)
            "gt_skeleton": feat_dict["gt_skeleton"],           # (F, 17, 3)
            "source_frame_ids": feat_dict["source_frame_ids"], # (F,) actual integer frame numbers
            "frame_ids": feat_dict["source_frame_ids"],        # Backward-compatibility alias
            "original_filenames": feat_dict["original_filenames"],
            "action_idx": act_num - 1,                         # 0-indexed action class
            "provenance": provenance,
            "source_manifest_sha256": source_digest,
            "metadata": {
                "env": env,
                "sub": sub,
                "act": act,
                "num_frames": feat_dict["num_frames"],
                "split": split
            }
        }

        temporary_path = out_fpath + ".tmp"
        torch.save(save_obj, temporary_path)
        os.replace(temporary_path, out_fpath)
        if split == "train":
            valid_train_files.append(out_fpath)
        processed_count += 1
        total_frames += feat_dict["num_frames"]
        pbar.set_postfix({"proc": processed_count, "frames": total_frames})

    print(f"\n[Done] Feature extraction finished! Processed: {processed_count}, Skipped: {skipped_count}, Total frames: {total_frames:,}")
    with open(os.path.join(output_dir, "extraction_rejections.json"), "w", encoding="utf-8") as handle:
        json.dump(rejected_actions, handle, indent=2)
    print(f"[Audit] Rejected recordings: {len(rejected_actions)}")

    # 5. Compute Normalization Statistics strictly on train/
    norm_stats_path = os.path.join(output_dir, "normalization_stats.pt")
    if args.dry_run or args.max_actions is not None:
        print("[Normalization] Skipped for partial extraction; full train split required.")
    else:
        compute_normalization_stats(valid_train_files, norm_stats_path, provenance)


if __name__ == "__main__":
    main()
