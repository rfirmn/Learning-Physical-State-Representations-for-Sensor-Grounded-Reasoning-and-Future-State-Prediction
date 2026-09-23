"""Generate deterministic Stage 4 QA from validated MM-Fi feature recordings.

The encoder normalizes radar points per frame, so questions use relative body
geometry. Train/val probes must confirm that the global latent carries it.
"""

import argparse
import csv
import glob
import json
import os
import re
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm


HISTORY = 16
HORIZON = 8
WINDOW_SIZE = HISTORY + HORIZON
LEFT_SHOULDER, LEFT_WRIST = 11, 13
RIGHT_SHOULDER, RIGHT_WRIST = 14, 16
MIN_SHOULDER_WIDTH_M = 0.05
CURRENT_MARGIN = 0.10
CHANGE_THRESHOLD = 0.20

# Canonical benchmark prompts; paraphrase generalization is a separate experiment.
QUESTIONS = {
    "current_wrist_separation": (
        "At the current frame t, is the 3D distance between the wrists "
        "narrower or wider than the 3D distance between the shoulders?"
    ),
    "future_wrist_separation_change": (
        "From the current frame t to frame t+8, does the 3D distance between "
        "the wrists close, stay stable, or open? Classify a change as closing "
        "or opening only when its magnitude exceeds 0.20 times the shoulder "
        "distance at t."
    ),
}


def load_segments(csv_path):
    """Read recording segments, preserving inverted ranges for quarantine."""
    segments = {}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["Environment"], row["Student"], row["Action"])
            if key in segments:
                raise ValueError(f"Duplicate segment metadata for {key}")
            ranges = []
            for ordinal, item in enumerate(row["Segments"].split(";"), start=1):
                match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", item)
                if not match:
                    raise ValueError(f"Malformed segment range {item!r} for {key}")
                ranges.append((ordinal, *map(int, match.groups())))
            segments[key] = ranges
    return segments


def physical_state(gt_window):
    """Compute the full relative state once from a valid 24-frame window."""
    current, future = gt_window[HISTORY - 1], gt_window[-1]
    shoulder_width = float(np.linalg.norm(
        current[LEFT_SHOULDER] - current[RIGHT_SHOULDER]
    ))
    if shoulder_width <= MIN_SHOULDER_WIDTH_M:
        raise ValueError("degenerate_shoulder_width")
    current_ratio = float(np.linalg.norm(
        current[LEFT_WRIST] - current[RIGHT_WRIST]
    )) / shoulder_width
    future_ratio = float(np.linalg.norm(
        future[LEFT_WRIST] - future[RIGHT_WRIST]
    )) / shoulder_width
    change = future_ratio - current_ratio

    if current_ratio < 1 - CURRENT_MARGIN:
        current_class = "narrower"
    elif current_ratio > 1 + CURRENT_MARGIN:
        current_class = "wider"
    else:
        current_class = None

    if change < -CHANGE_THRESHOLD:
        future_class = "closing"
    elif change > CHANGE_THRESHOLD:
        future_class = "opening"
    else:
        future_class = "stable"

    return {
        "current_wrist_to_shoulder_ratio": current_ratio,
        "future_wrist_to_current_shoulder_ratio": future_ratio,
        "wrist_separation_change_shoulder_widths": change,
        "current_wrist_separation": current_class,
        "future_wrist_separation_change": future_class,
    }


def validate_window(gt_window, frame_ids, ranges, raw_in_range, raw_matches):
    """Return (rejection reason, segment provenance)."""
    if not np.all(np.diff(frame_ids) == 1):
        return "nonconsecutive_source_frames", None
    if not np.all(raw_in_range):
        return "raw_gt_out_of_range", None
    if not np.all(raw_matches):
        return "raw_gt_mismatch", None
    if not np.isfinite(gt_window).all():
        return "nonfinite_gt", None
    if np.any(np.all(gt_window == 0, axis=(1, 2))):
        return "zero_gt_frame", None
    if np.any(np.all(gt_window[1:] == gt_window[:-1], axis=(1, 2))):
        return "repeated_gt_frame", None

    frame_segments = []
    for frame_id in frame_ids:
        # Quarantine every frame touched by an inverted CSV range.
        if any(last < first and last <= frame_id <= first for _, first, last in ranges):
            return "inverted_segment_range", None
        matches = [number for number, first, last in ranges if first <= frame_id <= last]
        if len(matches) != 1:
            return "missing_or_overlapping_segment", None
        frame_segments.append(matches[0])

    segment_ids = list(dict.fromkeys(frame_segments))
    return None, {
        "segment_ids": segment_ids,
        "current_segment_id": frame_segments[HISTORY - 1],
        "future_segment_id": frame_segments[-1],
        "boundary_frame_ids": [
            int(frame_ids[i]) for i in range(1, WINDOW_SIZE)
            if frame_segments[i] != frame_segments[i - 1]
        ],
        "crosses_segment_boundary": len(segment_ids) > 1,
    }


def load_recording(path, split, raw_dataset_dir):
    """Require matching metadata, frame IDs, GT, and latent lengths."""
    data = torch.load(path, weights_only=True)
    name = os.path.basename(path)
    match = re.fullmatch(r"(E\d+)_(S\d+)_(A\d+)\.pt", name)
    if not match:
        raise ValueError("invalid_feature_filename")
    env, subject, action = match.groups()
    metadata = data.get("metadata")
    if not isinstance(metadata, dict) or any((
        metadata.get("env") != env,
        metadata.get("sub") != subject,
        metadata.get("act") != action,
        metadata.get("split") != split,
    )):
        raise ValueError("metadata_mismatch")
    if "source_frame_ids" not in data:
        raise ValueError("missing_source_frame_ids")
    gt = np.asarray(data["gt_skeleton"])
    latent = np.asarray(data["latent_z"])
    ids = np.asarray(data["source_frame_ids"])
    if gt.ndim != 3 or gt.shape[1:] != (17, 3) or latent.ndim != 2 or ids.ndim != 1:
        raise ValueError("invalid_feature_shape")
    if not (len(gt) == len(latent) == len(ids) == metadata.get("num_frames")):
        raise ValueError("feature_length_mismatch")
    if not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 1) or np.any(np.diff(ids) <= 0):
        raise ValueError("invalid_source_frame_ids")
    if "frame_ids" in data and not np.array_equal(ids, np.asarray(data["frame_ids"])):
        raise ValueError("frame_id_alias_mismatch")
    filenames = data.get("original_filenames")
    if not isinstance(filenames, list) or len(filenames) != len(ids) or any(
        not (match := re.fullmatch(r"frame(\d+)\.bin", filename))
        or int(match.group(1)) != int(frame_id)
        for filename, frame_id in zip(filenames, ids)
    ):
        raise ValueError("source_filename_mismatch")
    if data.get("action_idx") != int(action[1:]) - 1:
        raise ValueError("action_index_mismatch")
    provenance = data.get("provenance")
    if not isinstance(provenance, dict) or not all(
        provenance.get(field) for field in ("encoder_sha256", "config_sha256", "preprocessing")
    ) or provenance.get("sampling_seed") is None:
        raise ValueError("missing_feature_provenance")
    if not np.isfinite(latent).all():
        raise ValueError("nonfinite_latent")
    raw_path = os.path.join(raw_dataset_dir, env, subject, action, "ground_truth.npy")
    if not os.path.isfile(raw_path):
        raise ValueError("missing_raw_gt")
    raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    if raw.ndim != 3 or raw.shape[1:] != (17, 3):
        raise ValueError("invalid_raw_gt_shape")
    raw_in_range = ids <= len(raw)
    raw_matches = np.zeros(len(ids), dtype=bool)
    valid_ids = ids[raw_in_range]
    raw_matches[raw_in_range] = np.all(
        gt[raw_in_range] == np.asarray(raw[valid_ids - 1], dtype=gt.dtype), axis=(1, 2)
    )
    return gt, ids, (env, subject, action), raw_in_range, raw_matches, provenance


def generate_split_qa(split, features_dir, output_file, segments, raw_dataset_dir, stride=4):
    """Write accepted QA and a deterministic ledger of rejected candidates."""
    if stride < 1:
        raise ValueError("stride must be positive")
    pt_files = sorted(glob.glob(os.path.join(features_dir, split, "*.pt")))
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    rejection_file = output_file.removesuffix(".jsonl") + "_rejections.jsonl"
    counts, labels = Counter(), Counter()
    subjects, actions, boundaries = Counter(), Counter(), Counter()
    sample_number = 0
    expected_provenance = None

    with open(output_file, "w", encoding="utf-8") as accepted, \
            open(rejection_file, "w", encoding="utf-8") as rejected:
        for path in tqdm(pt_files, desc=f"Generating {split} QA"):
            feature_file = os.path.basename(path)
            try:
                gt, ids, key, raw_in_range, raw_matches, provenance = load_recording(
                    path, split, raw_dataset_dir
                )
                if expected_provenance is None:
                    expected_provenance = provenance
                elif provenance != expected_provenance:
                    raise ValueError("feature_provenance_mismatch")
            except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
                reason = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
                rejected.write(json.dumps({
                    "feature_file": feature_file, "window_start": None, "reason": reason
                }) + "\n")
                counts[f"rejected:{reason}"] += 1
                continue
            env, subject, action = key
            ranges = segments.get(key)
            if ranges is None or len(gt) < WINDOW_SIZE:
                reason = "missing_segment_metadata" if ranges is None else "short_recording"
                rejected.write(json.dumps({
                    "feature_file": feature_file, "window_start": None, "reason": reason
                }) + "\n")
                counts[f"rejected:{reason}"] += 1
                continue

            for start in range(0, len(gt) - WINDOW_SIZE + 1, stride):
                counts["candidate_windows"] += 1
                window_ids = ids[start:start + WINDOW_SIZE]
                window_id = f"{split}:{env}_{subject}_{action}:{int(window_ids[0])}-{int(window_ids[-1])}"
                gt_window = gt[start:start + WINDOW_SIZE]
                reason, provenance = validate_window(
                    gt_window, window_ids, ranges,
                    raw_in_range[start:start + WINDOW_SIZE],
                    raw_matches[start:start + WINDOW_SIZE],
                )
                if reason is None:
                    try:
                        state = physical_state(gt_window)
                    except ValueError as exc:
                        reason = str(exc)
                if reason:
                    for task in QUESTIONS:
                        rejected.write(json.dumps({
                            "window_id": window_id, "feature_file": feature_file,
                            "window_start": start, "source_frame_ids": window_ids.tolist(),
                            "task_name": task, "reason": reason
                        }) + "\n")
                        counts[f"rejected:{reason}"] += 1
                    continue
                counts["valid_windows"] += 1
                if provenance["crosses_segment_boundary"]:
                    counts["valid_cross_boundary_windows"] += 1

                for task, question in QUESTIONS.items():
                    label = state[task]
                    if label is None:
                        rejected.write(json.dumps({
                            "window_id": window_id, "feature_file": feature_file,
                            "window_start": start, "source_frame_ids": window_ids.tolist(),
                            "task_name": task, "reason": "ambiguous_current_ratio"
                        }) + "\n")
                        counts["rejected:ambiguous_current_ratio"] += 1
                        continue
                    sample_number += 1
                    target = {task: label}
                    record = {
                        "sample_id": f"{split}_{sample_number:06d}",
                        "window_id": window_id,
                        "feature_file": feature_file,
                        "window_start": start,
                        "t_curr_frame": int(window_ids[HISTORY - 1]),
                        "t_future_frame": int(window_ids[-1]),
                        "source_frame_ids": window_ids.tolist(),
                        "environment": env,
                        "subject": subject,
                        "action": action,
                        "recording_id": f"{env}_{subject}_{action}",
                        **provenance,
                        "task_name": task,
                        "question": question,
                        "target_structured": target,
                        "target_text": json.dumps(target, separators=(",", ":")),
                        "physical_state": state,
                    }
                    accepted.write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts["accepted"] += 1
                    counts[f"task:{task}"] += 1
                    labels[f"{task}:{label}"] += 1
                    subjects[subject] += 1
                    actions[action] += 1
                    boundaries[str(provenance["crosses_segment_boundary"])] += 1

    print(f"[QA Generator] {split}: {sample_number} accepted")
    return {
        "samples": sample_number,
        "feature_files": len(pt_files),
        "counts": dict(sorted(counts.items())),
        "labels": dict(sorted(labels.items())),
        "subjects": dict(sorted(subjects.items())),
        "actions": dict(sorted(actions.items())),
        "crosses_segment_boundary": dict(sorted(boundaries.items())),
        "rejections_file": os.path.basename(rejection_file),
        "feature_provenance": expected_provenance,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate validated relative-geometry Stage 4 QA")
    parser.add_argument("--features_dir", default="datasets/MM-Fi_features_v2")
    parser.add_argument("--output_dir", default="datasets/MM-Fi_grounded_qa_stage4_recovery")
    parser.add_argument("--segments_csv", default="datasets/MM-Fi Dataset/MMFi_action_segments.csv")
    parser.add_argument("--raw_dataset_dir", default="datasets/MM-Fi Dataset/filtered_mmwave",
                        help="Required raw MM-Fi root for exact GT correspondence; missing files reject")
    parser.add_argument("--train_stride", type=int, default=4)
    parser.add_argument("--val_stride", type=int, default=8)
    parser.add_argument("--test_stride", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42,
                        help="Recorded for compatibility; canonical generation uses no RNG")
    args = parser.parse_args()

    for split in ("train", "val", "test"):
        split_dir = os.path.join(args.features_dir, split)
        if not os.path.isdir(split_dir) or not glob.glob(os.path.join(split_dir, "*.pt")):
            raise FileNotFoundError(f"No extracted feature .pt files for split {split}: {split_dir}")

    segments = load_segments(args.segments_csv)
    results = {}
    for split in ("train", "val", "test"):
        output = os.path.join(args.output_dir, f"mmfi_grounded_qa_{split}.jsonl")
        results[split] = generate_split_qa(
            split, args.features_dir, output, segments, args.raw_dataset_dir,
            getattr(args, f"{split}_stride")
        )
    provenances = [result["feature_provenance"] for result in results.values()
                   if result["feature_provenance"] is not None]
    if any(provenance != provenances[0] for provenance in provenances[1:]):
        raise ValueError("Feature provenance differs between splits")
    seen_subjects = set()
    for split in ("train", "val", "test"):
        split_subjects = set(results[split]["subjects"])
        if seen_subjects & split_subjects:
            raise ValueError("Subjects overlap between QA splits")
        seen_subjects.update(split_subjects)
    summary = {
        "schema_version": 2,
        "seed": args.seed,
        "frame_rate_hz_assumed": 10,
        "history_frames": HISTORY,
        "future_frames": HORIZON,
        "current_margin_shoulder_widths": CURRENT_MARGIN,
        "future_change_threshold_shoulder_widths": CHANGE_THRESHOLD,
        "segments_csv": args.segments_csv,
        "raw_dataset_dir": args.raw_dataset_dir,
        "splits": results,
        "total_samples": sum(result["samples"] for result in results.values()),
        "canonical_templates": True,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "qa_dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"[QA Generator] Summary: {summary_path}")
    if any(result["samples"] == 0 for result in results.values()):
        raise RuntimeError("At least one split has no valid QA; inspect rejection logs")


if __name__ == "__main__":
    main()
