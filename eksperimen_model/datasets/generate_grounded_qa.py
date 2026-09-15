"""
generate_grounded_qa.py
Deterministic Grounded-QA Dataset Generator for Stage 4 (Cognitive Alignment & SLM Inference).

Extracts physically verified, deterministic QA pairs from pre-extracted MM-Fi latent features
and 3D ground-truth skeletons across the 5 formal inference tasks:
  Task 1: Physical State Understanding (Posture & Active Moving Limbs via P75 threshold)
  Task 2: Spatial & Metric State Understanding (Depth in meters & Lateral position)
  Task 3: Temporal Kinematics Inference (Radial Velocity & Direction)
  Task 4: Inter-Limb Spatial Configuration (Wrist Euclidean distance & Arm posture)
  Task 5: Future-State Predictive Physical Reasoning (Anticipated motion at t+8 / 0.8s)
  + Multi-Attribute Compositional Queries.

Applies disjoint linguistic prompt templates between Train and Test sets to evaluate
true compositional generalization and prevent template memorization.
"""

import os
import sys
import glob
import json
import math
import random
import argparse
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Any, Tuple

# Fix UTF-8 encoding on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# MM-Fi Joint Indices
# 0: Pelvis, 1: R_Hip, 2: R_Knee, 3: R_Ankle, 4: L_Hip, 5: L_Knee, 6: L_Ankle,
# 7: Spine, 8: Thorax, 9: Neck/Nose, 10: Head,
# 11: L_Shoulder, 12: L_Elbow, 13: L_Wrist, 14: R_Shoulder, 15: R_Elbow, 16: R_Wrist
PELVIS_IDX = 0
R_WRIST_IDX = 16
L_WRIST_IDX = 13
R_ANKLE_IDX = 3
L_ANKLE_IDX = 6
R_KNEE_IDX = 2
L_KNEE_IDX = 5
UPPER_LIMBS = [11, 12, 13, 14, 15, 16]
LOWER_LIMBS = [1, 2, 3, 4, 5, 6]

ACTION_NAMES = {
    0: "Chest expansion", 1: "Side twist", 2: "Arm extensions",
    3: "Left extension", 4: "Right extension", 5: "High arm punch",
    6: "Low arm punch", 7: "Horizontal kick", 8: "Squat",
    9: "Front lunge", 10: "Left lunge", 11: "Right lunge",
    12: "Diagonal punch", 13: "Two-handed punch",
    14: "Shoulder abduct", 15: "Shoulder flex", 16: "Elbow flex",
    17: "Wrist flex", 18: "Hip flex", 19: "Knee flex",
    20: "Ankle flex", 21: "Side leg raise", 22: "Back leg raise",
    23: "Torso twist", 24: "Arm circle", 25: "Leg circle",
    26: "Balance stance"
}

# Linguistic Prompt Templates (Strictly Disjoint between Train and Test)
TRAIN_TEMPLATES = {
    "task1_state": [
        "Based on the physical sensor observations, what is the subject's current posture and which limbs are actively moving?",
        "From the observed physical state, determine the body posture and identify any actively moving extremities.",
        "Observe the current physical posture and state. What posture is the subject holding, and are the upper or lower limbs active?",
    ],
    "task2_spatial": [
        "Based on the sensor observations, what is the estimated depth distance of the subject from the sensor rig and their lateral position?",
        "Estimate the metric depth range of the subject relative to the sensor platform and identify whether they are on the left, center, or right.",
        "From the physical state, determine the subject's depth distance in meters and lateral horizontal alignment.",
    ],
    "task3_kinematics": [
        "Analyze the subject's kinematics: are they approaching, receding, or stationary relative to the sensor, and what is their radial velocity?",
        "What is the subject's radial motion direction relative to the transceiver platform, and what is the velocity in meters per second?",
        "From the temporal dynamics, determine whether the subject is moving toward or away from the sensor and quantify their radial velocity.",
    ],
    "task4_configuration": [
        "Analyze the inter-limb configuration: what is the Euclidean distance between the subject's wrists and their arm configuration?",
        "What is the estimated metric distance between the left and right wrists, and are the arms extended or contracted?",
        "Evaluate the upper limb spatial configuration. Estimate the wrist-to-wrist separation in meters.",
    ],
    "task5_future": [
        "Based on the historical momentum and current dynamics, predict the subject's expected radial motion direction and depth at horizon t+8 (0.8 seconds into the future).",
        "Forecast the future physical state 0.8 seconds ahead: what will be the subject's radial direction and estimated depth from the sensor?",
        "Project the subject's kinematics forward by 0.8 seconds. What is the anticipated radial trajectory and future depth?",
    ],
    "compositional": [
        "Synthesize the physical observations: is the subject approaching the sensor while maintaining an active upper-limb movement?",
        "Analyze the coordinated motion: is the subject's lateral position centered while their distance from the radar changes?",
        "Evaluate the spatio-temporal state: compare the subject's body posture with their radial direction of motion.",
    ]
}

# Unseen Test Templates (Completely distinct wording & structural framing)
TEST_TEMPLATES = {
    "task1_state": [
        "Inspect the grounded sensor representation to diagnose the subject's bodily stance and classify active limb engagement.",
        "Diagnose whether the individual is standing, squatting, or lunging, and identify active limb involvement from the sensor tokens.",
    ],
    "task2_spatial": [
        "Quantify the exact range in meters from the radar origin to the subject and specify their lateral quadrant.",
        "Calculate the metric depth separation between the subject and the sensor rig, and determine the lateral offset position.",
    ],
    "task3_kinematics": [
        "Assess the velocity vector of the body center: is radial translation oriented toward or away from the baseline, and at what speed?",
        "Examine the rate of displacement along the depth axis. Indicate the direction of translation and velocity in m/s.",
    ],
    "task4_configuration": [
        "Measure the spatial span between the two hands in meters and categorize the bilateral arm arrangement.",
        "Calculate the 3D metric separation separating the left and right wrist keypoints.",
    ],
    "task5_future": [
        "Extrapolate the physical trajectory to time step t+8 (0.8s later). What will be the final depth and radial direction?",
        "What state transition will occur over the upcoming 800ms window regarding depth position and movement orientation?",
    ],
    "compositional": [
        "Perform a multi-attribute query: assess if upper-body motion dominates while the subject translates along the radial axis.",
        "Cross-reference depth trajectory with posture: is the person actively moving toward the detector while in a non-standing posture?",
    ]
}


def calibrate_active_velocity_threshold(train_dir: str, delta_t: float = 0.1) -> float:
    """
    Computes P75 threshold of joint velocities strictly across the training set (zero data leakage).
    """
    print(f"[QA Generator] Calibrating joint activity threshold P75 strictly from train set ({train_dir})...")
    pt_files = sorted(glob.glob(os.path.join(train_dir, "*.pt")))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {train_dir}")

    all_velocities = []
    # Sample up to 100 files for fast, robust empirical percentile estimation
    sample_files = pt_files if len(pt_files) <= 100 else random.sample(pt_files, 100)

    for fpath in sample_files:
        data = torch.load(fpath, weights_only=True)
        gt = data["gt_skeleton"].numpy() # (T, 17, 3)
        if len(gt) > 1:
            diffs = np.linalg.norm(gt[1:] - gt[:-1], axis=-1) / delta_t # (T-1, 17)
            all_velocities.append(diffs.flatten())

    concatenated = np.concatenate(all_velocities)
    tau_p75 = float(np.percentile(concatenated, 75))
    print(f"[QA Generator] Calibrated Tau_active (P75): {tau_p75:.4f} m/s (over {len(concatenated)} joint instances)")
    return tau_p75


def extract_physical_parameters(
    gt_window: np.ndarray,
    tau_active: float,
    delta_t: float = 0.1
) -> Dict[str, Any]:
    """
    Computes exact, deterministic physical parameters from a 24-frame window (16 history + 8 future):
    Frames 0..15: History (Frame 15 is current frame t)
    Frames 16..23: Future (Frame 23 is horizon t+8)
    """
    t_curr_idx = 15
    t_fut_idx = 23

    p_curr = gt_window[t_curr_idx] # (17, 3)
    p_prev = gt_window[t_curr_idx - 1] # (17, 3)
    p_fut = gt_window[t_fut_idx] # (17, 3)

    # 1. Spatial Coordinates (Pelvis)
    pelvis_curr = p_curr[PELVIS_IDX]
    depth_m = float(pelvis_curr[2])
    lateral_x = float(pelvis_curr[0])
    elevation_y = float(pelvis_curr[1])

    if lateral_x < -0.15:
        lateral_pos = "left"
    elif lateral_x > 0.15:
        lateral_pos = "right"
    else:
        lateral_pos = "center"

    # 2. Kinematics (Radial velocity along Z axis)
    v_z = float((pelvis_curr[2] - p_prev[PELVIS_IDX, 2]) / delta_t)
    if v_z < -0.10:
        radial_dir = "approaching"
    elif v_z > 0.10:
        radial_dir = "receding"
    else:
        radial_dir = "stationary"

    # 3. Posture
    # Pelvis height and knee flexion:
    # In MM-Fi sensor rig frame, Y ~ -0.15 to -0.4 is standing/elevated, < -0.5 is low squat/lunge
    r_knee_angle = np.linalg.norm(p_curr[R_KNEE_IDX] - p_curr[PELVIS_IDX])
    if elevation_y < -0.45:
        posture = "squatting"
    elif elevation_y < -0.30:
        posture = "lunging"
    else:
        posture = "standing"

    # 4. Active Limbs (via P75 threshold)
    joint_vels = np.linalg.norm(p_curr - p_prev, axis=-1) / delta_t # (17,)
    upper_active = bool(np.any(joint_vels[UPPER_LIMBS] > tau_active))
    lower_active = bool(np.any(joint_vels[LOWER_LIMBS] > tau_active))

    if upper_active and lower_active:
        active_limbs = "both upper and lower limbs"
    elif upper_active:
        active_limbs = "upper limbs"
    elif lower_active:
        active_limbs = "lower limbs"
    else:
        active_limbs = "none"

    # 5. Inter-limb Spatial Configuration
    wrist_dist = float(np.linalg.norm(p_curr[L_WRIST_IDX] - p_curr[R_WRIST_IDX]))
    if wrist_dist > 0.65:
        arm_config = "extended"
    elif wrist_dist < 0.25:
        arm_config = "contracted"
    else:
        arm_config = "moderate"

    # 6. Future Prediction at t+8
    pelvis_fut = p_fut[PELVIS_IDX]
    fut_depth_m = float(pelvis_fut[2])
    fut_delta_z = float(pelvis_fut[2] - pelvis_curr[2])
    if fut_delta_z < -0.05:
        fut_dir = "approaching"
    elif fut_delta_z > 0.05:
        fut_dir = "receding"
    else:
        fut_dir = "stationary"

    return {
        "depth_m": round(depth_m, 2),
        "lateral_position": lateral_pos,
        "radial_direction": radial_dir,
        "radial_velocity_mps": round(v_z, 2),
        "posture": posture,
        "active_limbs": active_limbs,
        "wrist_distance_m": round(wrist_dist, 2),
        "arm_configuration": arm_config,
        "future_radial_direction": fut_dir,
        "future_depth_m": round(fut_depth_m, 2)
    }


def format_qa_sample(
    task_name: str,
    params: Dict[str, Any],
    action_name: str,
    templates: Dict[str, List[str]],
    file_rel: str,
    start_idx: int,
    sample_id: str
) -> Dict[str, Any]:
    """Formats an individual QA pair with structured key-values and concise explanatory text."""
    question = random.choice(templates[task_name])

    if task_name == "task1_state":
        structured = {
            "posture": params["posture"],
            "active_limbs": params["active_limbs"],
            "action_type": action_name
        }
        answer_text = (f"posture: {params['posture']}, active_limbs: {params['active_limbs']}. "
                       f"The subject exhibits a {params['posture']} posture with {params['active_limbs']} actively engaged during {action_name}.")

    elif task_name == "task2_spatial":
        structured = {
            "depth_m": params["depth_m"],
            "lateral_position": params["lateral_position"]
        }
        answer_text = (f"depth_m: {params['depth_m']}, lateral_position: {params['lateral_position']}. "
                       f"The subject is positioned at a depth distance of {params['depth_m']} meters, aligned towards the {params['lateral_position']} of the sensor midline.")

    elif task_name == "task3_kinematics":
        structured = {
            "radial_direction": params["radial_direction"],
            "radial_velocity_mps": params["radial_velocity_mps"]
        }
        answer_text = (f"radial_direction: {params['radial_direction']}, radial_velocity_mps: {params['radial_velocity_mps']}. "
                       f"The subject is currently {params['radial_direction']} relative to the sensor platform with an instantaneous radial speed of {abs(params['radial_velocity_mps'])} m/s.")

    elif task_name == "task4_configuration":
        structured = {
            "wrist_distance_m": params["wrist_distance_m"],
            "arm_configuration": params["arm_configuration"]
        }
        answer_text = (f"wrist_distance_m: {params['wrist_distance_m']}, arm_configuration: {params['arm_configuration']}. "
                       f"The Euclidean separation between wrists is {params['wrist_distance_m']} meters, corresponding to a {params['arm_configuration']} arm arrangement.")

    elif task_name == "task5_future":
        structured = {
            "future_radial_direction": params["future_radial_direction"],
            "future_depth_m": params["future_depth_m"]
        }
        answer_text = (f"future_radial_direction: {params['future_radial_direction']}, future_depth_m: {params['future_depth_m']}. "
                       f"At horizon t+8 (0.8s ahead), the subject is projected to be {params['future_radial_direction']} at an anticipated depth of {params['future_depth_m']} meters.")

    elif task_name == "compositional":
        is_approaching = (params["radial_direction"] == "approaching")
        upper_active = ("upper" in params["active_limbs"])
        structured = {
            "is_approaching": is_approaching,
            "upper_limbs_active": upper_active,
            "posture": params["posture"],
            "lateral_position": params["lateral_position"]
        }
        answer_text = (f"is_approaching: {is_approaching}, upper_limbs_active: {upper_active}, posture: {params['posture']}. "
                       f"The subject is {'moving towards' if is_approaching else 'not moving towards'} the sensor while {params['posture']} on the {params['lateral_position']} side.")

    return {
        "sample_id": sample_id,
        "feature_file": file_rel,
        "window_start": start_idx,
        "t_curr_frame": start_idx + 15,
        "task_name": task_name,
        "question": question,
        "target_structured": structured,
        "target_text": answer_text
    }


def generate_split_qa(
    split: str,
    features_dir: str,
    output_file: str,
    tau_active: float,
    stride: int = 4,
    window_size: int = 24
) -> int:
    """Generates and writes JSONL QA dataset for a given split."""
    split_dir = os.path.join(features_dir, split)
    pt_files = sorted(glob.glob(os.path.join(split_dir, "*.pt")))
    templates = TRAIN_TEMPLATES if split == "train" else TEST_TEMPLATES

    print(f"\n[QA Generator] Processing split '{split.upper()}' ({len(pt_files)} files, stride={stride})...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    qa_records = []
    sample_counter = 0

    task_pool = [
        "task1_state", "task2_spatial", "task3_kinematics",
        "task4_configuration", "task5_future", "compositional"
    ]

    for fpath in tqdm(pt_files, desc=f"Generating {split} QA"):
        data = torch.load(fpath, weights_only=True)
        gt_skel = data["gt_skeleton"].numpy() # (T, 17, 3)
        action_idx = data.get("action_idx", 0)
        action_name = ACTION_NAMES.get(action_idx, f"Action {action_idx}")
        file_rel = os.path.basename(fpath)

        T = len(gt_skel)
        if T < window_size:
            continue

        for start_idx in range(0, T - window_size + 1, stride):
            gt_window = gt_skel[start_idx : start_idx + window_size]
            params = extract_physical_parameters(gt_window, tau_active=tau_active)

            # Assign 2 distinct tasks per sliding window sample for balanced coverage
            sampled_tasks = random.sample(task_pool, 2)
            for t_name in sampled_tasks:
                sample_counter += 1
                sample_id = f"{split}_{sample_counter:06d}"
                record = format_qa_sample(
                    task_name=t_name,
                    params=params,
                    action_name=action_name,
                    templates=templates,
                    file_rel=file_rel,
                    start_idx=start_idx,
                    sample_id=sample_id
                )
                qa_records.append(record)

    with open(output_file, "w", encoding="utf-8") as f:
        for rec in qa_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[QA Generator] Finished '{split.upper()}': Saved {len(qa_records)} QA samples to {output_file}")
    return len(qa_records)


def main():
    parser = argparse.ArgumentParser(description="Deterministic Grounded-QA Dataset Generator for Stage 4")
    parser.add_argument("--features_dir", type=str, default="datasets/MM-Fi_features_v2",
                        help="Path to pre-extracted MM-Fi latent features")
    parser.add_argument("--output_dir", type=str, default="datasets/MM-Fi_grounded_qa",
                        help="Output directory to save generated JSONL files")
    parser.add_argument("--train_stride", type=int, default=4,
                        help="Sliding window stride for training set")
    parser.add_argument("--val_stride", type=int, default=8,
                        help="Sliding window stride for validation set")
    parser.add_argument("--test_stride", type=int, default=8,
                        help="Sliding window stride for held-out test set")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # 1. Calibrate empirical P75 velocity threshold strictly from Train set
    train_features_dir = os.path.join(args.features_dir, "train")
    tau_active = calibrate_active_velocity_threshold(train_features_dir)

    # 2. Generate Train, Val, and Test QA datasets
    train_out = os.path.join(args.output_dir, "mmfi_grounded_qa_train.jsonl")
    val_out = os.path.join(args.output_dir, "mmfi_grounded_qa_val.jsonl")
    test_out = os.path.join(args.output_dir, "mmfi_grounded_qa_test.jsonl")

    n_train = generate_split_qa("train", args.features_dir, train_out, tau_active, stride=args.train_stride)
    n_val = generate_split_qa("val", args.features_dir, val_out, tau_active, stride=args.val_stride)
    n_test = generate_split_qa("test", args.features_dir, test_out, tau_active, stride=args.test_stride)

    # Save summary metadata
    summary_path = os.path.join(args.output_dir, "qa_dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "tau_active_p75_mps": tau_active,
            "train_samples": n_train,
            "val_samples": n_val,
            "test_samples": n_test,
            "total_samples": n_train + n_val + n_test,
            "train_stride": args.train_stride,
            "val_stride": args.val_stride,
            "test_stride": args.test_stride,
            "unseen_test_templates": True
        }, f, indent=2)

    print(f"\n=======================================================")
    print(f" QA Dataset Generation Completed Successfully! ")
    print(f" Summary: Train={n_train}, Val={n_val}, Test={n_test}, Total={n_train+n_val+n_test}")
    print(f" Saved metadata to: {summary_path}")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
