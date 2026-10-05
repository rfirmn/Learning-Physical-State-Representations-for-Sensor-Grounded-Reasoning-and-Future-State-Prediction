"""Generate target-only QA from audited per-window physical reference artifacts."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from eksperimen_model.datasets.m4human_qa_dataset import STATUS, validate_qa
from eksperimen_model.utils.m4human_kinematics import TASK_DEFINITIONS, DEFAULT_RECIPE, build_evidence
from eksperimen_model.utils.m4human_runtime import atomic_json, canonical_hash, file_sha256, iter_jsonl


def windows_from_h_cache(root, joint_map_hash, recipe_hash):
    from eksperimen_model.m4human_training import WindowTensorCache
    source = WindowTensorCache(root, expected_lineage={"joint_map_hash": joint_map_hash, "recipe_hash": recipe_hash})
    if source.metadata.get("representation") != "full_H":
        raise ValueError("QA requires exact full-H source window cache")
    lineage = source.metadata["lineage"]
    for key in ("target_hash", "coordinate_hash", "split_hash"):
        if not lineage.get(key):
            raise ValueError(f"Missing audited H-cache lineage: {key}")
    root = Path(root).resolve()
    for index, item in enumerate(source):
        row, sensor = item["provenance"], item["sensor"]
        target = {}
        if row.get("target_path"):
            path = (root / row["target_path"]).resolve()
            if not path.is_relative_to(root) or file_sha256(path) != row.get("target_sha256"):
                raise ValueError(f"Changed/escaping target cache: {row['window_id']}")
            target = torch.load(path, map_location="cpu", weights_only=True)
        keys = ("P_relative_m", "r_m", "v_relative_mps", "v_root_mps")
        masks = ("position_valid_joint", "position_valid_root", "velocity_valid_joint", "velocity_valid_root")
        if target and (any(key not in target for key in (*keys, *masks)) or
                       any(not isinstance(target[key], torch.Tensor) for key in (*keys, *masks))):
            raise ValueError(f"Incomplete physical target payload: {row['window_id']}")
        yield {"window_id": row["window_id"], "subject_id": row["subject_id"], "recording_id": row["recording_id"],
            "split": row["split"], "action_id": row.get("action_id"), "source_keys": row["source_keys"],
            "frame_ids": row["frame_uids"], "time_s": sensor["time_s"].tolist(),
            "joint_map_hash": lineage["joint_map_hash"], "split_hash": lineage["split_hash"],
            "coordinate_hash": lineage["coordinate_hash"], "gt_source_hash": lineage["target_hash"],
            "h_cache_hash": canonical_hash(source.metadata), "window_support_hash": row["window_support_hash"],
            "annotation_audited": bool(target), "physical_values": {key: target[key].numpy() for key in keys} if target else {},
            "source_validity": {key: target[key].numpy() for key in masks} if target else {}}


def generate_records(windows, joint_map, recipe, selected_tasks):
    records, counts = [], Counter()
    for window in windows:
        if len(window["time_s"]) != 32:
            raise ValueError("QA requires exact window32 support")
        for task in selected_tasks:
            definition = TASK_DEFINITIONS[task]
            for body_part in definition.get("body_parts", ()) or (None,):
                annotation_ok = window.get("annotation_audited", False)
                evidence = build_evidence({key: np.asarray(value) for key, value in window["physical_values"].items()}, np.asarray(window["time_s"]), {key: np.asarray(value) for key, value in window["source_validity"].items()}, recipe=recipe, task=task, body_part=body_part, joint_map=joint_map, source="gt_reference") if annotation_ok else {"diagnostic_answer": "unknown", "evidence_status": "undefined", "reason": "annotation_not_audited", "support_frame_ids": []}
                answer = evidence["diagnostic_answer"]
                status = "undefined" if not annotation_ok or evidence.get("reason") == "insufficient_support" else ("unknown" if answer == "unknown" else "defined")
                relative = np.asarray(window["time_s"]) - window["time_s"][0]
                interval = [float(relative[6]), float(relative[-1])]
                question = definition["question"]
                if body_part:
                    question = question.replace("{body_part}", body_part)
                question += f" Interval {interval[0]:.6f}–{interval[1]:.6f} detik sejak awal observasi."
                row = {key: window[key] for key in ("window_id", "subject_id", "recording_id", "split", "source_keys", "frame_ids", "time_s", "joint_map_hash", "split_hash", "gt_source_hash", "coordinate_hash")}
                if "h_cache_hash" in window:
                    row.update(h_cache_hash=window["h_cache_hash"], window_support_hash=window["window_support_hash"])
                identity = {"window": window["window_id"], "task": task, "body_part": body_part, "recipe": canonical_hash(recipe)}
                row.update(qa_id=canonical_hash(identity), question_id=canonical_hash({"task": task, "body_part": body_part, "template": question}), task=task, body_part=body_part, question=question, answer=None if status == "undefined" else answer, target_status=status, validity=STATUS[status], evidence_status=evidence["evidence_status"], reason=evidence.get("reason"), observation_interval_s=[0.0, float(relative[-1])], task_interval_s=interval, label_support=evidence.get("support_frame_ids", []), label_support_kind="relative_frame_index_0_based", evidence_support=evidence.get("support_frame_ids", []), label_recipe_version=recipe.get("label_recipe_version", "kinematic_recipe_v3"), derivative_recipe_id=recipe.get("derivative_recipe_id", "aligned12hz_endpoint_sg_v2"), recipe_hash=canonical_hash(recipe), action_id=window.get("action_id"))
                validate_qa(row)
                records.append(row)
                counts[(task, status, row["answer"])] += 1
    return records, {"counts": [{"task": key[0], "status": key[1], "answer": key[2], "count": value} for key, value in counts.items()], "n_windows": len({row["window_id"] for row in records}), "n_subjects": len({row["subject_id"] for row in records}), "n_recordings": len({row["recording_id"] for row in records})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--references", help="Audited target-only JSONL physical reference windows")
    source.add_argument("--h-cache", help="Exact full-H window cache with offline target payloads")
    parser.add_argument("--joint-map", required=True)
    parser.add_argument("--recipe")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", nargs="+", default=["root_speed_trend", "relative_limb_motion"])
    args = parser.parse_args()
    recipe = json.loads(Path(args.recipe).read_text()) if args.recipe else DEFAULT_RECIPE
    joint_map = json.loads(Path(args.joint_map).read_text())
    if not joint_map.get("audited"):
        raise ValueError("QA needs audited joint map")
    if any(task not in {"root_speed_trend", "relative_limb_motion"} for task in args.tasks):
        raise ValueError("Optional tasks require independent reliability gate; P0 generator permits core tasks")
    joint_hash, recipe_hash = file_sha256(args.joint_map), canonical_hash(recipe)
    windows = windows_from_h_cache(args.h_cache, joint_hash, recipe_hash) if args.h_cache else iter_jsonl(args.references)
    records, coverage = generate_records(windows, joint_map, recipe, args.tasks)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in records), encoding="utf-8")
    temporary.replace(output)
    from eksperimen_model.datasets.m4human_qa_dataset import read_qa
    read_qa(output)
    atomic_json(output.with_suffix(".coverage.json"), {**coverage, "reference_sha256": file_sha256(Path(args.h_cache) / "metadata.json") if args.h_cache else file_sha256(args.references), "qa_sha256": file_sha256(output), "recipe_hash": recipe_hash, "joint_map_hash": joint_hash})


if __name__ == "__main__":
    main()
