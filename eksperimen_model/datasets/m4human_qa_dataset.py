"""Strict QA targets joined lazily to unique exact-U sensor caches."""
import json
import math
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset

from eksperimen_model.utils.m4human_kinematics import TASK_DEFINITIONS
from eksperimen_model.utils.m4human_runtime import CONTRACT_VERSION, file_sha256, require_lineage

STATUS = {"defined": "valid", "unknown": "unknown", "undefined": "excluded"}
REQUIRED = ("qa_id", "window_id", "subject_id", "recording_id", "split", "task", "question", "target_status", "validity", "answer", "recipe_hash", "joint_map_hash", "split_hash")


def cache_metadata_path(path):
    path = Path(path)
    return path / "metadata.json" if path.is_dir() else path


def canonical_answer(task, answer):
    return json.dumps({"task": task, "answer": answer}, ensure_ascii=False, separators=(",", ":"))


def validate_qa(record):
    missing = [key for key in REQUIRED if key not in record]
    if missing:
        raise ValueError(f"QA missing fields: {missing}")
    if record["task"] not in TASK_DEFINITIONS or record["split"] not in {"train", "val", "test"}:
        raise ValueError("Unknown task/split")
    if not all(isinstance(record[key], str) and record[key] for key in ("qa_id", "window_id", "recording_id", "question")) or not (type(record["subject_id"]) is int or isinstance(record["subject_id"], str) and record["subject_id"]):
        raise ValueError("QA identity and question fields have invalid type or are empty")
    status = record["target_status"]
    if STATUS.get(status) != record["validity"]:
        raise ValueError("Conflicting target status/validity")
    answer = record["answer"]
    if status == "undefined":
        if answer is not None or not record.get("reason"):
            raise ValueError("Excluded QA requires null answer and reason")
    elif answer not in TASK_DEFINITIONS[record["task"]]["answers"] or (answer == "unknown") != (status == "unknown"):
        raise ValueError("Answer/status violates task registry")
    if not all(isinstance(record[key], str) and record[key] for key in ("recipe_hash", "joint_map_hash", "split_hash")):
        raise ValueError("QA lineage must be resolved")


def read_qa(path):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate QA JSON field: {key}")
            result[key] = value
        return result
    records = [json.loads(line, object_pairs_hook=unique_keys,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite QA JSON: {value}")))
               for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids, subject_splits, recording_splits, window_splits = set(), {}, {}, {}
    for record in records:
        validate_qa(record)
        if record["qa_id"] in ids:
            raise ValueError("Duplicate QA ID")
        ids.add(record["qa_id"])
        for key, mapping in (("subject_id", subject_splits), ("recording_id", recording_splits), ("window_id", window_splits)):
            value, split = record[key], record["split"]
            if value in mapping and mapping[value] != split:
                raise ValueError(f"Split leakage: {key}={value}")
            mapping[value] = split
    return records


class M4HumanQADataset(Dataset):
    def __init__(self, qa_path, cache_manifest, split, condition, expected_lineage=None, test_contract=None):
        if condition not in {"C_base", "C_kin"}:
            raise ValueError("Only matched exact-U conditions accepted")
        self.records = [row for row in read_qa(qa_path) if row["split"] == split and row["target_status"] != "undefined"]
        if split == "test" and not test_contract:
            raise ValueError("Test access requires locked test_contract")
        metadata_path = cache_metadata_path(cache_manifest).resolve()
        self.cache_root = metadata_path.parent
        manifest = json.loads(metadata_path.read_text())
        if "windows" in manifest:
            self.metadata = manifest["metadata"]
            windows = manifest["windows"]
        else:
            self.metadata = manifest
            index_path = self.cache_root / "index.jsonl"
            if self.metadata.get("index_sha256") != file_sha256(index_path):
                raise ValueError("U index integrity failure")
            windows = [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
        if self.metadata.get("contract_version") != CONTRACT_VERSION or self.metadata.get("complete") is not True:
            raise ValueError("Foreign or incomplete U cache")
        self.expected_lineage = expected_lineage or {}
        require_lineage(self.metadata, self.expected_lineage)
        self.metadata.update({key: value for key, value in self.metadata.get("lineage", {}).items() if key not in self.metadata})
        if self.metadata.get("condition") != condition or self.metadata.get("representation") != "exact_U":
            raise ValueError("U cache must identify condition and exact_U")
        if split == "test":
            from eksperimen_model.m4human_evaluation import verify_test_contract
            verify_test_contract(test_contract, qa_path=qa_path, cache_manifest=cache_manifest, condition=condition)
        self.windows = {row["window_id"]: row for row in windows}
        if len(self.windows) != len(windows):
            raise ValueError("Duplicate U cache window")
        for row in self.records:
            cached = self.windows.get(row["window_id"])
            if cached is None or cached.get("split") != split:
                raise ValueError("QA window absent or split mismatch in U cache")
            for key in ("subject_id", "recording_id"):
                if cached.get(key) != row[key]:
                    raise ValueError(f"QA/U identity mismatch: {key}")
            for qa_key, lineage_key in (("recipe_hash", "recipe_hash"), ("joint_map_hash", "joint_map_hash"),
                                        ("split_hash", "split_hash"), ("coordinate_hash", "coordinate_hash"),
                                        ("gt_source_hash", "target_hash"), ("h_cache_hash", "h_cache_hash")):
                if not row.get(qa_key) or row[qa_key] != self.metadata.get(lineage_key):
                    raise ValueError(f"QA/U lineage mismatch: {qa_key}")
            for qa_key, cache_key in (("window_support_hash", "window_support_hash"), ("source_keys", "source_keys"), ("frame_ids", "frame_uids")):
                if qa_key not in row or cached.get(cache_key) != row[qa_key]:
                    raise ValueError(f"QA/U exact-window mismatch: {qa_key}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        cached = self.windows[row["window_id"]]
        path = (self.cache_root / cached["tensor_path"]).resolve()
        if not path.is_relative_to(self.cache_root):
            raise ValueError("U cache tensor path escapes manifest root")
        from eksperimen_model.utils.m4human_runtime import file_sha256
        if cached.get("tensor_sha256") != file_sha256(path):
            raise ValueError("U cache tensor checksum mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if "time_s" not in payload or len(payload["time_s"]) != 32 or len(row["time_s"]) != 32 or any(not math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-9) for a, b in zip(payload["time_s"], row["time_s"])):
            raise ValueError("QA/U exact-window timestamp mismatch")
        U = payload["U"].detach().clone()
        mask = payload.get("token_mask", payload.get("token_valid"))
        if mask is None:
            raise ValueError("Missing sensor-derived token validity")
        mask = mask.bool()
        K = self.metadata["K"]
        if U.shape != (K, 256) or mask.shape != (K,) or not torch.isfinite(U[mask]).all():
            raise ValueError("Invalid exact U cache")
        sample = {key: row[key] for key in ("qa_id", "window_id", "question", "task")}
        sample.update(U=U, token_mask=mask, answers=TASK_DEFINITIONS[row["task"]]["answers"], canonical_answer=canonical_answer(row["task"], row["answer"]))
        return sample
