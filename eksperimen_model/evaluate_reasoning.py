"""Matched Stage 4 reasoning benchmark with frozen Dynamics forecasts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eksperimen_model.datasets.grounded_qa_dataset import GroundedQADataset
from eksperimen_model.models.projector import PhysicalSLMWrapper

TASK_LABELS = {
    "current_wrist_separation": ("narrower", "wider"),
    "future_wrist_separation_change": ("closing", "stable", "opening"),
}


def unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_prediction(raw: str, task: str) -> tuple[str | None, str]:
    """Accept exactly the one JSON field and vocabulary specified by the task."""
    if task not in TASK_LABELS:
        raise ValueError(f"Unsupported task: {task}")
    try:
        value = json.loads(raw.strip(), object_pairs_hook=unique_json_object)
    except (ValueError, TypeError):
        return None, "invalid_json"
    if not isinstance(value, dict) or set(value) != {task}:
        return None, "wrong_schema"
    label = value[task]
    if not isinstance(label, str) or label not in TASK_LABELS[task]:
        return None, "invalid_value"
    return label, "parsed"


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"QA manifest missing: {path}")
    records, seen = [], set()
    required = {
        "sample_id", "window_id", "feature_file", "window_start", "task_name",
        "question", "target_structured", "target_text", "subject", "action", "recording_id",
        "crosses_segment_boundary",
    }
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            missing = required - set(rec)
            task = rec.get("task_name")
            target = rec.get("target_structured")
            sample_id = rec.get("sample_id")
            if missing or task not in TASK_LABELS or not isinstance(target, dict) or set(target) != {task}:
                raise ValueError(f"Invalid QA schema at {path}:{line_number}; missing={sorted(missing)}")
            if target[task] not in TASK_LABELS[task]:
                raise ValueError(f"Invalid label at {path}:{line_number}: {target[task]!r}")
            if json.loads(rec["target_text"], object_pairs_hook=unique_json_object) != target:
                raise ValueError(f"Answer text differs from structured target at {path}:{line_number}")
            if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
                raise ValueError(f"Duplicate or invalid sample ID at {path}:{line_number}")
            if not isinstance(rec["crosses_segment_boundary"], bool):
                raise ValueError(f"Invalid boundary stratum at {path}:{line_number}")
            seen.add(sample_id)
            records.append(rec)
    if not records:
        raise ValueError(f"QA manifest is empty: {path}")
    return records


def load_checkpoint(path: Path, condition: str, base_model: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{condition} trained projector checkpoint missing: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    provenance = checkpoint.get("provenance") if isinstance(checkpoint, dict) else None
    expected_tokens = 16 if condition == "B3" else 24
    if (
        not isinstance(checkpoint, dict)
        or "projector_state_dict" not in checkpoint
        or checkpoint.get("condition") != condition
        or not isinstance(provenance, dict)
        or provenance.get("base_model") != base_model
        or provenance.get("token_count") != expected_tokens
    ):
        raise ValueError(f"{condition} checkpoint has incompatible weights/provenance: {path}")
    return checkpoint


def row(rec: dict[str, Any], condition: str, raw: str, donor: dict[str, Any] | None = None) -> dict[str, Any]:
    task = rec["task_name"]
    prediction, status = parse_prediction(raw, task)
    result = {
        "sample_id": rec["sample_id"], "window_id": rec["window_id"],
        "recording_id": rec["recording_id"], "subject": rec["subject"],
        "action": rec["action"], "environment": rec.get("environment"),
        "crosses_segment_boundary": rec["crosses_segment_boundary"],
        "boundary_frame_ids": rec.get("boundary_frame_ids"),
        "segment_ids": rec.get("segment_ids"),
        "current_segment_id": rec.get("current_segment_id"),
        "future_segment_id": rec.get("future_segment_id"),
        "source_frame_ids": rec.get("source_frame_ids"),
        "task_name": task, "condition": condition,
        "output_type": "rule_baseline" if condition == "B0" else "language_model",
        "target": rec["target_structured"][task], "prediction": prediction,
        "parse_status": status, "correct": prediction == rec["target_structured"][task],
        "raw_response": raw,
    }
    if donor is not None:
        result.update({
            "donor_sample_id": donor["sample_id"], "donor_subject": donor["subject"],
            "donor_action": donor["action"], "donor_target": donor["target_structured"][task],
            "matches_donor_target": prediction == donor["target_structured"][task],
        })
    return result


def score(rows: list[dict[str, Any]], task: str) -> dict[str, Any]:
    subset = [r for r in rows if r["task_name"] == task]
    if not subset:
        return {"n": 0, "parsed": 0, "parse_coverage": None, "accuracy": None, "macro_f1": None}
    targets = [r["target"] for r in subset]
    predictions = [r["prediction"] if r["prediction"] is not None else "__unparsed__" for r in subset]
    n = len(subset)
    parsed = sum(r["parse_status"] == "parsed" for r in subset)
    return {
        "n": n, "parsed": parsed, "parse_coverage": parsed / n,
        "accuracy": sum(r["correct"] for r in subset) / n,
        "macro_f1": float(f1_score(targets, predictions, labels=TASK_LABELS[task], average="macro", zero_division=0)),
        "target_counts": dict(Counter(targets)),
        "parse_status_counts": dict(Counter(r["parse_status"] for r in subset)),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"tasks": {task: score(rows, task) for task in TASK_LABELS}}
    for field, output_field in (("subject", "by_subject"), ("crosses_segment_boundary", "by_segment_boundary")):
        groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for entry in rows:
            groups[entry[field]].append(entry)
        result[output_field] = {
            str(key).lower() if isinstance(key, bool) else key: {task: score(group, task) for task in TASK_LABELS}
            for key, group in groups.items()
        }
    return result


def paired_delta(a_rows: list[dict[str, Any]], b_rows: list[dict[str, Any]], task: str, seed: int) -> dict[str, Any]:
    a = {r["sample_id"]: r for r in a_rows if r["task_name"] == task}
    b = {r["sample_id"]: r for r in b_rows if r["task_name"] == task}
    if set(a) != set(b):
        raise ValueError(f"Unequal paired sample IDs for {task}")
    if not a:
        return {"n": 0, "delta_accuracy": None, "delta_macro_f1": None, "cluster_ci95": None}
    clusters: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for sample_id in sorted(a):
        if a[sample_id]["target"] != b[sample_id]["target"]:
            raise ValueError(f"Paired target mismatch: {sample_id}")
        clusters[a[sample_id]["recording_id"]].append((a[sample_id], b[sample_id]))
    keys = sorted(clusters)
    rng = random.Random(seed)
    acc, f1 = [], []
    for _ in range(1000):
        sample = [pair for key in (rng.choice(keys) for _ in keys) for pair in clusters[key]]
        left, right = [p[0] for p in sample], [p[1] for p in sample]
        acc.append(score(left, task)["accuracy"] - score(right, task)["accuracy"])
        f1.append(score(left, task)["macro_f1"] - score(right, task)["macro_f1"])
    return {
        "n": len(a), "n_recording_clusters": len(keys),
        "delta_accuracy": score(a_rows, task)["accuracy"] - score(b_rows, task)["accuracy"],
        "delta_macro_f1": score(a_rows, task)["macro_f1"] - score(b_rows, task)["macro_f1"],
        "cluster_ci95": {
            "delta_accuracy": [float(x) for x in np.percentile(acc, [2.5, 97.5])],
            "delta_macro_f1": [float(x) for x in np.percentile(f1, [2.5, 97.5])],
        },
    }


def select_donors(records: list[dict[str, Any]], kind: str, seed: int) -> dict[str, dict[str, Any]]:
    rng = random.Random(seed)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_task[rec["task_name"]].append(rec)
    selected = {}
    for rec in records:
        task = rec["task_name"]
        candidates = [
            d for d in by_task[task]
            if d["recording_id"] != rec["recording_id"]
            and d["target_structured"][task] != rec["target_structured"][task]
            and (
                d["action"] != rec["action"] if kind == "cross_action"
                else d["action"] == rec["action"] and d["subject"] != rec["subject"]
            )
        ]
        if candidates:
            selected[rec["sample_id"]] = rng.choice(sorted(candidates, key=lambda d: d["sample_id"]))
    return selected


def generate(wrapper: PhysicalSLMWrapper, dataset: GroundedQADataset, rec: dict[str, Any],
             device: torch.device, condition: str, donor: dict[str, Any] | None = None) -> str:
    prefix, suffix, _ = dataset.encode_prompt(rec["question"])
    prefix, suffix = prefix.unsqueeze(0).to(device), suffix.unsqueeze(0).to(device)
    if condition == "B1":
        ids = torch.cat((prefix, suffix), dim=1)
        with torch.inference_mode():
            output = wrapper.llm.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=32,
                do_sample=False, pad_token_id=wrapper.tokenizer.pad_token_id,
                eos_token_id=wrapper.tokenizer.eos_token_id,
            )
        return wrapper.tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True).strip()
    tokens = dataset.get_physical_tokens(donor or rec).unsqueeze(0).to(device)
    with torch.inference_mode():
        response = wrapper.generate_response(
            prefix_input_ids=prefix, prefix_attention_mask=torch.ones_like(prefix),
            physical_tokens=tokens, suffix_input_ids=suffix,
            suffix_attention_mask=torch.ones_like(suffix),
            max_new_tokens=32, temperature=0.0,
        )
    return response[0].strip()


def evaluate_probe(path: Path, dataset: GroundedQADataset, records: list[dict[str, Any]],
                   device: torch.device, train_digest: str, val_digest: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score an optional, separately trained B2 probe on the same QA panel."""
    from eksperimen_model.train_probe import CLASS_NAMES, MultiTaskPhysicalProbe

    if not path.is_file():
        raise FileNotFoundError(f"B2 probe checkpoint missing: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid B2 probe checkpoint: {path}")
    if checkpoint.get("class_names") != CLASS_NAMES or tuple(checkpoint.get("task_names", ())) != tuple(TASK_LABELS):
        raise ValueError(f"B2 probe class order differs from QA tasks: {path}")
    if checkpoint.get("model_config") != {"in_dim": 384, "history_frames": 16}:
        raise ValueError(f"B2 probe input contract differs from evaluation: {path}")
    provenance = checkpoint.get("provenance", {})
    if provenance.get("train", {}).get("qa_sha256") != train_digest or provenance.get("val", {}).get("qa_sha256") != val_digest:
        raise ValueError(f"B2 probe trained with different QA manifests: {path}")
    probe_lineage = provenance.get("train", {}).get("feature_provenance")
    if (
        not isinstance(probe_lineage, dict)
        or not probe_lineage
        or probe_lineage != provenance.get("val", {}).get("feature_provenance")
        or probe_lineage != dataset.provenance.get("feature_provenance")
    ):
        raise ValueError(f"B2 probe feature lineage differs from the test panel: {path}")
    probe = MultiTaskPhysicalProbe(**checkpoint["model_config"]).to(device).eval()
    probe.load_state_dict(checkpoint["model_state_dict"], strict=True)
    rows = []
    with torch.inference_mode():
        for rec in tqdm(records, desc="B2 probe"):
            task = rec["task_name"]
            history = dataset.get_physical_tokens(rec).unsqueeze(0).to(device)
            logits = probe(history)[task].squeeze(0)
            label = CLASS_NAMES[task][int(logits.argmax().item())]
            answer = row(rec, "B2", json.dumps({task: label}))
            answer["output_type"] = "direct_probe"
            answer["raw_logits"] = [float(x) for x in logits.cpu().tolist()]
            rows.append(answer)
    return rows, {"path": str(path), "sha256": sha256(path), "provenance": provenance}


def report(result: dict[str, Any]) -> str:
    lines = ["# Stage 4 matched reasoning benchmark", "",
             f"Test QA SHA-256: {result['manifest']['test_qa_sha256']}", "",
             "| Condition | Task | N | Parsed | Accuracy | Macro-F1 |",
             "|---|---|---:|---:|---:|---:|"]
    for condition, summary in result["conditions"].items():
        for task, metric in summary["tasks"].items():
            if metric["n"]:
                lines.append(
                    f"| {condition} | {task} | {metric['n']} | {metric['parsed']} | "
                    f"{metric['accuracy']:.3f} | {metric['macro_f1']:.3f} |"
                )
    primary = result["paired_comparisons"]["B4_minus_B3P"]["future_wrist_separation_change"]
    if primary["n"]:
        delta = primary["delta_macro_f1"]
        low, high = primary["cluster_ci95"]["delta_macro_f1"]
        lines.extend([
            "",
            f"Primary paired B4 minus B3P macro-F1: {delta:+.3f} "
            f"(95% recording-cluster interval {low:+.3f} to {high:+.3f}; "
            f"N={primary['n']}, recordings={primary['n_recording_clusters']}).",
        ])
    lines.extend([
        "", "Primary paired comparison: B4 minus B3P on future wrist-separation change.",
        "Additional ablations: B4 minus B3, B3P minus B3, and optional B5 minus B4.",
        "See JSON for recording-cluster intervals, subject and repetition-boundary strata, "
        "raw responses, and shuffle donor mappings.",
        "B2 is a direct probe when supplied; B5 is an optional observed-future diagnostic.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="eksperimen_model/configs/mmfi_projector_qwen.yaml")
    parser.add_argument("--test_file")
    parser.add_argument("--train_file")
    parser.add_argument("--features_dir")
    parser.add_argument("--dynamics_checkpoint")
    parser.add_argument("--dynamics_config")
    parser.add_argument("--checkpoint_b3")
    parser.add_argument("--checkpoint_b3p")
    parser.add_argument("--checkpoint_b4", "--projector_checkpoint", "--checkpoint")
    parser.add_argument("--checkpoint_b2", help="Optional validated direct-probe checkpoint")
    parser.add_argument("--checkpoint_b5", help="Optional separately trained B5 checkpoint")
    parser.add_argument("--output_json", default="docs/report_training/stage4_recovery_reasoning_benchmark.json")
    parser.add_argument("--output_report", default="docs/report_training/stage4_recovery_reasoning_benchmark.md")
    parser.add_argument("--overwrite_output", action="store_true",
                        help="Explicitly replace existing recovery benchmark outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output_path = Path(args.output_json)
    report_path = Path(args.output_report) if args.output_report else None
    if not args.overwrite_output:
        existing = [path for path in (output_path, report_path) if path is not None and path.exists()]
        if existing:
            raise FileExistsError(f"Evaluation output already exists; choose new paths or --overwrite_output: {existing}")

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Projector config missing: {config_path}")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    qa_path = Path(args.test_file or cfg["dataset"]["test_file"])
    train_path = Path(args.train_file or cfg["dataset"]["train_file"])
    features_dir = args.features_dir or cfg["dataset"]["features_dir"]
    records, train_records = load_records(qa_path), load_records(train_path)
    if not (Path(features_dir) / "test").is_dir():
        raise FileNotFoundError(f"Test feature directory missing: {Path(features_dir) / 'test'}")
    base_model = cfg["model"]["base_model"]
    projector_dir = Path(cfg["training"]["output_dir"])
    paths = {
        "B3": Path(args.checkpoint_b3) if args.checkpoint_b3 else projector_dir / "b3" / "best_projector.pth",
        "B3P": Path(args.checkpoint_b3p) if args.checkpoint_b3p else projector_dir / "b3p" / "best_projector.pth",
        "B4": Path(args.checkpoint_b4) if args.checkpoint_b4 else projector_dir / "b4" / "best_projector.pth",
    }
    if args.checkpoint_b5:
        paths["B5"] = Path(args.checkpoint_b5)
    checkpoints = {name: load_checkpoint(path, name, base_model) for name, path in paths.items()}
    config_digest, train_digest = sha256(config_path), sha256(train_path)
    val_path = Path(cfg["dataset"]["val_file"])
    if not val_path.is_file():
        raise FileNotFoundError(f"Validation QA manifest missing: {val_path}")
    val_records = load_records(val_path)
    split_subjects = {
        "train": {rec["subject"] for rec in train_records},
        "val": {rec["subject"] for rec in val_records},
        "test": {rec["subject"] for rec in records},
    }
    if (
        split_subjects["train"] & split_subjects["val"]
        or split_subjects["train"] & split_subjects["test"]
        or split_subjects["val"] & split_subjects["test"]
    ):
        raise ValueError("Train, validation, and test QA must have disjoint subjects")
    val_digest = sha256(val_path)
    seeds = set()
    matched_training = None
    expected_train_ids = hashlib.sha256(json.dumps(
        [rec["sample_id"] for rec in train_records], separators=(",", ":")
    ).encode()).hexdigest()
    expected_val_ids = hashlib.sha256(json.dumps(
        [rec["sample_id"] for rec in val_records], separators=(",", ":")
    ).encode()).hexdigest()
    for name, checkpoint in checkpoints.items():
        provenance = checkpoint["provenance"]
        if (
            provenance.get("config_sha256") != config_digest
            or provenance.get("train_qa_sha256") != train_digest
            or provenance.get("val_qa_sha256") != val_digest
        ):
            raise ValueError(f"{name} checkpoint was trained with different config or QA manifests")
        seeds.add(provenance.get("seed"))
        if name in ("B3", "B3P", "B4"):
            if (
                provenance.get("train_sample_ids_sha256") != expected_train_ids
                or provenance.get("val_sample_ids_sha256") != expected_val_ids
            ):
                raise ValueError(f"{name} checkpoint used a different QA sample panel")
            training = {
                key: provenance.get(key) for key in
                ("train_sample_count", "val_sample_count", "train_sample_ids_sha256",
                 "val_sample_ids_sha256", "optimization")
            }
            if matched_training is None:
                matched_training = training
            elif training != matched_training:
                raise ValueError(f"{name} training panel or optimization differs from matched conditions")
    if len(seeds) != 1 or None in seeds:
        raise ValueError("Projector comparison requires the same recorded training seed")
    dynamics_path = Path(args.dynamics_checkpoint or cfg["dataset"]["dynamics_checkpoint"])
    if not dynamics_path.is_file():
        raise FileNotFoundError(f"Frozen Dynamics checkpoint missing: {dynamics_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device.type == "cuda" else torch.float32
    )
    wrapper = PhysicalSLMWrapper(
        model_name_or_path=base_model, in_dim=cfg["model"]["in_dim"],
        hidden_dim=cfg["model"]["hidden_dim"], freeze_slm=True,
        use_gradient_checkpointing=False, torch_dtype=dtype, device_map=device.type,
    )
    datasets = {
        name: GroundedQADataset(
            qa_jsonl_path=str(qa_path), features_dir=features_dir,
            tokenizer=wrapper.tokenizer, split="test", condition=name,
            dynamics_checkpoint=str(dynamics_path) if name == "B4" else None,
            dynamics_config=(args.dynamics_config or cfg["dataset"].get("dynamics_config")) if name == "B4" else None,
            dynamics_device=device.type, preload_features=True,
        )
        for name in paths
    }
    panel = [rec["sample_id"] for rec in records]
    for name, dataset in datasets.items():
        if [rec["sample_id"] for rec in dataset.records] != panel:
            raise ValueError(f"{name} dataset changed the matched QA panel")
        if dataset.token_count != checkpoints[name]["provenance"]["token_count"]:
            raise ValueError(f"{name} dataset token count differs from training")
        trained_lineage = checkpoints[name]["provenance"].get("dataset", {}).get("feature_provenance")
        observed_lineage = dataset.provenance.get("feature_provenance")
        if not isinstance(trained_lineage, dict) or not trained_lineage or trained_lineage != observed_lineage:
            raise ValueError(f"{name} test feature lineage differs from training")
        if name == "B4":
            trained = checkpoints[name]["provenance"].get("dataset", {})
            for key in ("dynamics_checkpoint_sha256", "normalization_stats_sha256"):
                if trained.get(key) != dataset.provenance.get(key):
                    raise ValueError(f"B4 {key} differs from training provenance")
    # Reject a damaged window or mismatched manifest before spending time on generation.
    inspected_windows = set()
    for rec in records:
        window_key = (rec["feature_file"], rec["window_start"])
        if window_key not in inspected_windows:
            _, frame_ids = datasets["B3"]._window(rec)
            source_ids = rec.get("source_frame_ids")
            if not isinstance(source_ids, list) or frame_ids.tolist() != source_ids:
                raise ValueError(f"QA frame IDs differ from feature window: {rec['sample_id']}")
            if int(frame_ids[15]) != rec.get("t_curr_frame") or int(frame_ids[-1]) != rec.get("t_future_frame"):
                raise ValueError(f"QA time anchors differ from feature window: {rec['sample_id']}")
            for dataset in datasets.values():
                dataset.get_physical_tokens(rec)
            inspected_windows.add(window_key)
    rows: dict[str, list[dict[str, Any]]] = {}
    rows["B1"] = [
        row(rec, "B1", generate(wrapper, datasets["B3"], rec, device, "B1"))
        for rec in tqdm(records, desc="B1 text only")
    ]
    majority = {}
    for task in TASK_LABELS:
        labels = Counter(rec["target_structured"][task] for rec in train_records if rec["task_name"] == task)
        if not labels:
            raise ValueError(f"No B0 training labels for {task}")
        majority[task] = labels.most_common(1)[0][0]
    rows["B0"] = [
        row(rec, "B0", json.dumps({rec["task_name"]: (
            "stable" if rec["task_name"] == "future_wrist_separation_change" else majority[rec["task_name"]]
        )}))
        for rec in records
    ]
    probe_manifest = None
    if args.checkpoint_b2:
        rows["B2"], probe_manifest = evaluate_probe(
            Path(args.checkpoint_b2), datasets["B3"], records, device, train_digest, val_digest
        )
    for name, checkpoint in checkpoints.items():
        wrapper.projector.load_state_dict(checkpoint["projector_state_dict"], strict=True)
        wrapper.eval()
        dataset = datasets[name]
        rows[name] = [
            row(rec, name, generate(wrapper, dataset, rec, device, name))
            for rec in tqdm(records, desc=name)
        ]
        if name == "B4":
            for kind in ("cross_action", "within_action"):
                donors = select_donors(records, kind, args.seed)
                control = f"B4_{kind}_shuffle"
                rows[control] = [
                    row(rec, control, generate(wrapper, dataset, rec, device, name, donors[rec["sample_id"]]),
                        donors[rec["sample_id"]])
                    for rec in tqdm(records, desc=control) if rec["sample_id"] in donors
                ]

    comparisons = {
        f"B4_minus_{other}": {
            task: paired_delta(rows["B4"], rows[other], task, args.seed) for task in TASK_LABELS
        }
        for other in ("B3P", "B3", "B1")
    }
    comparisons["B3P_minus_B3"] = {
        task: paired_delta(rows["B3P"], rows["B3"], task, args.seed) for task in TASK_LABELS
    }
    if "B5" in rows:
        comparisons["B5_minus_B4"] = {
            task: paired_delta(rows["B5"], rows["B4"], task, args.seed) for task in TASK_LABELS
        }
    controls = {}
    for kind in ("cross_action", "within_action"):
        name = f"B4_{kind}_shuffle"
        control_rows = rows[name]
        by_task = {}
        for task in TASK_LABELS:
            task_rows = [r for r in control_rows if r["task_name"] == task]
            eligible = {r["sample_id"] for r in task_rows}
            normal = [r for r in rows["B4"] if r["sample_id"] in eligible]
            normal_by_id = {r["sample_id"]: r for r in normal}
            by_task[task] = {
                "n": len(task_rows),
                "donor_target_match_rate": (
                    sum(r["matches_donor_target"] for r in task_rows) / len(task_rows)
                    if task_rows else None
                ),
                "unshuffled_donor_target_match_rate": (
                    sum(normal_by_id[r["sample_id"]]["prediction"] == r["donor_target"] for r in task_rows) / len(task_rows)
                    if task_rows else None
                ),
                "paired_B4_minus_shuffle": paired_delta(normal, task_rows, task, args.seed),
            }
        controls[name] = {
            "eligible_n": len(control_rows), "ineligible_n": len(records) - len(control_rows),
            "by_task": by_task,
        }
    result = {
        "manifest": {
            "test_qa": str(qa_path), "test_qa_sha256": sha256(qa_path),
            "train_qa": str(train_path), "train_qa_sha256": sha256(train_path),
            "val_qa": str(val_path), "val_qa_sha256": val_digest,
            "config": str(config_path), "config_sha256": sha256(config_path),
            "dynamics_checkpoint": str(dynamics_path),
            "dynamics_checkpoint_sha256": sha256(dynamics_path),
            "checkpoints": {
                name: {"path": str(path), "sha256": sha256(path), "provenance": checkpoints[name]["provenance"]}
                for name, path in paths.items()
            },
            "probe_checkpoint": probe_manifest,
            "test_dataset_provenance": {name: dataset.provenance for name, dataset in datasets.items()},
            "sample_count": len(records), "sample_ids": panel, "seed": args.seed,
            "split_subjects": {split: sorted(subjects) for split, subjects in split_subjects.items()},
        },
        "conditions": {name: summarize(data) for name, data in rows.items()},
        "paired_comparisons": comparisons, "shuffle_controls": controls,
        "rows": rows,
        "omitted": {"B2": "No validated task-matched probe checkpoint supplied"} if not args.checkpoint_b2 else {},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = report(result)
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
