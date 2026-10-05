"""P0 structured scoring, locked panels, paired subject uncertainty and donor audit."""
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from eksperimen_model.utils.m4human_kinematics import TASK_DEFINITIONS
from eksperimen_model.utils.m4human_runtime import CONTRACT_VERSION, canonical_hash, file_sha256


def parse_response(raw, task):
    def no_duplicate(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_key")
            result[key] = value
        return result
    try:
        obj = json.loads(raw, object_pairs_hook=no_duplicate, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite_json")))
        if not isinstance(obj, dict) or set(obj) != {"task", "answer"}:
            raise ValueError("wrong_keys")
        if obj["task"] != task or not isinstance(obj["answer"], str) or obj["answer"] not in TASK_DEFINITIONS[task]["answers"]:
            raise ValueError("wrong_task_or_enum")
        return {"parse_valid": True, "parsed_answer": obj["answer"], "parse_error": None}
    except (ValueError, TypeError, KeyError) as error:
        return {"parse_valid": False, "parsed_answer": None, "parse_error": str(error)}


def score_rows(qa_rows, predictions, selected_tasks=None, weights=None):
    from eksperimen_model.datasets.m4human_qa_dataset import validate_qa
    for row in qa_rows:
        validate_qa(row)
    if len({row["qa_id"] for row in qa_rows}) != len(qa_rows):
        raise ValueError("Duplicate reference QA IDs")
    tasks = selected_tasks if selected_tasks is not None else sorted({row["task"] for row in qa_rows})
    if not tasks or len(set(tasks)) != len(tasks) or any(task not in TASK_DEFINITIONS for task in tasks):
        raise ValueError("Empty or unsupported selected task panel")
    if weights is not None and (len(weights) != len(qa_rows) or any(not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0 for weight in weights)):
        raise ValueError("Invalid scoring weights")
    by_id = {row["qa_id"]: row for row in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("Duplicate prediction IDs")
    if set(by_id) - {row["qa_id"] for row in qa_rows}:
        raise ValueError("Unexpected prediction QA IDs")
    counts = {task: defaultdict(float) for task in tasks}
    for index, row in enumerate(qa_rows):
        if row["target_status"] == "undefined" or row["task"] not in tasks:
            continue
        weight = 1.0 if weights is None else weights[index]
        parsed = parse_response(by_id.get(row["qa_id"], {}).get("raw_text", ""), row["task"])
        answer = parsed["parsed_answer"] if parsed["parse_valid"] else "__invalid__"
        counts[row["task"]][(row["answer"], answer)] += weight
    result = {}
    for task in tasks:
        classes = TASK_DEFINITIONS[task]["answers"]
        confusion = counts[task]
        total = sum(confusion.values())
        f1, support = {}, {}
        for label in classes:
            tp = confusion[(label, label)]
            fp = sum(value for (truth, pred), value in confusion.items() if pred == label and truth != label)
            fn = sum(value for (truth, pred), value in confusion.items() if truth == label and pred != label)
            f1[label] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            support[label] = tp + fn
        valid = sum(value for (_, pred), value in confusion.items() if pred != "__invalid__")
        correct = sum(confusion[(label, label)] for label in classes)
        result[task] = {"n": total, "macro_f1": sum(f1.values()) / len(classes), "accuracy": correct / total if total else None, "parse_coverage": valid / total if total else None, "conditional_accuracy": correct / valid if valid else None, "f1": f1, "class_support": support, "abstention_defined": sum(value for (truth, pred), value in confusion.items() if truth != "unknown" and pred == "unknown"), "false_certainty_unknown": sum(value for (truth, pred), value in confusion.items() if truth == "unknown" and pred not in {"unknown", "__invalid__"}), "confusion": {truth: {pred: confusion[(truth, pred)] for pred in (*classes, "__invalid__")} for truth in classes}}
    return {"score": sum(row["macro_f1"] for row in result.values()) / len(tasks), "tasks": result, "unsupported_tasks": [task for task in tasks if not result[task]["n"]]}


def paired_bootstrap(qa_rows, base, kin, selected_tasks, draws=2000, seed=42):
    if type(draws) is not int or draws < 1:
        raise ValueError("Positive integer bootstrap draws required")
    subjects = sorted({row["subject_id"] for row in qa_rows if row["target_status"] != "undefined"})
    if not subjects:
        raise ValueError("No eligible subjects")
    rng, deltas, unsupported = random.Random(seed), [], 0
    for _ in range(draws):
        frequencies = Counter(rng.choices(subjects, k=len(subjects)))
        weights = [frequencies[row["subject_id"]] for row in qa_rows]
        baseline = score_rows(qa_rows, base, selected_tasks, weights)
        proposed = score_rows(qa_rows, kin, selected_tasks, weights)
        unsupported += bool(baseline["unsupported_tasks"])
        deltas.append(proposed["score"] - baseline["score"])
    deltas.sort()
    per_subject = {}
    for subject in subjects:
        cohort = [row for row in qa_rows if row["subject_id"] == subject]
        ids = {row["qa_id"] for row in cohort}
        per_subject[subject] = score_rows(cohort, [row for row in kin if row["qa_id"] in ids], selected_tasks)["score"] - score_rows(cohort, [row for row in base if row["qa_id"] in ids], selected_tasks)["score"]
    return {"method": "paired_subject_cluster_row_weighted", "conditional_on_fitted_checkpoints": True, "seed": seed, "draws": draws, "n_subjects": len(subjects), "delta": score_rows(qa_rows, kin, selected_tasks)["score"] - score_rows(qa_rows, base, selected_tasks)["score"], "ci95": [deltas[int(.025 * (draws - 1))], deltas[int(.975 * (draws - 1))]] if len(subjects) >= 3 and not unsupported else None, "unsupported_resamples": unsupported, "per_subject_delta": per_subject, "limitation": "Exploratory one-seed PoC; insufficient subjects or unsupported resamples yield descriptive deltas only"}


def language_dependency_hashes():
    root = Path(__file__).parent
    names = ("m4human_evaluation.py", "models/m4human_projector.py", "datasets/m4human_qa_dataset.py",
             "train_m4human_projector.py", "evaluate_m4human_reasoning.py", "evaluate_m4human.py")
    return {name: file_sha256(root / name) for name in names}


def donor_compatibility_key(row):
    required = ("action_id", "coordinate_hash", "observation_interval_s", "task_interval_s", "time_s", "label_support")
    if any(row.get(key) is None for key in required) or not (type(row["action_id"]) is int or isinstance(row["action_id"], str) and row["action_id"]) or not row["coordinate_hash"]:
        return None
    times = row["time_s"]
    if len(times) != 32 or not all(isinstance(t, (int, float)) and math.isfinite(t) for t in times):
        return None
    grid = tuple(round(t - times[0], 6) for t in times)
    return (row["action_id"], row["task"], row.get("body_part"), row["question"],
            tuple(round(t, 6) for t in row["task_interval_s"]), tuple(round(t, 6) for t in row["observation_interval_s"]),
            grid, tuple(row["label_support"]), row["recipe_hash"], row["joint_map_hash"], row["coordinate_hash"], row["split"])


def donor_pairs(qa_rows):
    eligible = [row for row in qa_rows if row["target_status"] == "defined"]
    groups = defaultdict(list)
    for row in eligible:
        # Task/query compatibility is exact and source-independent, including relative intervals.
        key = donor_compatibility_key(row)
        if key is not None:
            groups[key].append(row)
    pairs, reasons = [], Counter()
    for row in eligible:
        key = donor_compatibility_key(row)
        candidates = [other for other in groups.get(key, []) if other["recording_id"] != row["recording_id"] and other["answer"] != row["answer"]]
        if candidates:
            donor = min(candidates, key=lambda candidate: candidate["qa_id"])
            pairs.append({"source_qa_id": row["qa_id"], "donor_qa_id": donor["qa_id"], "source_reference": row["answer"], "donor_reference": donor["answer"]})
        else:
            reasons["no_compatible_different_reference_recording"] += 1
    return {"pairs": pairs, "eligible_query_windows": len({row["window_id"] for row in eligible}), "paired_windows": len({row["window_id"] for row in eligible if any(pair["source_qa_id"] == row["qa_id"] for pair in pairs)}), "reasons": dict(reasons)}


def score_donors(panel, qa_rows, predictions):
    by_qa = {row["qa_id"]: row for row in qa_rows}
    by_pred = {row["qa_id"]: row for row in predictions}
    if len(by_pred) != len(predictions):
        raise ValueError("Duplicate donor prediction IDs")
    both, initial, raw_change = 0, 0, 0
    for pair in panel["pairs"]:
        source, donor = (by_qa[pair[key]] for key in ("source_qa_id", "donor_qa_id"))
        if (source["target_status"] != "defined" or donor["target_status"] != "defined"
                or source["answer"] == donor["answer"] or source["recording_id"] == donor["recording_id"]
                or donor_compatibility_key(source) is None or donor_compatibility_key(source) != donor_compatibility_key(donor)
                or pair["source_reference"] != source["answer"] or pair["donor_reference"] != donor["answer"]):
            raise ValueError("Invalid or stale donor reference pair")
        parsed = [parse_response(by_pred.get(row["qa_id"], {}).get("raw_text", ""), row["task"]) for row in (source, donor)]
        correct = [result["parse_valid"] and result["parsed_answer"] == row["answer"] for result, row in zip(parsed, (source, donor))]
        both += all(correct)
        initial += correct[0]
        raw_change += all(result["parse_valid"] for result in parsed) and parsed[0]["parsed_answer"] != parsed[1]["parsed_answer"]
    n = len(panel["pairs"])
    return {"n_eligible_pairs": n, "PairBothCorrect": both / n if n else None, "CorrectChange": both / n if n else None, "ConditionalCorrectChange": both / initial if initial else None, "n_initial_correct": initial, "RawChangeRate": raw_change / n if n else None, "coverage": panel["paired_windows"] / panel["eligible_query_windows"] if panel["eligible_query_windows"] else None, "pair_ids": panel["pairs"], "n_unique_recordings": len({by_qa[pair[key]]["recording_id"] for pair in panel["pairs"] for key in ("source_qa_id", "donor_qa_id")}), "n_unique_subjects": len({by_qa[pair[key]]["subject_id"] for pair in panel["pairs"] for key in ("source_qa_id", "donor_qa_id")}), "ci": None, "ci_reason": "Donor reuse/cross-subject dependencies require grouping both pair members"}


def verify_test_contract(path, qa_path=None, cache_manifest=None, config=None, checkpoint_paths=None, condition=None):
    contract = json.loads(Path(path).read_text())
    required = ("contract_version", "qa_sha256", "cache_sha256", "selected_tasks", "recipe_hash", "split_hash", "parser_hash", "language_dependency_sha256", "generation", "checkpoint_sha256", "effect_tolerance", "uncertainty", "donor_panel")
    if any(contract.get(key) is None for key in required):
        raise ValueError("Incomplete test contract; fail closed")
    if contract["contract_version"] != CONTRACT_VERSION:
        raise ValueError("Foreign test contract")
    if not isinstance(contract["checkpoint_sha256"], dict) or set(contract["checkpoint_sha256"]) != {"C_base", "C_kin"}:
        raise ValueError("Both selected checkpoints must be locked")
    if not isinstance(contract["cache_sha256"], dict) or set(contract["cache_sha256"]) != {"C_base", "C_kin"}:
        raise ValueError("Both exact-U caches must be locked")
    if not isinstance(contract["selected_tasks"], list) or not contract["selected_tasks"] or any(task not in TASK_DEFINITIONS for task in contract["selected_tasks"]):
        raise ValueError("Unsupported locked tasks")
    payload = {key: value for key, value in contract.items() if key != "lock_hash"}
    if contract.get("lock_hash") != canonical_hash(payload):
        raise ValueError("Test contract checksum mismatch")
    if qa_path and contract["qa_sha256"] != file_sha256(qa_path):
        raise ValueError("Test QA differs from locked panel")
    from eksperimen_model.datasets.m4human_qa_dataset import cache_metadata_path
    if cache_manifest and (condition not in contract["cache_sha256"] or file_sha256(cache_metadata_path(cache_manifest)) != contract["cache_sha256"][condition]):
        raise ValueError("Test U cache differs from lock")
    if checkpoint_paths:
        for name, checkpoint_path in checkpoint_paths.items():
            if name not in contract["checkpoint_sha256"] or file_sha256(checkpoint_path) != contract["checkpoint_sha256"][name]:
                raise ValueError(f"Checkpoint {name} differs from test lock")
    if contract["parser_hash"] != file_sha256(__file__):
        raise ValueError("Parser/evaluator changed after test lock")
    if contract["language_dependency_sha256"] != language_dependency_hashes():
        raise ValueError("Language implementation changed after test lock")
    uncertainty = contract["uncertainty"]
    if not isinstance(uncertainty, dict) or uncertainty.get("method") != "paired_subject_cluster_row_weighted" or type(uncertainty.get("draws")) is not int or uncertainty["draws"] < 1 or type(uncertainty.get("seed")) is not int:
        raise ValueError("Unsupported locked uncertainty policy")
    if config is not None and contract.get("config_hash") != canonical_hash(config):
        raise ValueError("Test config differs from lock")
    if config is not None and (contract["selected_tasks"] != config["selected_tasks"] or contract["generation"] != config["generation"]):
        raise ValueError("Task/generation policy differs from lock")
    return contract
