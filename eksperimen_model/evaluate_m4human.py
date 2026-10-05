"""Recompute paired P0 metrics from immutable QA and raw prediction artifacts."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
import yaml

from eksperimen_model.datasets.m4human_qa_dataset import canonical_answer, read_qa
from eksperimen_model.m4human_evaluation import language_dependency_hashes, paired_bootstrap, score_rows, score_donors, donor_pairs, verify_test_contract
from eksperimen_model.utils.m4human_runtime import CONTRACT_VERSION, atomic_json, canonical_hash, file_sha256, require_lineage, state_dict_hash, validate_run_artifacts


def create_test_lock(args, rows):
    required = ("config", "base_cache", "kin_cache", "base_checkpoint", "kin_checkpoint", "effect_tolerance_json", "uncertainty_json")
    if any(not getattr(args, name) for name in required):
        raise ValueError(f"Test lock requires {', '.join(required)}")
    config = yaml.safe_load(Path(args.config).read_text())
    if any(value is None for value in config["lineage"].values()):
        raise ValueError("Resolve audited lineage before test lock")
    caches, checkpoints, alignment = {}, {}, {}
    for condition, cache_path, checkpoint_path in (("C_base", args.base_cache, args.base_checkpoint), ("C_kin", args.kin_cache, args.kin_checkpoint)):
        cache = Path(cache_path)
        metadata_path = cache / "metadata.json" if cache.is_dir() else cache
        metadata = json.loads(metadata_path.read_text())
        metadata = metadata.get("metadata", metadata)
        require_lineage(metadata, config["lineage"])
        if metadata.get("condition") != condition or metadata.get("representation") != "exact_U" or metadata.get("complete") is not True or metadata.get("K") != config["primary_budget"] or metadata.get("data_kind") != "m4human" or metadata.get("scientific_eligible") is not True:
            raise ValueError(f"Foreign/incomplete exact-U cache for {condition}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        require_lineage(payload["metadata"], config["lineage"])
        tokenizer_hash = metadata.get("tokenizer_hash", metadata.get("lineage", {}).get("tokenizer_hash"))
        if payload["metadata"].get("condition") != condition or payload["metadata"].get("K") != metadata["K"] or payload["metadata"].get("tokenizer_hash") != tokenizer_hash or payload["state_hash"] != state_dict_hash(payload["state_dict"]):
            raise ValueError(f"Foreign/invalid selected checkpoint for {condition}")
        if payload["metadata"].get("cache_sha256") != file_sha256(metadata_path) or payload["metadata"].get("qa_sha256") != file_sha256(args.qa) or payload["metadata"].get("config_hash") != canonical_hash(config):
            raise ValueError(f"Selected checkpoint provenance mismatch for {condition}")
        run = Path(checkpoint_path).parent
        if Path(checkpoint_path).name != "best.pt" or validate_run_artifacts(run):
            raise ValueError(f"Selected run artifacts incomplete for {condition}")
        metrics = json.loads((run / "metrics.json").read_text())
        if metrics.get("status") != "complete" or metrics.get("analysis_ready") is not True:
            raise ValueError(f"Selected run incomplete for {condition}")
        alignment[condition] = (payload["metadata"].get("initial_sha256"), metrics.get("exposure_hash"), metrics.get("successful_updates"), metrics.get("n_QA_exposures"))
        caches[condition], checkpoints[condition] = file_sha256(metadata_path), file_sha256(checkpoint_path)
    if any(value is None for value in alignment["C_base"]) or alignment["C_base"] != alignment["C_kin"]:
        raise ValueError("Paired projector initialization, QA exposure or update budget differs")
    for row in rows:
        if row["recipe_hash"] != config["lineage"]["recipe_hash"] or row["split_hash"] != config["lineage"]["split_hash"] or row["joint_map_hash"] != config["lineage"]["joint_map_hash"]:
            raise ValueError("QA reference lineage differs from locked config")
    test_rows = [row for row in rows if row["split"] == "test" and row["target_status"] != "undefined"]
    if not test_rows:
        raise ValueError("No eligible test QA")
    uncertainty = json.loads(Path(args.uncertainty_json).read_text())
    if not isinstance(uncertainty, dict) or uncertainty.get("method") != "paired_subject_cluster_row_weighted" or type(uncertainty.get("draws")) is not int or uncertainty["draws"] < 1 or type(uncertainty.get("seed")) is not int:
        raise ValueError("Unsupported uncertainty lock policy")
    payload = {"contract_version": CONTRACT_VERSION, "qa_sha256": file_sha256(args.qa), "cache_sha256": caches,
        "checkpoint_sha256": checkpoints, "selected_tasks": config["selected_tasks"],
        "recipe_hash": config["lineage"]["recipe_hash"], "split_hash": config["lineage"]["split_hash"],
        "parser_hash": file_sha256(Path(__file__).with_name("m4human_evaluation.py")),
        "language_dependency_sha256": language_dependency_hashes(),
        "generation": config["generation"], "config_hash": canonical_hash(config),
        "paired_alignment": {"initial_sha256": alignment["C_base"][0], "exposure_hash": alignment["C_base"][1], "successful_updates": alignment["C_base"][2], "n_QA_exposures": alignment["C_base"][3]},
        "effect_tolerance": json.loads(Path(args.effect_tolerance_json).read_text()),
        "uncertainty": uncertainty,
        "donor_panel": donor_pairs(test_rows)}
    atomic_json(args.output, {**payload, "lock_hash": canonical_hash(payload)})


def load_predictions(path, contract=None, condition=None):
    payload = json.loads(Path(path).read_text())
    if contract:
        if not isinstance(payload, dict) or payload.get("condition") != condition or payload.get("split") != "test":
            raise ValueError("Test prediction artifact has wrong condition/split")
        if payload.get("qa_sha256") != contract["qa_sha256"] or payload.get("checkpoint_sha256") != contract["checkpoint_sha256"][condition] or payload.get("cache_sha256") != contract["cache_sha256"][condition]:
            raise ValueError("Test prediction artifact differs from locked inputs")
    return payload["predictions"] if isinstance(payload, dict) else payload


def majority_predictions(train, cohort):
    classes = defaultdict(Counter)
    for row in train:
        if row["target_status"] != "undefined":
            classes[(row["task"],row.get("body_part"))][row["answer"]] += 1
    answers = {task: sorted(counts, key=lambda label: (-counts[label], label))[0] for task, counts in classes.items()}
    return [{"qa_id": row["qa_id"], "raw_text": canonical_answer(row["task"], answers[(row["task"],row.get("body_part"))]), "raw_ids": []} for row in cohort]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", required=True)
    parser.add_argument("--base")
    parser.add_argument("--kin")
    parser.add_argument("--create-test-lock", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--base-cache")
    parser.add_argument("--kin-cache")
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--kin-checkpoint")
    parser.add_argument("--effect-tolerance-json")
    parser.add_argument("--uncertainty-json")
    parser.add_argument("--text")
    parser.add_argument("--rule", help="Raw category prediction artifact produced from sensor predicted physical states")
    parser.add_argument("--probe", help="Raw category predictions from independent physical U probe + shared recipe")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--test-contract")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = read_qa(args.qa)
    if args.create_test_lock:
        create_test_lock(args, rows)
        return
    if not args.base or not args.kin:
        raise ValueError("Paired scoring requires --base and --kin prediction artifacts")
    cohort = [row for row in rows if row["split"] == args.split and row["target_status"] != "undefined"]
    contract = None
    if args.split == "test":
        if not args.test_contract:
            raise ValueError("Test results require locked protocol")
        contract = verify_test_contract(args.test_contract, qa_path=args.qa)
    tasks = contract["selected_tasks"] if contract else ["root_speed_trend", "relative_limb_motion"]
    if args.draws < 1:
        raise ValueError("At least one bootstrap draw required")
    base, kin = load_predictions(args.base, contract, "C_base"), load_predictions(args.kin, contract, "C_kin")
    panel = contract["donor_panel"] if contract else donor_pairs(cohort)
    policy = contract["uncertainty"] if contract else {"draws": args.draws, "seed": 42}
    report = {"C_base": score_rows(cohort, base, tasks), "C_kin": score_rows(cohort, kin, tasks), "C_majority": score_rows(cohort, majority_predictions([row for row in rows if row["split"] == "train"], cohort), tasks), "paired_uncertainty": paired_bootstrap(cohort, base, kin, tasks, policy["draws"], policy["seed"]), "grounding": {"C_base": score_donors(panel, cohort, base), "C_kin": score_donors(panel, cohort, kin)}, "counts": {"n_qa": len(cohort), "n_window": len({row["window_id"] for row in cohort}), "n_recording": len({row["recording_id"] for row in cohort}), "n_subject": len({row["subject_id"] for row in cohort}), "excluded": sum(row["target_status"] == "undefined" and row["split"] == args.split for row in rows)}, "lineage": {"qa_sha256": file_sha256(args.qa), "base_sha256": file_sha256(args.base), "kin_sha256": file_sha256(args.kin)}, "status": "exploratory_one_seed_PoC"}
    report.update(test_lock_hash=contract['lock_hash'] if contract else None,
                  effect_tolerance=contract['effect_tolerance'] if contract else None)
    for name, path in (("C_text", args.text), ("C_rule", args.rule), ("C_probe", args.probe)):
        if path:
            report[name] = score_rows(cohort, load_predictions(path), tasks)
        else:
            report[name] = {"status": "not_run", "reason": "Prediction artifact not supplied"}
    atomic_json(args.output, report)


if __name__ == "__main__":
    main()
