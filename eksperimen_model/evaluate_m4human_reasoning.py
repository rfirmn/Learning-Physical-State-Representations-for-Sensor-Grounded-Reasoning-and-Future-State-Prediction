"""Generate held-out raw QA answers with locked checkpoint, cache and protocol."""
import argparse
from pathlib import Path

import yaml

from eksperimen_model.datasets.m4human_qa_dataset import M4HumanQADataset, cache_metadata_path
from eksperimen_model.models.m4human_projector import load_frozen_qwen
from eksperimen_model.m4human_evaluation import verify_test_contract, score_rows, donor_pairs, score_donors
from eksperimen_model.train_m4human_projector import generate_predictions
from eksperimen_model.utils.m4human_runtime import atomic_json, file_sha256, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="eksperimen_model/configs/m4human_projector.yaml")
    parser.add_argument("--qa", required=True)
    parser.add_argument("--u-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--condition", choices=["C_base", "C_kin"], required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--test-contract")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    contract = None
    if args.split == "test":
        if not args.test_contract:
            raise ValueError("Test evaluation requires prelocked contract")
        contract = verify_test_contract(args.test_contract, args.qa, args.u_cache, config,
            checkpoint_paths={args.condition: args.checkpoint}, condition=args.condition)
    dataset = M4HumanQADataset(args.qa, args.u_cache, args.split, args.condition, config["lineage"], args.test_contract)
    model = load_frozen_qwen(config["llm"], args.device, config["training"]["precision"])
    model.max_prefix_tokens = config["training"]["max_prefix_tokens"]
    model.max_total_tokens = config["training"]["max_total_tokens"]
    load_checkpoint(args.checkpoint, model.projector, dict(config["lineage"], condition=args.condition, K=dataset.metadata["K"], tokenizer_hash=dataset.metadata["tokenizer_hash"]))
    if args.text_only:
        original = model.generate
        model.generate = lambda samples, max_new_tokens: original(samples, max_new_tokens=max_new_tokens, text_only=True)
    predictions = generate_predictions(model, dataset, config["generation"]["max_new_tokens"])
    panel = contract["donor_panel"] if contract else donor_pairs(dataset.records)
    atomic_json(args.output, {"condition": "C_text" if args.text_only else args.condition, "split": args.split, "predictions": predictions, "metrics": score_rows(dataset.records, predictions, config["selected_tasks"]), "grounding": score_donors(panel, dataset.records, predictions), "qa_sha256": file_sha256(args.qa), "checkpoint_sha256": file_sha256(args.checkpoint), "cache_sha256": file_sha256(cache_metadata_path(args.u_cache)), "test_contract_sha256": file_sha256(args.test_contract) if contract else None})


if __name__ == "__main__":
    main()
