"""Matched stage-L alignment; only projector parameters enter the optimizer."""
import argparse
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import yaml

from eksperimen_model.datasets.m4human_qa_dataset import M4HumanQADataset, cache_metadata_path
from eksperimen_model.models.m4human_projector import load_frozen_qwen
from eksperimen_model.m4human_evaluation import score_rows, parse_response
from eksperimen_model.utils.m4human_runtime import (RunLogger, adamw_parameters, atomic_json, canonical_hash,
    file_sha256, load_checkpoint, save_checkpoint, seed_everything, state_dict_hash, rng_state, restore_rng_state)


def write_jsonl(path, rows):
    path = Path(path)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def generate_predictions(model, dataset, max_new_tokens):
    records = []
    for index, row in enumerate(dataset.records):
        record = {key: row[key] for key in ("qa_id", "window_id", "subject_id", "recording_id", "task")}
        record.update(condition=dataset.metadata["condition"], K=dataset.metadata["K"])
        try:
            sample = dataset[index]
            record["n_valid_tokens"] = int(sample["token_mask"].sum())
            result, _ = model.generate([sample], max_new_tokens=max_new_tokens)
            record.update(result[0], error=None)
        except (ValueError, RuntimeError) as error:
            record.update(raw_ids=[], raw_text="", error=str(error), n_valid_tokens=record.get("n_valid_tokens"))
        record.update(parse_response(record["raw_text"], row["task"]))
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="eksperimen_model/configs/m4human_projector.yaml")
    parser.add_argument("--qa", required=True)
    parser.add_argument("--u-cache", required=True)
    parser.add_argument("--condition", choices=["C_base", "C_kin"], required=True)
    parser.add_argument("--initial", required=True, help="Shared initial projector checkpoint, including train-only alpha")
    parser.add_argument("--initialize", action="store_true", help="Create common initial state from C_base train <=64 unique windows")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if any(value is None for value in config["lineage"].values()):
        raise ValueError("Resolve all audited lineage fields before alignment")
    seed_everything(config["training"]["seed"])
    settings = config["training"]
    if settings["micro_batch"] != 1 or settings["accumulation"] < 1 or settings["max_successful_updates"] < 1 or settings["epochs_cap"] < 1:
        raise ValueError("Locked reference requires micro_batch=1 and positive training budget")
    model = load_frozen_qwen(config["llm"], args.device, settings["precision"])
    model.max_prefix_tokens = config["training"]["max_prefix_tokens"]
    model.max_total_tokens = config["training"]["max_total_tokens"]
    train = M4HumanQADataset(args.qa, args.u_cache, "train", args.condition, config["lineage"])
    validation = M4HumanQADataset(args.qa, args.u_cache, "val", args.condition, config["lineage"])
    if not len(train) or not len(validation):
        raise ValueError("Both train and generated validation panels are required")
    if any(panel.metadata.get("data_kind") != "m4human" or panel.metadata.get("scientific_eligible") is not True for panel in (train, validation)):
        raise ValueError("Scientific alignment requires audited real M4Human exact-U caches")
    if train.metadata["K"] != validation.metadata["K"] or train.metadata["K"] != config["primary_budget"]:
        raise ValueError("Train/validation token budget differs from locked primary budget")
    initial_metadata = dict(config["lineage"], llm_revision=config["llm"]["revision"], seed=config["training"]["seed"], role="paired_initial")
    if args.initialize:
        if args.condition != "C_base" or Path(args.initial).exists():
            raise ValueError("Calibrate once on C_base train and never overwrite initial")
        unique, U, masks, text, calibration_ids = set(), [], [], [], []
        for index in range(len(train)):
            sample = train[index]
            if sample["window_id"] in unique:
                continue
            unique.add(sample["window_id"])
            calibration_ids.append(sample["window_id"])
            U.append(sample["U"].to(args.device))
            masks.append(sample["token_mask"].to(args.device))
            pre, post = model.prefix_ids(sample["question"], sample["task"], sample["answers"])
            text.append(model.llm.get_input_embeddings()(torch.tensor(pre + post, device=args.device)))
            if len(unique) == 64:
                break
        alpha = model.projector.calibrate(torch.stack(U), torch.stack(masks), torch.cat(text))
        save_checkpoint(args.initial, model.projector, initial_metadata, calibration_window_ids=calibration_ids, alpha=alpha)
        return
    initial = load_checkpoint(args.initial, model.projector, initial_metadata)
    metadata = dict(config["lineage"], condition=args.condition, K=train.metadata["K"], tokenizer_hash=train.metadata["tokenizer_hash"], initial_sha256=file_sha256(args.initial), qa_sha256=file_sha256(args.qa), cache_sha256=file_sha256(cache_metadata_path(args.u_cache)), config_hash=canonical_hash(config), llm_revision=config["llm"]["revision"], chat_template_hash=canonical_hash(model.tokenizer.chat_template), parser_hash=file_sha256(Path(__file__).with_name("m4human_evaluation.py")), prompt_hash=canonical_hash(model.prefix_ids.__code__.co_code.hex()))
    output = Path(args.output)
    logger = RunLogger(output, config, "L")
    optimizer = torch.optim.AdamW(adamw_parameters(model.projector, settings["weight_decay"]), lr=settings["learning_rate"])
    precision = settings["precision"]
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16_amp_grad_scaler")
    update, start_epoch, start_offset, best, history = 0, 0, 0, (-1.0, -1.0), []
    exposure_ids, skipped = [], 0
    frozen_hash = state_dict_hash(model.llm)
    if args.resume:
        resumed = load_checkpoint(args.resume, model.projector, metadata)
        optimizer.load_state_dict(resumed["optimizer"])
        scaler.load_state_dict(resumed["scaler"])
        restore_rng_state(resumed["rng"])
        update, start_epoch, start_offset = resumed["update"], resumed["next_epoch"], resumed["next_offset"]
        if start_offset < 0 or start_offset >= len(train) or start_offset % settings["accumulation"]:
            raise ValueError("Unsupported resume sampler boundary")
        best, history = tuple(resumed["best"]), resumed["history"]
        exposure_ids, skipped = resumed["exposure_ids"], resumed["skipped_updates"]
        previous_best = Path(args.resume).with_name("best.pt")
        if not previous_best.is_file():
            raise ValueError("Resume requires selected best checkpoint in parent run")
        load_checkpoint(previous_best, model.projector, metadata)
        shutil.copyfile(previous_best, output / "best.pt")
        model.projector.load_state_dict(resumed["state_dict"], strict=True)
        logger.log("resume", parent_run_id=Path(args.resume).parent.name, update=update, next_epoch=start_epoch, next_offset=start_offset)
    successful_budget = settings["max_successful_updates"]
    warmup = max(1, int(settings["warmup_fraction"] * successful_budget))
    if update >= successful_budget or start_epoch >= settings["epochs_cap"]:
        raise ValueError("Resume has no remaining update/epoch budget")
    try:
        for epoch in range(start_epoch, settings["epochs_cap"]):
            model.train()
            # Same order independent of condition, epoch-boundary resume exact.
            order = torch.randperm(len(train), generator=torch.Generator().manual_seed(settings["seed"] + epoch)).tolist()
            accumulation = settings["accumulation"]
            epoch_losses = []
            for offset in range(start_offset if epoch == start_epoch else 0, len(order), accumulation):
                group = order[offset:offset + accumulation]
                optimizer.zero_grad(set_to_none=True)
                group_losses = []
                started = time.perf_counter()
                for index in group:
                    with torch.autocast("cuda", dtype=torch.float16 if precision == "fp16_amp_grad_scaler" else torch.bfloat16, enabled=precision != "fp32"):
                        loss = model([train[index]])
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite answer-only CE")
                    scaler.scale(loss / len(group)).backward()
                    group_losses.append(float(loss.detach()))
                scaler.unscale_(optimizer)
                gradient = torch.nn.utils.clip_grad_norm_(model.projector.parameters(), settings["clip_grad_norm"])
                if not torch.isfinite(gradient):
                    raise ValueError("Nonfinite projector gradient")
                factor = (update + 1) / warmup if update < warmup else .5 * (1 + math.cos(math.pi * (update - warmup) / max(1, successful_budget - warmup)))
                for optimizer_group in optimizer.param_groups:
                    optimizer_group["lr"] = settings["learning_rate"] * factor
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                successful = scaler.get_scale() >= previous_scale
                if not successful:
                    skipped += 1
                    logger.log("overflow", epoch=epoch, offset=offset, skipped_updates=skipped)
                    raise ValueError("AMP overflow invalidates matched QA exposure; rerun both conditions with stable precision")
                update += 1
                exposure_ids.extend(train.records[index]["qa_id"] for index in group)
                epoch_losses.extend(group_losses)
                logger.log("update", epoch=epoch, update=update, answer_only_ce=sum(group_losses) / len(group_losses), gradient_norm=float(gradient), lr=optimizer.param_groups[0]["lr"], skipped_updates=skipped, seconds=time.perf_counter() - started, n_QA=len(group), exposure_hash=canonical_hash(exposure_ids))
                if any(parameter.grad is not None for parameter in model.llm.parameters()):
                    raise ValueError("Frozen LLM acquired parameter gradients")
                if update >= successful_budget:
                    break
            predictions = generate_predictions(model, validation, config["generation"]["max_new_tokens"])
            metrics = score_rows(validation.records, predictions, config["selected_tasks"])
            coverage = sum(row["parse_valid"] for row in predictions) / len(predictions)
            selected = (metrics["score"], coverage)
            history.append({"epoch": epoch, "update": update, "answer_only_ce": sum(epoch_losses) / max(1, len(epoch_losses)), "validation": metrics, "parse_coverage": coverage})
            atomic_json(output / f"validation_epoch_{epoch:03d}.json", predictions)
            if selected > best:
                best = selected
                save_checkpoint(output / "best.pt", model.projector, metadata, epoch=epoch, update=update, selection=metrics)
            if state_dict_hash(model.llm) != frozen_hash:
                raise ValueError("Frozen LLM weights changed")
            next_epoch = epoch + 1 if offset + accumulation >= len(order) else epoch
            next_offset = 0 if next_epoch != epoch else offset + accumulation
            save_checkpoint(output / "last.pt", model.projector, metadata, epoch=epoch, next_epoch=next_epoch, next_offset=next_offset, update=update, best=best, optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), rng=rng_state(), history=history, skipped_updates=skipped, exposure_ids=exposure_ids)
            atomic_json(output / "history.json", history, overwrite=True)
            logger.log("validation", epoch=epoch, metrics=metrics, n_QA_exposures=len(exposure_ids), exposure_hash=canonical_hash(exposure_ids))
            if update >= successful_budget:
                break
        load_checkpoint(output / "best.pt", model.projector, metadata)
        selected_predictions = generate_predictions(model, validation, config["generation"]["max_new_tokens"])
        write_jsonl(output / "predictions_val.jsonl", selected_predictions)
        write_jsonl(output / "qa_reference_val.jsonl", validation.records)
        selected_metrics = score_rows(validation.records, selected_predictions, config["selected_tasks"])
        atomic_json(output / "metric_recomputation.json", {"status": "verified", "kind": "qa", "predictions_sha256": file_sha256(output / "predictions_val.jsonl"), "qa_reference_path": "qa_reference_val.jsonl", "qa_reference_sha256": file_sha256(output / "qa_reference_val.jsonl"), "selected_tasks": config["selected_tasks"], "metrics": selected_metrics})
        diagnostic_records = []
        diagnostic_U, diagnostic_mask = [], []
        by_id = {row["qa_id"]: row for row in selected_predictions}
        for index in range(min(8, len(validation))):
            sample = validation[index]
            diagnostic_U.append(sample["U"].numpy())
            diagnostic_mask.append(sample["token_mask"].numpy())
            diagnostic_records.append({"qa_id": sample["qa_id"], "window_id": sample["window_id"], "prediction": by_id[sample["qa_id"]]})
        np.savez_compressed(output / "diagnostic_samples.npz", U=np.stack(diagnostic_U), token_mask=np.stack(diagnostic_mask))
        atomic_json(output / "diagnostic_samples.json", {"checkpoint_hash": file_sha256(output / "best.pt"), "records": diagnostic_records, "policy": "first_8_fixed_validation_QA"})
        logger.finish({"best_macro_f1": best[0], "selected_metrics": selected_metrics, "successful_updates": update, "n_QA_exposures": len(exposure_ids), "exposure_hash": canonical_hash(exposure_ids), "skipped_updates": skipped, "frozen_llm_hash": frozen_hash, "initial_projector_hash": initial["state_hash"], "precision": precision, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()})
    except Exception as error:
        logger.fail(error)
        raise


if __name__ == "__main__":
    main()
