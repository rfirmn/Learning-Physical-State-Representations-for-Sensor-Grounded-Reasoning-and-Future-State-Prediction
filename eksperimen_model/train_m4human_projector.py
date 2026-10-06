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
from eksperimen_model.utils.m4human_performance import configure_runtime, finish_optimizer_step, ResourceMonitor
from eksperimen_model.utils.m4human_runtime import (RunLogger, adamw_parameters, atomic_json, canonical_hash,
    file_sha256, load_checkpoint, save_checkpoint, seed_everything, state_dict_hash, rng_state, restore_rng_state)


def write_jsonl(path, rows):
    path = Path(path)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def generate_predictions(model, dataset, max_new_tokens, batch_size=1, resource_monitor=None):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('generation batch_size must be positive')
    records = []
    for start in range(0, len(dataset), batch_size):
        if resource_monitor is not None:
            resource_monitor.check()
        indices = list(range(start, min(start + batch_size, len(dataset))))
        group = [{key: dataset.records[index][key] for key in ("qa_id", "window_id", "subject_id", "recording_id", "task")} for index in indices]
        for record in group:
            record.update(condition=dataset.metadata["condition"], K=dataset.metadata["K"])
        try:
            samples = [dataset[index] for index in indices]
            result, _ = model.generate(samples, max_new_tokens=max_new_tokens)
            if len(result) != len(group):
                raise ValueError('generation returned incorrect batch count')
            for record, sample, prediction in zip(group, samples, result):
                record.update(prediction, error=None, n_valid_tokens=int(sample['token_mask'].sum()))
        except (ValueError, RuntimeError) as error:
            # One invalid sample must not turn its valid batch peers into failures.
            if isinstance(error,torch.cuda.OutOfMemoryError):
                # Release failed-generation activations before trying smaller batches.
                error.__traceback__ = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            for index, record in zip(indices, group):
                try:
                    sample = dataset[index]
                    result, _ = model.generate([sample], max_new_tokens=max_new_tokens)
                    record.update(result[0], error=None, n_valid_tokens=int(sample['token_mask'].sum()))
                except (ValueError, RuntimeError) as individual_error:
                    record.update(raw_ids=[], raw_text="", error=str(individual_error), n_valid_tokens=None)
        if resource_monitor is not None:
            resource_monitor.check()
        for record in group:
            record.update(parse_response(record['raw_text'], record['task']))
        records.extend(group)
    return records


def _projector_logical_step(model, samples, settings, optimizer, scaler, device):
    """Each QA has equal weight; AMP retries replay the same samples and RNG."""
    retries = settings.get('amp_max_retries', 8)
    if type(retries) is not int or retries < 0:
        raise ValueError('amp_max_retries must be a nonnegative integer')
    replay_rng = rng_state() if scaler.is_enabled() else None
    for attempt in range(retries + 1):
        if attempt:
            restore_rng_state(replay_rng)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.
        for start in range(0, len(samples), settings['micro_batch']):
            micro = samples[start:start + settings['micro_batch']]
            with torch.autocast(torch.device(device).type, dtype=torch.float16 if settings['precision'] == 'fp16_amp_grad_scaler' else torch.bfloat16, enabled=settings['precision'] != 'fp32'):
                loss = model(micro)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite answer-only CE')
            scaler.scale(loss * len(micro) / len(samples)).backward()
            loss_sum += float(loss.detach()) * len(micro)
        norm, successful = finish_optimizer_step(optimizer, scaler, model.projector.parameters(), settings['clip_grad_norm'])
        if successful:
            return loss_sum / len(samples), norm, attempt
    raise ValueError('AMP overflow exceeded same-QA retry budget; use verified stable precision')


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
    logger = None
    try:
        with ResourceMonitor(config['training'], args.device) as monitor:
            if any(value is None for value in config["lineage"].values()):
                raise ValueError("Resolve all audited lineage fields before alignment")
            seed_everything(config["training"]["seed"])
            settings = config["training"]
            configure_runtime(settings, args.device)
            if settings["max_successful_updates"] < 1 or settings["epochs_cap"] < 1:
                raise ValueError("Positive training budget required")
            model = load_frozen_qwen(config["llm"], args.device, settings["precision"])
            checkpoint_language = settings.get('activation_checkpointing', False)
            if type(checkpoint_language) is not bool:
                raise ValueError('activation_checkpointing must be boolean')
            model.checkpoint_language = checkpoint_language
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
            monitor.check()
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
                if start_offset < 0 or start_offset >= len(train) or start_offset % (settings["micro_batch"] * settings["accumulation"]):
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
            for epoch in range(start_epoch, settings["epochs_cap"]):
                model.train()
                # Same order independent of condition, epoch-boundary resume exact.
                order = torch.randperm(len(train), generator=torch.Generator().manual_seed(settings["seed"] + epoch)).tolist()
                accumulation = settings["micro_batch"] * settings["accumulation"]
                epoch_losses = []
                for offset in range(start_offset if epoch == start_epoch else 0, len(order), accumulation):
                    monitor.check()
                    group = order[offset:offset + accumulation]
                    started = time.perf_counter()
                    factor = (update + 1) / warmup if update < warmup else .5 * (1 + math.cos(math.pi * (update - warmup) / max(1, successful_budget - warmup)))
                    for optimizer_group in optimizer.param_groups:
                        optimizer_group["lr"] = settings["learning_rate"] * factor
                    group_loss, gradient, overflow_retries = _projector_logical_step(model, [train[index] for index in group], settings, optimizer, scaler, args.device)
                    monitor.check()
                    skipped += overflow_retries
                    if overflow_retries:
                        logger.log('overflow_retry', epoch=epoch, offset=offset, attempts=overflow_retries,
                                   scale=scaler.get_scale(), policy='same_QA_batch_rng_replay')
                    update += 1
                    exposure_ids.extend(train.records[index]["qa_id"] for index in group)
                    epoch_losses.extend([group_loss] * len(group))
                    logger.log("update", epoch=epoch, update=update, answer_only_ce=group_loss, gradient_norm=float(gradient), lr=optimizer.param_groups[0]["lr"], skipped_updates=skipped, seconds=time.perf_counter() - started, n_QA=len(group), exposure_hash=canonical_hash(exposure_ids))
                    if any(parameter.grad is not None for parameter in model.llm.parameters()):
                        raise ValueError("Frozen LLM acquired parameter gradients")
                    if update >= successful_budget:
                        break
                monitor.check()
                predictions = generate_predictions(model, validation, config["generation"]["max_new_tokens"], config['generation'].get('batch_size', 1), resource_monitor=monitor)
                monitor.check()
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
            selected_predictions = generate_predictions(model, validation, config["generation"]["max_new_tokens"], config['generation'].get('batch_size', 1), resource_monitor=monitor)
            write_jsonl(output / "predictions_val.jsonl", selected_predictions)
            write_jsonl(output / "qa_reference_val.jsonl", validation.records)
            monitor.check()
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
            summary = {"best_macro_f1": best[0], "selected_metrics": selected_metrics, "successful_updates": update, "n_QA_exposures": len(exposure_ids), "exposure_hash": canonical_hash(exposure_ids), "skipped_updates": skipped, "frozen_llm_hash": frozen_hash, "initial_projector_hash": initial["state_hash"], "precision": precision, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None, "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None}
        summary['resource_safety'] = monitor.snapshot()
        logger.finish(summary)
    except Exception as error:
        if logger is not None and not logger.closed:
            logger.fail(error)
        raise


if __name__ == "__main__":
    main()
