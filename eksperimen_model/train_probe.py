"""B2 probes for the two validated relative wrist-separation QA tasks."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = {
    "current_wrist_separation": ("narrower", "wider"),
    "future_wrist_separation_change": ("closing", "stable", "opening"),
}
TASK_NAMES = tuple(CLASS_NAMES)
HISTORY = 16
HORIZON = 8


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ProbeDataset(Dataset):
    """Read only the 16 observed latents; validate target and frame provenance."""

    def __init__(self, jsonl_path, features_dir, split):
        self.jsonl_path = Path(jsonl_path)
        self.split_dir = Path(features_dir) / split
        if not self.jsonl_path.is_file():
            raise FileNotFoundError(self.jsonl_path)
        if not self.split_dir.is_dir():
            raise FileNotFoundError(self.split_dir)
        with self.jsonl_path.open(encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        if not self.records:
            raise ValueError(f"No QA records in {self.jsonl_path}")

        self.features = {}
        self.provenance = {
            "split": split,
            "qa_path": str(self.jsonl_path.resolve()),
            "qa_sha256": sha256(self.jsonl_path),
            "features_dir": str(self.split_dir.resolve()),
            "feature_provenance": None,
            "feature_sha256": {},
        }
        seen_ids = set()
        counts = Counter()
        for record in self.records:
            sample_id = record["sample_id"]
            if sample_id in seen_ids or not sample_id.startswith(f"{split}_"):
                raise ValueError(f"Duplicate or mismatched sample ID: {sample_id}")
            seen_ids.add(sample_id)
            task = record["task_name"]
            target = record["target_structured"]
            if task not in CLASS_NAMES or not isinstance(target, dict) or set(target) != {task}:
                raise ValueError(f"Unsupported or mismatched target for {sample_id}")
            if target[task] not in CLASS_NAMES[task]:
                raise ValueError(f"Unknown target label for {sample_id}: {target[task]}")
            if json.loads(record["target_text"]) != target:
                raise ValueError(f"Target text differs from structured target for {sample_id}")
            counts[task] += 1
            self._load_feature(record["feature_file"])
        if any(counts[task] == 0 for task in TASK_NAMES):
            raise ValueError(f"Both QA tasks are required in {split}: {dict(counts)}")

    def _load_feature(self, name):
        if name in self.features:
            return self.features[name]
        if Path(name).name != name or not name.endswith(".pt"):
            raise ValueError(f"Invalid feature filename: {name}")
        path = self.split_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        data = torch.load(path, map_location="cpu", weights_only=True)
        latent = data["latent_z"]
        frame_ids = data.get("source_frame_ids")
        if (not isinstance(latent, torch.Tensor) or latent.ndim != 2
                or latent.shape[1] != 384 or not isinstance(frame_ids, torch.Tensor)
                or frame_ids.ndim != 1 or len(frame_ids) != len(latent)):
            raise ValueError(f"Invalid feature shape or source frame IDs: {path}")
        lineage = data.get("provenance")
        if (not isinstance(lineage, dict)
                or any(lineage.get(key) is None for key in (
                    "encoder_sha256", "config_sha256", "preprocessing", "sampling_seed"
                )) or not data.get("source_manifest_sha256")):
            raise ValueError(f"Missing feature extraction lineage: {path}")
        if self.provenance["feature_provenance"] is None:
            self.provenance["feature_provenance"] = lineage
        elif self.provenance["feature_provenance"] != lineage:
            raise ValueError(f"Mixed feature extraction lineage in {self.split_dir}: {path}")
        self.features[name] = (latent, frame_ids)
        self.provenance["feature_sha256"][name] = sha256(path)
        return self.features[name]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        latent, frame_ids = self._load_feature(record["feature_file"])
        start = record["window_start"]
        sample_id = record["sample_id"]
        end = start + HISTORY + HORIZON
        if type(start) is not int or start < 0 or end > len(latent):
            raise ValueError(f"Invalid 16+8 window for {sample_id}")
        ids = frame_ids[start:end]
        if not torch.all(ids[1:] - ids[:-1] == 1):
            raise ValueError(f"Non-contiguous source frames for {sample_id}")
        if ids.tolist() != record["source_frame_ids"]:
            raise ValueError(f"QA frame IDs differ from features for {sample_id}")
        if int(ids[HISTORY - 1]) != record["t_curr_frame"] or int(ids[-1]) != record["t_future_frame"]:
            raise ValueError(f"QA time anchors differ from features for {sample_id}")
        history = latent[start:start + HISTORY].float()
        if not torch.isfinite(history).all():
            raise ValueError(f"Non-finite observed state for {sample_id}")
        task = record["task_name"]
        label = CLASS_NAMES[task].index(record["target_structured"][task])
        return {
            "history": history,
            "task_id": TASK_NAMES.index(task),
            "label": label,
            "sample_id": sample_id,
        }


class MultiTaskPhysicalProbe(nn.Module):
    """A linear z_t probe and a separate linear 16-state history probe."""

    def __init__(self, in_dim=384, history_frames=HISTORY):
        super().__init__()
        self.in_dim = in_dim
        self.history_frames = history_frames
        self.current_head = nn.Linear(in_dim, len(CLASS_NAMES[TASK_NAMES[0]]))
        self.future_head = nn.Linear(in_dim * history_frames, len(CLASS_NAMES[TASK_NAMES[1]]))

    def forward(self, history):
        if history.ndim != 3 or history.shape[1:] != (self.history_frames, self.in_dim):
            raise ValueError("Probe expects [batch, 16, 384] observed states")
        return {
            TASK_NAMES[0]: self.current_head(history[:, -1]),
            TASK_NAMES[1]: self.future_head(history.flatten(1)),
        }


def batch_loss(logits, task_ids, labels):
    """Supervise only the task asked by each QA record."""
    total = 0
    for task_id, task in enumerate(TASK_NAMES):
        mask = task_ids == task_id
        if mask.any():
            total = total + F.cross_entropy(logits[task][mask], labels[mask], reduction="sum")
    return total / len(labels)


def score_model(model, loader, device):
    model.eval()
    truths = {task: [] for task in TASK_NAMES}
    predictions = {task: [] for task in TASK_NAMES}
    loss_sum = 0.0
    sample_count = 0
    with torch.inference_mode():
        for batch in loader:
            history = batch["history"].to(device)
            task_ids = batch["task_id"].to(device)
            labels = batch["label"].to(device)
            logits = model(history)
            loss_sum += batch_loss(logits, task_ids, labels).item() * len(labels)
            sample_count += len(labels)
            for task_id, task in enumerate(TASK_NAMES):
                mask = task_ids == task_id
                truths[task].extend(labels[mask].tolist())
                predictions[task].extend(logits[task][mask].argmax(-1).tolist())

    metrics = {"count": sample_count, "cross_entropy": loss_sum / sample_count, "tasks": {}}
    for task in TASK_NAMES:
        y_true, y_pred = truths[task], predictions[task]
        count = len(y_true)
        if count == 0:
            raise ValueError(f"No evaluation examples for {task}")
        per_class_f1 = {}
        for class_id, label in enumerate(CLASS_NAMES[task]):
            tp = sum(y == class_id and p == class_id for y, p in zip(y_true, y_pred))
            fp = sum(y != class_id and p == class_id for y, p in zip(y_true, y_pred))
            fn = sum(y == class_id and p != class_id for y, p in zip(y_true, y_pred))
            per_class_f1[label] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        metrics["tasks"][task] = {
            "count": count,
            "accuracy": sum(y == p for y, p in zip(y_true, y_pred)) / count,
            "macro_f1": sum(per_class_f1.values()) / len(per_class_f1),
            "class_counts": {label: y_true.count(i) for i, label in enumerate(CLASS_NAMES[task])},
            "predicted_counts": {label: y_pred.count(i) for i, label in enumerate(CLASS_NAMES[task])},
            "per_class_f1": per_class_f1,
        }
    metrics["mean_macro_f1"] = sum(metrics["tasks"][task]["macro_f1"] for task in TASK_NAMES) / len(TASK_NAMES)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train B2 direct probes on validated Stage 4 QA")
    parser.add_argument("--qa_dir", default="datasets/MM-Fi_grounded_qa_stage4_recovery")
    parser.add_argument("--features_dir", default="datasets/MM-Fi_features_stage4_recovery")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", default="eksperimen_model/checkpoints/probe_stage4_recovery")
    parser.add_argument("--evaluate_test", action="store_true",
                        help="Explicitly evaluate the frozen best probe on held-out test QA")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError("epochs, batch_size, and lr must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    qa_dir = Path(args.qa_dir)
    train_ds = ProbeDataset(qa_dir / "mmfi_grounded_qa_train.jsonl", args.features_dir, "train")
    val_ds = ProbeDataset(qa_dir / "mmfi_grounded_qa_val.jsonl", args.features_dir, "val")
    if train_ds.provenance["feature_provenance"] != val_ds.provenance["feature_provenance"]:
        raise ValueError("Train and validation feature extraction lineage differs")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = MultiTaskPhysicalProbe().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_probe_model.pth"
    best_score = -1.0
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch in train_loader:
            history = batch["history"].to(device)
            task_ids = batch["task_id"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model(history), task_ids, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
            train_count += len(labels)
        val_metrics = score_model(model, val_loader, device)
        score = val_metrics["mean_macro_f1"]
        val_loss = val_metrics["cross_entropy"]
        if score > best_score or (score == best_score and val_loss < best_loss):
            best_score, best_loss = score, val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": {"in_dim": 384, "history_frames": HISTORY},
                "class_names": CLASS_NAMES,
                "task_names": TASK_NAMES,
                "epoch": epoch,
                "selection_metric": "mean_macro_f1",
                "validation_metrics": val_metrics,
                "seed": args.seed,
                "training": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr},
                "provenance": {"train": train_ds.provenance, "val": val_ds.provenance},
            }, checkpoint_path)
        print(f"Epoch {epoch:02d}: train CE={train_loss / train_count:.4f}, "
              f"val CE={val_loss:.4f}, val mean macro-F1={score:.4f}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    report = {
        "baseline": "B2_Direct_Probe",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "selected_epoch": checkpoint["epoch"],
        "validation": checkpoint["validation_metrics"],
        "provenance": checkpoint["provenance"],
    }
    if args.evaluate_test:
        test_ds = ProbeDataset(qa_dir / "mmfi_grounded_qa_test.jsonl", args.features_dir, "test")
        if test_ds.provenance["feature_provenance"] != train_ds.provenance["feature_provenance"]:
            raise ValueError("Test feature extraction lineage differs from the B2 training checkpoint")
        report["test"] = score_model(model, DataLoader(test_ds, batch_size=args.batch_size), device)
        report["provenance"]["test"] = test_ds.provenance
    report_path = output_dir / "probe_benchmark_metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved B2 checkpoint: {checkpoint_path}")
    print(f"Saved B2 metrics: {report_path}")


if __name__ == "__main__":
    main()
