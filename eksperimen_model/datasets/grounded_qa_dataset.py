"""Stage 4 QA data with explicit physical-token conditions."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import yaml
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class GroundedQADataset(Dataset):
    """Build B3/B3P/B4/B5 inputs from one validated 16+8 window."""

    HISTORY = 16
    HORIZON = 8
    CONDITIONS = {"B3", "B3P", "B4", "B5"}
    SYSTEM_PROMPT = (
        "<|im_start|>system\n"
        "You are a sensor-grounded physical reasoning assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "[Physical Observations]: "
    )

    def __init__(
        self,
        qa_jsonl_path: str,
        features_dir: str,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        max_seq_len: int = 256,
        preload_features: bool = True,
        condition: str = "B4",
        dynamics_checkpoint: Optional[str] = None,
        dynamics_config: Optional[Union[str, Dict[str, Any]]] = None,
        dynamics_device: str = "cpu",
    ):
        self.condition = condition.upper()
        if self.condition not in self.CONDITIONS:
            raise ValueError(f"Unknown condition {condition!r}; expected {sorted(self.CONDITIONS)}")
        self.token_count = self.HISTORY + (self.HORIZON if self.condition != "B3" else 0)
        self.qa_jsonl_path = Path(qa_jsonl_path)
        self.features_dir = Path(features_dir)
        self.split = split
        self.split_features_dir = self.features_dir / split
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.preload_features = preload_features
        self.feature_cache: Dict[str, Dict[str, Any]] = {}
        self._token_cache: Dict[Tuple[str, int], torch.Tensor] = {}
        self.system_prompt = self.SYSTEM_PROMPT

        if not self.qa_jsonl_path.is_file():
            raise FileNotFoundError(f"QA JSONL file not found: {self.qa_jsonl_path}")
        if not self.split_features_dir.is_dir():
            raise FileNotFoundError(f"Feature split directory not found: {self.split_features_dir}")
        with self.qa_jsonl_path.open("r", encoding="utf-8") as handle:
            self.records: List[Dict[str, Any]] = [json.loads(line) for line in handle if line.strip()]
        if not self.records:
            raise ValueError(f"QA JSONL contains no records: {self.qa_jsonl_path}")

        self.provenance: Dict[str, Any] = {
            "condition": self.condition,
            "history_frames": self.HISTORY,
            "forecast_frames": self.HORIZON if self.condition != "B3" else 0,
            "physical_token_count": self.token_count,
            "qa_path": str(self.qa_jsonl_path.resolve()),
            "qa_sha256": _sha256(self.qa_jsonl_path),
            "features_dir": str(self.features_dir.resolve()),
            "feature_sha256": {},
            "split": split,
        }
        self.dynamics_model = None
        self.mean_z = None
        self.std_z = None
        self.expected_feature_provenance = None
        if self.condition == "B4":
            if dynamics_checkpoint is None:
                raise ValueError("B4 requires a frozen Stage 3 dynamics checkpoint")
            checkpoint_path = Path(dynamics_checkpoint)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Dynamics checkpoint not found: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, dict) or "config" not in checkpoint or "model_state_dict" not in checkpoint:
                raise ValueError("Dynamics checkpoint must contain config and model_state_dict")
            config = checkpoint["config"]
            temporal = config["temporal"]
            if temporal.get("t_in") != self.HISTORY or temporal.get("t_out") != self.HORIZON:
                raise ValueError("Dynamics checkpoint horizon must be 16 observed + 8 forecast states")
            self.expected_feature_provenance = checkpoint.get("feature_provenance")
            feature_manifest = checkpoint.get("train_feature_manifest_sha256")
            if (not isinstance(self.expected_feature_provenance, dict)
                    or not self.expected_feature_provenance
                    or not isinstance(feature_manifest, str) or not feature_manifest):
                raise ValueError(
                    "Dynamics checkpoint lacks verified feature lineage; retrain Stage 3 "
                    "or audit and migrate the original artifacts before B4"
                )
            if dynamics_config is not None:
                supplied = yaml.safe_load(Path(dynamics_config).read_text(encoding="utf-8")) if isinstance(dynamics_config, str) else dynamics_config
                if supplied.get("temporal") != temporal or supplied.get("dynamics_model") != config["dynamics_model"]:
                    raise ValueError("Dynamics config differs from checkpoint training config")
            from eksperimen_model.train_dynamics import build_dynamics_model

            self.dynamics_device = torch.device(dynamics_device)
            self.dynamics_model = build_dynamics_model(config["dynamics_model"], self.HISTORY, self.HORIZON)
            self.dynamics_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            self.dynamics_model.to(self.dynamics_device).eval()
            for parameter in self.dynamics_model.parameters():
                parameter.requires_grad_(False)
            self.provenance.update({
                "dynamics_checkpoint": str(checkpoint_path.resolve()),
                "dynamics_checkpoint_sha256": _sha256(checkpoint_path),
                "dynamics_epoch": checkpoint.get("epoch"),
                "dynamics_model_config": config["dynamics_model"],
                "dynamics_temporal_config": temporal,
                "train_feature_manifest_sha256": feature_manifest,
            })
            if temporal.get("normalize_z", False):
                stats_path = self.features_dir / "normalization_stats.pt"
                if not stats_path.is_file():
                    raise FileNotFoundError(f"Stage 3 normalization stats not found: {stats_path}")
                stats_digest = _sha256(stats_path)
                if checkpoint.get("normalization_stats_sha256") != stats_digest:
                    raise ValueError(
                        "Stage 3 normalization stats do not match the Dynamics checkpoint; "
                        "legacy checkpoints without a stats hash cannot be used for B4"
                    )
                stats = torch.load(stats_path, map_location="cpu", weights_only=True)
                if (stats.get("feature_provenance") != self.expected_feature_provenance
                        or stats.get("source_manifest_sha256") != feature_manifest):
                    raise ValueError("Stage 3 feature provenance or train manifest differs from normalization stats")
                self.mean_z = torch.as_tensor(stats["mean_z"], dtype=torch.float32)
                self.std_z = torch.as_tensor(stats["std_z"], dtype=torch.float32)
                dim = self.dynamics_model.embed_dim
                if (self.mean_z.shape != (dim,) or self.std_z.shape != (dim,)
                        or not torch.isfinite(self.mean_z).all()
                        or not torch.isfinite(self.std_z).all() or (self.std_z <= 0).any()):
                    raise ValueError("Invalid Stage 3 normalization statistics")
                self.provenance.update({
                    "normalization_stats": str(stats_path.resolve()),
                    "normalization_stats_sha256": stats_digest,
                })
        if preload_features:
            for name in sorted({record["feature_file"] for record in self.records}):
                self._load_feature(name)

    def __len__(self) -> int:
        return len(self.records)

    def _load_feature(self, name: str) -> Dict[str, Any]:
        if name in self.feature_cache:
            return self.feature_cache[name]
        if Path(name).name != name or not name.endswith(".pt"):
            raise ValueError(f"Invalid feature filename: {name!r}")
        path = self.split_features_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Feature file not found: {path}")
        data = torch.load(path, map_location="cpu", weights_only=True)
        z = data["latent_z"]
        frame_ids = data.get("source_frame_ids", data.get("frame_ids"))
        if not isinstance(z, torch.Tensor) or z.ndim != 2:
            raise ValueError(f"Invalid latent_z in {path}")
        if not isinstance(frame_ids, torch.Tensor) or frame_ids.ndim != 1 or len(frame_ids) != len(z):
            raise ValueError(f"Missing or invalid source frame IDs in {path}")
        if self.dynamics_model is not None and z.shape[1] != self.dynamics_model.embed_dim:
            raise ValueError(f"Latent dimension mismatches Dynamics checkpoint in {path}")
        feature_provenance = data.get("provenance")
        if (not isinstance(feature_provenance, dict)
                or not all(feature_provenance.get(key) is not None for key in (
                    "encoder_sha256", "config_sha256", "preprocessing", "sampling_seed"
                )) or not data.get("source_manifest_sha256")):
            raise ValueError(f"Feature lineage is missing in {path}")
        if self.expected_feature_provenance is not None and feature_provenance != self.expected_feature_provenance:
            raise ValueError(f"Feature lineage differs from Dynamics checkpoint: {path}")
        common_provenance = self.provenance.get("feature_provenance")
        if common_provenance is not None and feature_provenance != common_provenance:
            raise ValueError(f"Feature lineage differs within the {self.split} split: {path}")
        self.provenance["feature_provenance"] = feature_provenance
        if name not in self.provenance["feature_sha256"]:
            self.provenance["feature_sha256"][name] = _sha256(path)
        feature = {"latent_z": z, "source_frame_ids": frame_ids}
        if self.preload_features:
            self.feature_cache[name] = feature
        return feature

    def _window(self, record: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        name = record["feature_file"]
        start = record["window_start"]
        if type(start) is not int or start < 0:
            raise ValueError(f"Invalid window_start for {record.get('sample_id', name)}")
        data = self._load_feature(name)
        z = data["latent_z"]
        ids = data.get("source_frame_ids", data.get("frame_ids"))
        end = start + self.HISTORY + self.HORIZON
        if end > len(z):
            raise ValueError(f"Window exceeds feature file for {record.get('sample_id', name)}")
        frame_ids = ids[start:end]
        if not torch.all(frame_ids[1:] - frame_ids[:-1] == 1):
            raise ValueError(f"Non-contiguous source frames for {record.get('sample_id', name)}")
        if not all(key in record for key in ("source_frame_ids", "t_curr_frame", "t_future_frame")):
            raise ValueError(f"QA frame provenance missing for {record.get('sample_id', name)}")
        if frame_ids.tolist() != record["source_frame_ids"]:
            raise ValueError(f"QA frame IDs differ from features for {record.get('sample_id', name)}")
        if int(frame_ids[self.HISTORY - 1]) != record["t_curr_frame"]:
            raise ValueError(f"QA current frame differs from features for {record.get('sample_id', name)}")
        if int(frame_ids[-1]) != record["t_future_frame"]:
            raise ValueError(f"QA future frame differs from features for {record.get('sample_id', name)}")
        latent_end = end if self.condition == "B5" else start + self.HISTORY
        window = z[start:latent_end].float()
        if not torch.isfinite(window).all():
            raise ValueError(f"Non-finite observed latent state for {record.get('sample_id', name)}")
        return window, frame_ids

    def get_physical_tokens(self, record: Dict[str, Any]) -> torch.Tensor:
        """Return raw-scale tokens; B4 never reads future latent values."""
        key = (record["feature_file"], record["window_start"])
        if key in self._token_cache:
            return self._token_cache[key].clone()
        window, _ = self._window(record)
        history = window[:self.HISTORY].clone()
        if self.condition == "B3":
            tokens = history
        elif self.condition == "B3P":
            tokens = torch.cat((history, history[-1:].repeat(self.HORIZON, 1)), dim=0)
        elif self.condition == "B5":
            tokens = window.clone()
        else:
            input_z = history.to(self.dynamics_device)
            if self.mean_z is not None:
                input_z = (input_z - self.mean_z.to(self.dynamics_device)) / self.std_z.to(self.dynamics_device)
            with torch.inference_mode():
                future = self.dynamics_model(input_z.unsqueeze(0)).squeeze(0)
            if future.shape != (self.HORIZON, history.shape[1]):
                raise ValueError("Dynamics returned an invalid forecast shape")
            if self.mean_z is not None:
                future = future * self.std_z.to(self.dynamics_device) + self.mean_z.to(self.dynamics_device)
            tokens = torch.cat((history, future.cpu().float()), dim=0)
        if tokens.shape[0] != self.token_count or not torch.isfinite(tokens).all():
            raise ValueError("Invalid physical tokens")
        if len(self._token_cache) >= 128:
            self._token_cache.clear()
        self._token_cache[key] = tokens
        return tokens.clone()

    def encode_prompt(self, question: str, target_text: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """One tokenizer path for training and generation prompts."""
        suffix = f"\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n"
        prefix_ids = self.tokenizer(self.system_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
        prompt_ids = self.tokenizer(suffix, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
        if target_text is None:
            return prefix_ids, prompt_ids, len(prompt_ids)
        answer_ids = self.tokenizer(target_text + "<|im_end|>", add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
        return prefix_ids, torch.cat((prompt_ids, answer_ids)), len(prompt_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        tokens = self.get_physical_tokens(record)
        prefix_ids, suffix_ids, prompt_len = self.encode_prompt(record["question"], record["target_text"])
        total = len(prefix_ids) + len(tokens) + len(suffix_ids)
        if total > self.max_seq_len:
            raise ValueError(f"QA sample {record['sample_id']} exceeds max_seq_len={self.max_seq_len}")
        labels = torch.full((total,), -100, dtype=torch.long)
        labels[len(prefix_ids) + len(tokens) + prompt_len:] = suffix_ids[prompt_len:]
        if not (labels != -100).any():
            raise ValueError(f"QA sample {record['sample_id']} has no supervised answer tokens")
        return {
            "prefix_input_ids": prefix_ids,
            "suffix_input_ids": suffix_ids,
            "physical_tokens": tokens,
            "labels": labels,
            "sample_id": record["sample_id"],
            "task_name": record["task_name"],
            "target_structured": json.dumps(record["target_structured"], sort_keys=True),
            "condition": self.condition,
        }


def collate_grounded_qa(batch: List[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, Any]:
    """Right-pad text and labels; physical length is fixed within a condition."""
    if not batch:
        raise ValueError("Cannot collate an empty QA batch")
    physical_tokens = torch.stack([item["physical_tokens"] for item in batch])
    batch_size = len(batch)

    def pad_text(key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        tensors = [item[key] for item in batch]
        width = max(map(len, tensors))
        ids = torch.full((batch_size, width), pad_token_id, dtype=torch.long)
        mask = torch.zeros((batch_size, width), dtype=torch.long)
        for row, tensor in enumerate(tensors):
            ids[row, :len(tensor)] = tensor
            mask[row, :len(tensor)] = 1
        return ids, mask

    prefix_ids, prefix_mask = pad_text("prefix_input_ids")
    suffix_ids, suffix_mask = pad_text("suffix_input_ids")
    width = max(len(item["labels"]) for item in batch)
    labels = torch.full((batch_size, width), -100, dtype=torch.long)
    for row, item in enumerate(batch):
        labels[row, :len(item["labels"])] = item["labels"]
    return {
        "prefix_input_ids": prefix_ids,
        "prefix_attention_mask": prefix_mask,
        "physical_tokens": physical_tokens,
        "suffix_input_ids": suffix_ids,
        "suffix_attention_mask": suffix_mask,
        "labels": labels,
        "sample_ids": [item["sample_id"] for item in batch],
        "task_names": [item["task_name"] for item in batch],
        "target_structured": [item["target_structured"] for item in batch],
        "conditions": [item["condition"] for item in batch],
    }
