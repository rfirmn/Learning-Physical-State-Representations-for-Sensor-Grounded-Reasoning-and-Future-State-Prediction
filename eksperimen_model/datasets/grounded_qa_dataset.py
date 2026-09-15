"""
grounded_qa_dataset.py
PyTorch Dataset and DataLoader for Stage 4 (Cognitive Alignment & SLM Inference).
Pairs 13 physical latent tokens with grounded question-answer instruction texts.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Optional, Tuple
from transformers import PreTrainedTokenizer


class GroundedQADataset(Dataset):
    """
    Multimodal Dataset for Stage 4:
    Loads pre-extracted physical latent sequences (13 tokens: 4 history + 1 current + 8 future)
    and pairs them with grounded natural language questions and supervised answers.
    """
    def __init__(
        self,
        qa_jsonl_path: str,
        features_dir: str,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        max_seq_len: int = 256,
        preload_features: bool = True
    ):
        self.qa_jsonl_path = qa_jsonl_path
        self.features_dir = features_dir
        self.tokenizer = tokenizer
        self.split = split
        self.max_seq_len = max_seq_len
        self.preload_features = preload_features

        # 1. Load QA records from JSONL
        if not os.path.exists(qa_jsonl_path):
            raise FileNotFoundError(f"QA JSONL file not found: {qa_jsonl_path}")

        print(f"[GroundedQADataset] Loading QA records from {qa_jsonl_path}...")
        self.records: List[Dict[str, Any]] = []
        with open(qa_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))
        print(f"[GroundedQADataset] Loaded {len(self.records)} records for split '{split.upper()}'.")

        # 2. Preload or Cache .pt feature files in RAM
        self.feature_cache: Dict[str, torch.Tensor] = {}
        self.split_features_dir = os.path.join(features_dir, split)
        if preload_features and os.path.exists(self.split_features_dir):
            print(f"[GroundedQADataset] Preloading .pt feature files from {self.split_features_dir} into RAM...")
            pt_files = [f for f in os.listdir(self.split_features_dir) if f.endswith(".pt")]
            for fname in pt_files:
                fpath = os.path.join(self.split_features_dir, fname)
                try:
                    data = torch.load(fpath, weights_only=True)
                    self.feature_cache[fname] = data["latent_z"] # (T, 384)
                except Exception as e:
                    print(f"  [Warning] Failed to preload {fname}: {e}")
            print(f"[GroundedQADataset] Preloaded {len(self.feature_cache)} feature files.")

        # Fixed prefix text
        self.system_prompt = (
            "<|im_start|>system\n"
            "You are a sensor-grounded physical reasoning assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "[Physical Observations]: "
        )

    def __len__(self) -> int:
        return len(self.records)

    def _get_feature_tensor(self, fname: str) -> torch.Tensor:
        if fname in self.feature_cache:
            return self.feature_cache[fname]
        fpath = os.path.join(self.split_features_dir, fname)
        data = torch.load(fpath, weights_only=True)
        z = data["latent_z"]
        if self.preload_features:
            self.feature_cache[fname] = z
        return z

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        fname = rec["feature_file"]
        w_start = rec["window_start"]

        # 1. Slice 13 Spatio-Temporal Physical Tokens
        # Window size is 24: frames 0..15 (history, 15 is current t), frames 16..23 (future t+1..t+8)
        latent_seq = self._get_feature_tensor(fname)
        T = len(latent_seq)

        if w_start + 24 <= T:
            window = latent_seq[w_start : w_start + 24] # (24, 384)
        else:
            # Boundary padding if sequence was edge-clipped
            window = latent_seq[max(0, T - 24) : T]
            if len(window) < 24:
                pad = window[-1:].repeat(24 - len(window), 1)
                window = torch.cat([window, pad], dim=0)

        # 4 History tokens: uniformly subsampled from 16 frames [0, 5, 10, 15]
        t_hist = window[[0, 5, 10, 15]] # (4, 384)
        # 1 Current token: frame 15
        t_curr = window[15:16]          # (1, 384)
        # 8 Future tokens: frames 16..23
        t_fut = window[16:24]           # (8, 384)

        # Concatenate 13 physical tokens: (13, 384)
        physical_tokens = torch.cat([t_hist, t_curr, t_fut], dim=0).float()

        # 2. Tokenize Text Prefix & Suffix
        question_text = rec["question"]
        target_text = rec["target_text"]

        # Suffix prompt: Question ending with assistant marker
        suffix_prompt = f"\nQuestion: {question_text}<|im_end|>\n<|im_start|>assistant\n"
        # Full suffix: Prompt + Response + EOS
        full_suffix = f"{suffix_prompt}{target_text}<|im_end|>"

        prefix_enc = self.tokenizer(self.system_prompt, add_special_tokens=False, return_tensors="pt")
        suffix_enc = self.tokenizer(full_suffix, add_special_tokens=False, return_tensors="pt")
        prompt_only_enc = self.tokenizer(suffix_prompt, add_special_tokens=False, return_tensors="pt")

        prefix_ids = prefix_enc["input_ids"].squeeze(0) # (L_pre,)
        suffix_ids = suffix_enc["input_ids"].squeeze(0) # (L_suf,)
        prompt_len = prompt_only_enc["input_ids"].squeeze(0).shape[0]

        # Truncate if suffix exceeds max length
        max_suf_len = self.max_seq_len - len(prefix_ids) - 13
        if len(suffix_ids) > max_suf_len:
            suffix_ids = suffix_ids[:max_suf_len]

        # 3. Construct Labels for Supervised Fine-Tuning
        # Total tokens = L_pre + 13 + L_suf
        # Supervision strictly on assistant response tokens (everything else masked with -100)
        tot_len = len(prefix_ids) + 13 + len(suffix_ids)
        labels = torch.full((tot_len,), -100, dtype=torch.long)

        # Assistant tokens start after (len(prefix_ids) + 13 + prompt_len)
        resp_start = len(prefix_ids) + 13 + prompt_len
        if resp_start < tot_len:
            labels[resp_start:] = suffix_ids[prompt_len:]

        return {
            "prefix_input_ids": prefix_ids,
            "suffix_input_ids": suffix_ids,
            "physical_tokens": physical_tokens,
            "labels": labels,
            "sample_id": rec["sample_id"],
            "task_name": rec["task_name"],
            "target_structured": json.dumps(rec["target_structured"])
        }


def collate_grounded_qa(batch: List[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """Custom collator with dynamic right-padding for prefix, suffix, and labels."""
    batch_size = len(batch)

    # 1. Physical tokens: (B, 13, 384)
    physical_tokens = torch.stack([item["physical_tokens"] for item in batch], dim=0)

    # 2. Pad prefix sequences
    prefix_list = [item["prefix_input_ids"] for item in batch]
    max_pre_len = max(len(t) for t in prefix_list)
    prefix_ids = torch.full((batch_size, max_pre_len), pad_token_id, dtype=torch.long)
    prefix_mask = torch.zeros((batch_size, max_pre_len), dtype=torch.long)
    for i, t in enumerate(prefix_list):
        prefix_ids[i, :len(t)] = t
        prefix_mask[i, :len(t)] = 1

    # 3. Pad suffix sequences
    suffix_list = [item["suffix_input_ids"] for item in batch]
    max_suf_len = max(len(t) for t in suffix_list)
    suffix_ids = torch.full((batch_size, max_suf_len), pad_token_id, dtype=torch.long)
    suffix_mask = torch.zeros((batch_size, max_suf_len), dtype=torch.long)
    for i, t in enumerate(suffix_list):
        suffix_ids[i, :len(t)] = t
        suffix_mask[i, :len(t)] = 1

    # 4. Pad labels (padded with -100)
    labels_list = [item["labels"] for item in batch]
    max_tot_len = max(len(l) for l in labels_list)
    labels = torch.full((batch_size, max_tot_len), -100, dtype=torch.long)
    for i, l in enumerate(labels_list):
        labels[i, :len(l)] = l

    return {
        "prefix_input_ids": prefix_ids,
        "prefix_attention_mask": prefix_mask,
        "physical_tokens": physical_tokens,
        "suffix_input_ids": suffix_ids,
        "suffix_attention_mask": suffix_mask,
        "labels": labels,
        "sample_ids": [item["sample_id"] for item in batch],
        "task_names": [item["task_name"] for item in batch],
        "target_structured": [item["target_structured"] for item in batch]
    }
