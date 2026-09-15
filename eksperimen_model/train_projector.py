"""
train_projector.py
Stage 4 Training Script: Two-Layer MLP Projector Alignment with Frozen Qwen2.5-1.5B-Instruct.

Optimizes cross-modal adapter parameters (~1.97M) using Cross-Entropy loss on assistant response tokens,
while the base SLM parameters (~1.54B) remain strictly frozen.
Includes gradient checkpointing, micro-batch accumulation, and comprehensive loss tracking.
"""

import os
import sys
import yaml
import json
import time
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

# Fix Windows terminal UTF-8 encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.abspath("."))
from eksperimen_model.models.projector import PhysicalSLMWrapper
from eksperimen_model.datasets.grounded_qa_dataset import GroundedQADataset, collate_grounded_qa


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 4: MLP Projector Alignment Training")
    parser.add_argument("--config", type=str, default="eksperimen_model/configs/mmfi_projector_qwen.yaml",
                        help="Path to YAML configuration file")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs")
    parser.add_argument("--micro_batch_size", type=int, default=None,
                        help="Override micro-batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=None,
                        help="Override gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Limit train samples for fast iteration (e.g., 10000)")
    parser.add_argument("--max_val_samples", type=int, default=1000,
                        help="Limit validation samples for fast evaluation")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load Configuration
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Overrides
    epochs = args.epochs or cfg["training"].get("epochs", 5)
    micro_bs = args.micro_batch_size or cfg["training"].get("micro_batch_size", 2)
    grad_accum = args.grad_accum_steps or cfg["training"].get("gradient_accumulation_steps", 8)
    lr = args.lr or cfg["training"].get("learning_rate", 1.0e-4)
    output_dir = args.output_dir or cfg["training"].get("output_dir", "eksperimen_model/checkpoints/projector")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 75)
    print(" STAGE 4: TWO-LAYER MLP PROJECTOR TRAINING (QWEN2.5-1.5B) ")
    print("=" * 75)
    print(f"[*] Configuration: {args.config}")
    print(f"[*] Base SLM: {cfg['model']['base_model']} (Strictly Frozen)")
    print(f"[*] Epochs: {epochs} | Micro Batch: {micro_bs} | Grad Accum: {grad_accum} (Effective Batch = {micro_bs * grad_accum})")
    print(f"[*] Learning Rate: {lr} (Cosine Annealing with Warmup)")
    print(f"[*] Output Directory: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    # 2. Initialize Model Wrapper
    print("\n[1/5] Loading Physical-SLM Wrapper...")
    wrapper = PhysicalSLMWrapper(
        model_name_or_path=cfg["model"]["base_model"],
        in_dim=cfg["model"]["in_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        freeze_slm=cfg["model"]["freeze_slm"],
        use_gradient_checkpointing=cfg["model"]["use_gradient_checkpointing"],
        torch_dtype=dtype,
        device_map="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 3. Load Grounded-QA Datasets
    print("\n[2/5] Preparing Datasets & DataLoaders...")
    train_dataset = GroundedQADataset(
        qa_jsonl_path=cfg["dataset"]["train_file"],
        features_dir=cfg["dataset"]["features_dir"],
        tokenizer=wrapper.tokenizer,
        split="train",
        max_seq_len=cfg["dataset"].get("max_seq_len", 256),
        preload_features=True
    )

    val_dataset = GroundedQADataset(
        qa_jsonl_path=cfg["dataset"]["val_file"],
        features_dir=cfg["dataset"]["features_dir"],
        tokenizer=wrapper.tokenizer,
        split="val",
        max_seq_len=cfg["dataset"].get("max_seq_len", 256),
        preload_features=True
    )

    # Subsample if requested for fast iteration
    if args.max_train_samples and args.max_train_samples < len(train_dataset.records):
        print(f"[*] Subsampling train dataset to {args.max_train_samples} samples (fast iteration mode)...")
        train_dataset.records = train_dataset.records[:args.max_train_samples]

    if args.max_val_samples and args.max_val_samples < len(val_dataset.records):
        print(f"[*] Subsampling validation dataset to {args.max_val_samples} samples...")
        val_dataset.records = val_dataset.records[:args.max_val_samples]

    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_bs,
        shuffle=True,
        collate_fn=lambda b: collate_grounded_qa(b, pad_token_id=wrapper.tokenizer.pad_token_id),
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=micro_bs,
        shuffle=False,
        collate_fn=lambda b: collate_grounded_qa(b, pad_token_id=wrapper.tokenizer.pad_token_id),
        num_workers=0
    )

    # 4. Optimizer & LR Scheduler
    print("\n[3/5] Setting up AdamW Optimizer strictly on Projector Parameters...")
    trainable_params = [p for p in wrapper.projector.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=cfg["training"].get("weight_decay", 0.01)
    )

    total_update_steps = (len(train_loader) // grad_accum) * epochs
    warmup_steps = int(total_update_steps * cfg["training"].get("warmup_ratio", 0.05))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps
    )
    print(f"[+] Total Update Steps: {total_update_steps} (Warmup Steps: {warmup_steps})")

    # 5. Training Loop
    print("\n[4/5] Beginning Projector Alignment Training...")
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "lr": [], "epoch_time_s": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        wrapper.train()
        wrapper.llm.eval() # Keep frozen SLM dropout/norms in eval mode

        running_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        step_in_epoch = 0

        for step, batch in enumerate(pbar):
            step_in_epoch += 1
            prefix_ids = batch["prefix_input_ids"].to(device)
            prefix_mask = batch["prefix_attention_mask"].to(device)
            phys_tokens = batch["physical_tokens"].to(device)
            suffix_ids = batch["suffix_input_ids"].to(device)
            suffix_mask = batch["suffix_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss, _ = wrapper(
                prefix_input_ids=prefix_ids,
                prefix_attention_mask=prefix_mask,
                physical_tokens=phys_tokens,
                suffix_input_ids=suffix_ids,
                suffix_attention_mask=suffix_mask,
                labels=labels
            )

            # Scale loss for gradient accumulation
            loss_scaled = loss / grad_accum
            loss_scaled.backward()
            running_loss += loss.item()

            if step_in_epoch % grad_accum == 0 or step_in_epoch == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg["training"].get("max_grad_norm", 1.0))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            current_lr = scheduler.get_last_lr()[0]
            current_vram = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{running_loss / (step + 1):.4f}",
                "lr": f"{current_lr:.2e}",
                "vram": f"{current_vram:.1f}GB"
            })

        epoch_train_loss = running_loss / len(train_loader)
        epoch_time = time.time() - t0

        # Validation Phase
        wrapper.eval()
        val_running_loss = 0.0
        print(f"[*] Running Validation for Epoch {epoch}...")
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                prefix_ids = batch["prefix_input_ids"].to(device)
                prefix_mask = batch["prefix_attention_mask"].to(device)
                phys_tokens = batch["physical_tokens"].to(device)
                suffix_ids = batch["suffix_input_ids"].to(device)
                suffix_mask = batch["suffix_attention_mask"].to(device)
                labels = batch["labels"].to(device)

                loss, _ = wrapper(
                    prefix_input_ids=prefix_ids,
                    prefix_attention_mask=prefix_mask,
                    physical_tokens=phys_tokens,
                    suffix_input_ids=suffix_ids,
                    suffix_attention_mask=suffix_mask,
                    labels=labels
                )
                val_running_loss += loss.item()

        epoch_val_loss = val_running_loss / len(val_loader)
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["lr"].append(current_lr)
        history["epoch_time_s"].append(epoch_time)

        print(f"\n[Epoch {epoch}/{epochs} Summary] Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Duration: {epoch_time:.1f}s")

        # Save Latest Checkpoint (only projector weights ~4MB)
        latest_ckpt = os.path.join(output_dir, "latest_projector.pth")
        torch.save({
            "epoch": epoch,
            "projector_state_dict": wrapper.projector.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "config": cfg
        }, latest_ckpt)

        # Save Best Checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_ckpt = os.path.join(output_dir, "best_projector.pth")
            torch.save({
                "epoch": epoch,
                "projector_state_dict": wrapper.projector.state_dict(),
                "val_loss": epoch_val_loss,
                "config": cfg
            }, best_ckpt)
            print(f"[+] NEW BEST MODEL SAVED! (Val Loss: {best_val_loss:.4f}) -> {best_ckpt}")

    # 6. Save Training History & Plot Loss Curve
    print("\n[5/5] Saving Training History and Diagnostic Curves...")
    history_file = os.path.join(output_dir, "projector_training_history.json")
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_file = os.path.join(output_dir, "projector_loss_curve.png")
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, epochs + 1), history["train_loss"], marker="o", label="Train Cross-Entropy Loss", color="#1f77b4", linewidth=2.0)
    plt.plot(range(1, epochs + 1), history["val_loss"], marker="s", label="Val Cross-Entropy Loss", color="#d62728", linewidth=2.0)
    plt.title("Stage 4: MLP Projector Alignment Loss Curve", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Cross-Entropy Loss", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(plot_file, dpi=200)
    plt.close()

    print(f"[+] Loss curve saved to: {plot_file}")
    print(f"[+] History saved to: {history_file}")
    print("\n" + "=" * 75)
    print(" STAGE 4 MLP PROJECTOR TRAINING COMPLETE! ")
    print(f" Best Validation Loss: {best_val_loss:.4f} ")
    print(f" Best Checkpoint: {os.path.join(output_dir, 'best_projector.pth')} ")
    print("=" * 75)


if __name__ == "__main__":
    main()
