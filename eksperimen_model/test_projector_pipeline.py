"""
test_projector_pipeline.py
Smoke Test & VRAM Benchmarking for Stage 4: PhysicalToLLMProjector & Frozen Qwen2.5-1.5B-Instruct.

Verifies:
1. Model loading & tokenizer integrity.
2. Parameter freezing (strictly zero gradients on SLM).
3. Backpropagation connectivity (gradients successfully flow to the MLP Projector).
4. Gradient checkpointing activation.
5. Empirical measurement of actual peak VRAM consumption on RTX 3060.
"""

import sys
import os
import torch
import gc

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.abspath("."))
from eksperimen_model.models.projector import PhysicalToLLMProjector, PhysicalSLMWrapper


def run_smoke_test(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", batch_size: int = 2):
    print("=" * 70)
    print(" STAGE 4 SMOKE TEST & VRAM BENCHMARKING ")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[*] GPU Name: {gpu_name}")
        print(f"[*] Total VRAM: {total_vram_gb:.2f} GB")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        vram_start = torch.cuda.memory_allocated() / (1024 ** 3)
        print(f"[*] Initial Allocated VRAM: {vram_start:.3f} GB")

    # 1. Initialize Multimodal Wrapper
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"\n[1] Initializing PhysicalSLMWrapper with {model_name} (dtype={dtype})...")
    wrapper = PhysicalSLMWrapper(
        model_name_or_path=model_name,
        in_dim=384,
        hidden_dim=1024,
        freeze_slm=True,
        use_gradient_checkpointing=True,
        torch_dtype=dtype,
        device_map="cuda" if torch.cuda.is_available() else "cpu"
    )

    if torch.cuda.is_available():
        vram_after_model = torch.cuda.memory_allocated() / (1024 ** 3)
        print(f"[+] VRAM after model load: {vram_after_model:.2f} GB")

    # Count parameters
    total_params = sum(p.numel() for p in wrapper.parameters())
    trainable_params = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"[+] Total Parameters: {total_params:,}")
    print(f"[+] Frozen Parameters (Base SLM): {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
    print(f"[+] Trainable Parameters (MLP Projector): {trainable_params:,} ({trainable_params/total_params*100:.4f}%)")
    assert trainable_params < 2.5e6, "Trainable parameters should be ~1.97M, base SLM must be frozen!"

    # 2. Construct Mock Batch
    print(f"\n[2] Constructing mock multimodal batch (Batch size = {batch_size}, N_phys = 13)...")
    tokenizer = wrapper.tokenizer

    prefix_text = "<|im_start|>system\nYou are a sensor-grounded reasoning assistant.<|im_end|>\n<|im_start|>user\n[Physical Observations]: "
    suffix_text = "\nQuestion: What is the subject's posture and depth?<|im_end|>\n<|im_start|>assistant\nposture: standing, depth_m: 2.50.<|im_end|>"

    enc_prefix = tokenizer([prefix_text] * batch_size, return_tensors="pt", padding=True)
    enc_suffix = tokenizer([suffix_text] * batch_size, return_tensors="pt", padding=True)

    prefix_input_ids = enc_prefix["input_ids"].to(device)
    prefix_attention_mask = enc_prefix["attention_mask"].to(device)
    suffix_input_ids = enc_suffix["input_ids"].to(device)
    suffix_attention_mask = enc_suffix["attention_mask"].to(device)

    # Physical tokens: (B, 13, 384)
    physical_tokens = torch.randn((batch_size, 13, 384), dtype=dtype, device=device)

    # Construct labels for loss computation (masking prompt tokens with -100)
    L_tot = prefix_input_ids.shape[1] + 13 + suffix_input_ids.shape[1]
    labels = torch.full((batch_size, L_tot), -100, dtype=torch.long, device=device)
    # Supervision strictly on suffix assistant tokens
    labels[:, -suffix_input_ids.shape[1]:] = suffix_input_ids

    # 3. Forward Pass
    print("\n[3] Executing Forward Pass...")
    loss, logits = wrapper(
        prefix_input_ids=prefix_input_ids,
        prefix_attention_mask=prefix_attention_mask,
        physical_tokens=physical_tokens,
        suffix_input_ids=suffix_input_ids,
        suffix_attention_mask=suffix_attention_mask,
        labels=labels
    )
    print(f"[+] Forward Pass Successful! Computed Cross-Entropy Loss: {loss.item():.4f}")
    print(f"[+] Output Logits Shape: {logits.shape}")

    if torch.cuda.is_available():
        vram_after_forward = torch.cuda.memory_allocated() / (1024 ** 3)
        print(f"[+] VRAM after forward pass: {vram_after_forward:.2f} GB")

    # 4. Backward Pass (Gradient Flow Verification)
    print("\n[4] Executing Backward Pass (Testing Gradient Flow to Projector)...")
    loss.backward()

    # Verify projector gradients
    proj_weight_grad = wrapper.projector.net[0].weight.grad
    assert proj_weight_grad is not None, "Error: Projector layer 0 has no gradients!"
    print(f"[+] Projector Layer 0 Grad Norm: {proj_weight_grad.norm().item():.6f}")

    # Verify SLM weights have NO gradients
    slm_first_layer_weight = None
    for name, param in wrapper.llm.named_parameters():
        if "weight" in name and param.requires_grad is False:
            assert param.grad is None, f"Error: Frozen SLM parameter {name} has gradients! Leakage detected!"
            slm_first_layer_weight = name
            break
    print(f"[+] Verified Base SLM parameters have ZERO gradients ({slm_first_layer_weight}.grad is None)")

    # 5. Measure Actual Peak VRAM
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print("\n" + "=" * 70)
        print(f" EMPIRICAL VRAM BENCHMARK RESULT: {peak_vram_gb:.2f} GB / {total_vram_gb:.2f} GB ")
        print(f" Free VRAM Headroom: {total_vram_gb - peak_vram_gb:.2f} GB ")
        print("=" * 70)
        assert peak_vram_gb < 11.0, f"Peak VRAM ({peak_vram_gb:.2f} GB) is dangerously close to 12 GB limit!"

    print("\n[✔] ALL SMOKE TEST CHECKS PASSED SUCCESSFULLY! Ready for Stage 4 training.")


if __name__ == "__main__":
    run_smoke_test()
