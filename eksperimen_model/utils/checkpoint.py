import os
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

def load_pretrained_point_mae(
    model: nn.Module,
    checkpoint_path: str,
    load_backbone_only: bool = True
) -> Tuple[int, int]:
    """
    Loads Point-MAE pretrained weights (from ShapeNet) into the model.
    Strips 'module.' prefix from DataParallel/DDP keys.
    Returns: (matched_keys_count, missing_keys_count)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"[Checkpoint] Loading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # In Point-MAE pretrain.pth, weights are stored under 'base_model' or 'model'
    if 'base_model' in checkpoint:
        state_dict = checkpoint['base_model']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Clean keys: remove 'module.' prefix
    cleaned_state_dict: Dict[str, Any] = {}
    for k, v in state_dict.items():
        clean_k = k
        if clean_k.startswith('module.'):
            clean_k = clean_k[7:]
        cleaned_state_dict[clean_k] = v

    model_dict = model.state_dict()
    matched_dict: Dict[str, Any] = {}
    missing_keys = []
    mismatched_keys = []

    for name, param in model_dict.items():
        if name in cleaned_state_dict:
            ckpt_val = cleaned_state_dict[name]
            if ckpt_val.shape == param.shape:
                matched_dict[name] = ckpt_val
            else:
                mismatched_keys.append((name, param.shape, ckpt_val.shape))
        else:
            missing_keys.append(name)

    model_dict.update(matched_dict)
    model.load_state_dict(model_dict)

    print(f"[Checkpoint] Successfully loaded {len(matched_dict)} layers from pretrained checkpoint.")
    if mismatched_keys:
        print(f"[Checkpoint] Warning: {len(mismatched_keys)} layers had shape mismatch (e.g. classification/regression heads):")
        for k, s1, s2 in mismatched_keys[:5]:
            print(f"   {k}: model {s1} vs ckpt {s2}")
    if missing_keys:
        print(f"[Checkpoint] Note: {len(missing_keys)} layers freshly initialized (e.g. newly added pose head).")

    return len(matched_dict), len(missing_keys)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: str
):
    """Saves training checkpoint including model, optimizer, epoch, and validation metrics."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(state, save_path)
    print(f"[Checkpoint] Checkpoint saved successfully to: {save_path}")
