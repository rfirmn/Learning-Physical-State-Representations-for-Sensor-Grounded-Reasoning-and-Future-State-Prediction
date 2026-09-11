import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))

from eksperimen_model.datasets import MMFiDataset
from eksperimen_model.models import PointMAE, ChamferDistanceLoss
from eksperimen_model.utils import load_pretrained_point_mae, save_checkpoint, ExperimentLogger

def main():
    parser = argparse.ArgumentParser(description="Point-MAE Self-Supervised Domain Adaptation on MM-Fi Radar Point Clouds")
    parser.add_argument("--config", type=str, default="eksperimen_model/configs/mmfi_mae_pretrain.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--env", type=str, default="E01")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    lr = args.lr or cfg["training"]["lr"]
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    logger = ExperimentLogger(output_dir, cfg["experiment_name"])

    print("=" * 65)
    print(f" SELF-SUPERVISED POINT-MAE ADAPTATION: {cfg['experiment_name']}")
    print(f" DEVICE: {device} | BATCH SIZE: {batch_size} | EPOCHS: {epochs}")
    print("=" * 65)

    dataset_root = cfg["dataset"]["root_dir"]
    num_points = cfg["model"]["num_points"]

    train_dataset = MMFiDataset(
        root_dir=dataset_root,
        subjects=cfg["dataset"].get("train_subjects"),
        environments=[args.env],
        num_points=num_points,
        augment=True,
        normalize=True
    )
    val_dataset = MMFiDataset(
        root_dir=dataset_root,
        subjects=cfg["dataset"].get("val_subjects"),
        environments=[args.env],
        num_points=num_points,
        augment=False,
        normalize=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = PointMAE(
        num_points=num_points,
        num_groups=cfg["model"]["num_groups"],
        group_size=cfg["model"]["group_size"],
        mask_ratio=cfg["model"]["mask_ratio"],
        embed_dim=cfg["model"]["embed_dim"],
        depth=cfg["model"]["depth"],
        decoder_depth=cfg["model"]["decoder_depth"],
        num_heads=cfg["model"]["num_heads"]
    ).to(device)

    # Load initial ShapeNet pretrained weights
    ckpt_path = cfg["model"]["pretrained_weights"]
    if os.path.exists(ckpt_path):
        load_pretrained_point_mae(model, ckpt_path)

    criterion = ChamferDistanceLoss(metric='l2')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["training"].get("weight_decay", 0.05))

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]")
        for batch in pbar:
            pts = batch['points'].to(device)
            optimizer.zero_grad()
            pred_patches, target_patches = model(pts)
            # Flatten patches for Chamfer distance: (B, G*K, 3)
            B, G, K, C = pred_patches.shape
            loss = criterion(pred_patches.view(B, G * K, C), target_patches.view(B, G * K, C))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batches += 1
            pbar.set_postfix({'Chamfer Loss': f"{loss.item():.5f}"})

        train_loss /= max(1, batches)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Val]  "):
                pts = batch['points'].to(device)
                pred_patches, target_patches = model(pts)
                B, G, K, C = pred_patches.shape
                loss = criterion(pred_patches.view(B, G * K, C), target_patches.view(B, G * K, C))
                val_loss += loss.item()
                val_batches += 1

        val_loss /= max(1, val_batches)
        current_lr = optimizer.param_groups[0]['lr']

        logger.log_epoch(
            epoch=epoch,
            train_metrics={'chamfer_loss': train_loss},
            val_metrics={'chamfer_loss': val_loss},
            lr=current_lr
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, {'chamfer_loss': val_loss}, os.path.join(output_dir, "best_mae_model.pth"))
            print(f"   >>> NEW BEST MAE MODEL! Val Chamfer Loss: {val_loss:.6f}")

    print("MAE Self-Supervised Adaptation Training Completed!")

if __name__ == "__main__":
    main()
