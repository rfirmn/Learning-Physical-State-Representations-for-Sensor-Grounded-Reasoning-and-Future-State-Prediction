import sys
import os
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

from eksperimen_model.datasets import MMFiDataset
from eksperimen_model.models import PointMAEPoseEstimator, PointMAE, MPJPELoss, ChamferDistanceLoss
from eksperimen_model.utils import load_pretrained_point_mae

def test_pipeline():
    print("==========================================================")
    print("   RUNNING SANITY VERIFICATION FOR TRAINING PIPELINE")
    print("==========================================================")

    # 1. Device check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Test 1/5] Hardware Target Device: {device}")
    if torch.cuda.is_available():
        print(f"           GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"           VRAM Allocated: {torch.cuda.memory_allocated(0)/(1024**2):.1f} MB")

    # 2. Dataset & DataLoader check
    print("\n[Test 2/5] Testing MMFiDataset...")
    dataset_root = "datasets/MM-Fi Dataset/filtered_mmwave"
    dataset = MMFiDataset(
        root_dir=dataset_root,
        environments=["E01"],
        subjects=["S01"],
        num_points=128,
        augment=True,
        normalize=True
    )
    print(f"           Total frames in E01/S01 test subset: {len(dataset)}")
    assert len(dataset) > 0, "Dataset returned 0 samples!"

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(dataloader))
    pts = batch['points'].to(device)
    skeleton = batch['skeleton'].to(device)

    print(f"           Batch 'points' shape: {pts.shape} (Expected: [4, 128, 3])")
    print(f"           Batch 'skeleton' shape: {skeleton.shape} (Expected: [4, 17, 3])")
    assert pts.shape == (4, 128, 3), f"Unexpected points shape: {pts.shape}"
    assert skeleton.shape == (4, 17, 3), f"Unexpected skeleton shape: {skeleton.shape}"

    # Verify that points are clean, finite, and strictly bounded
    assert not torch.isnan(pts).any(), "NaN found in points!"
    assert not torch.isinf(pts).any(), "Inf found in points!"
    max_coord = float(torch.max(torch.abs(pts)))
    assert max_coord <= 1.25, f"Points exceeded normalized sphere bound: {max_coord}"
    print(f"           Points numerical range: [-{max_coord:.3f}, +{max_coord:.3f}] (Guaranteed finite & bounded)")

    # 3. Model instantiation & Pretrained weights loading
    print("\n[Test 3/5] Testing PointMAEPoseEstimator and Pretrained Weights Loading...")
    model = PointMAEPoseEstimator(
        num_points=128,
        num_groups=16,
        group_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_joints=17
    ).to(device)

    ckpt_path = "models/Point-MAE/pretrain.pth"
    matched, missing = load_pretrained_point_mae(model, ckpt_path)
    print(f"           Matched backbone layers: {matched}, Missing/New layers: {missing}")
    assert matched > 50, f"Too few matched weights ({matched})!"

    # 4. Forward pass & Physical State Representation extraction
    print("\n[Test 4/5] Testing Forward Pass on GPU...")
    pred_joints, z_t = model(pts)
    print(f"           Predicted Joints shape: {pred_joints.shape} (Expected: [4, 17, 3])")
    print(f"           Physical State (Z_t) shape: {z_t.shape} (Expected: [4, 384])")
    assert pred_joints.shape == (4, 17, 3)
    assert z_t.shape == (4, 384)

    # 5. Loss calculation & Backward Gradient step
    print("\n[Test 5/5] Testing MPJPE Loss and Backward Gradient Flow...")
    criterion = MPJPELoss()
    loss = criterion(pred_joints, skeleton)
    print(f"           Initial MPJPE Loss: {loss.item():.4f} meters ({loss.item()*1000:.1f} mm)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("           Backward pass & optimizer step executed successfully!")

    print("\n==========================================================")
    print("   ALL TESTS PASSED! PIPELINE IS READY FOR TRAINING.")
    print("==========================================================")

if __name__ == "__main__":
    test_pipeline()
