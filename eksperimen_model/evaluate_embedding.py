import os
import sys
import argparse
import yaml
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, accuracy_score
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))

from eksperimen_model.datasets import MMFiDataset
from eksperimen_model.models import PointMAEPoseEstimator

def evaluate_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: str = "docs/report_training/embedding_analysis",
    max_samples: int = 3000
):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    all_z = []
    all_actions = []
    all_envs = []
    all_subs = []

    print(f"\n[Embedding Analysis] Extracting physical state representations Z_t (limit: {max_samples} samples)...")
    sample_count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Z_t"):
            pts = batch['points'].to(device)
            actions = batch['action'].cpu().numpy()
            envs = batch.get('env', ['E01'] * len(pts))
            subs = batch.get('sub', ['S01'] * len(pts))

            _, z_t = model(pts)
            z_np = z_t.cpu().numpy() # (B, 384)

            all_z.append(z_np)
            all_actions.extend(actions)
            all_envs.extend(envs)
            all_subs.extend(subs)

            sample_count += len(z_np)
            if sample_count >= max_samples:
                break

    all_z = np.concatenate(all_z, axis=0)[:max_samples]
    all_actions = np.array(all_actions)[:max_samples]
    all_envs = np.array(all_envs)[:max_samples]
    all_subs = np.array(all_subs)[:max_samples]

    print(f"[Embedding Analysis] Collected Z_t shape: {all_z.shape} (384-dimensional physical states)")

    # 1. Linear Probe Evaluation (Predict Action from Z_t)
    print("\n[Linear Probe] Training linear classifier on Z_t to evaluate action decodability...")
    n_train = int(len(all_z) * 0.75)
    indices = np.random.permutation(len(all_z))
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(all_z[train_idx], all_actions[train_idx])
    pred_actions = clf.predict(all_z[test_idx])
    probe_acc = float(accuracy_score(all_actions[test_idx], pred_actions) * 100.0)
    print(f" >>> Linear Probe Action Top-1 Accuracy: {probe_acc:.2f}% (Chance level: {100.0/len(np.unique(all_actions)):.2f}%)")

    # 2. Silhouette Score (Cluster quality across actions)
    print("\n[Silhouette Score] Computing cluster separation quality across actions...")
    try:
        sil_score = float(silhouette_score(all_z, all_actions, metric='cosine'))
        print(f" >>> Physical State Silhouette Score (Cosine): {sil_score:.4f}")
    except Exception as e:
        sil_score = 0.0
        print(f" >>> Could not compute silhouette score: {e}")

    # 3. Intra-action vs Inter-action Cosine Similarity
    print("\n[Cosine Similarity] Calculating intra-action vs inter-action physical similarity...")
    norm_z = all_z / (np.linalg.norm(all_z, axis=1, keepdims=True) + 1e-8)
    # Sample 500 random pairs
    n_pairs = min(500, len(norm_z))
    p_idx1 = np.random.randint(0, len(norm_z), n_pairs)
    p_idx2 = np.random.randint(0, len(norm_z), n_pairs)

    same_mask = (all_actions[p_idx1] == all_actions[p_idx2])
    diff_mask = ~same_mask

    sims = np.sum(norm_z[p_idx1] * norm_z[p_idx2], axis=1)
    intra_sim = float(np.mean(sims[same_mask])) if np.any(same_mask) else 1.0
    inter_sim = float(np.mean(sims[diff_mask])) if np.any(diff_mask) else 0.0
    print(f" >>> Intra-Action Cosine Similarity : {intra_sim:.4f}")
    print(f" >>> Inter-Action Cosine Similarity : {inter_sim:.4f}")
    print(f" >>> Semantic Margin (Intra - Inter): {intra_sim - inter_sim:.4f}")

    # 4. 2D t-SNE Visualization
    print("\n[t-SNE] Computing 2D manifold projection for visualization...")
    perp = min(30, max(5, (len(all_z) - 1) // 3))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    z_2d = tsne.fit_transform(all_z)

    # Figure 1: t-SNE by Action Class
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=all_actions, cmap='tab20', alpha=0.7, s=20)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Action Class Index (0 - 26)', fontsize=10)
    ax.set_title(f'Physical State Representation Z_t Manifold by Action (Probe Acc: {probe_acc:.1f}%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=10)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    action_plot_path = os.path.join(output_dir, "tsne_actions.png")
    fig.savefig(action_plot_path)
    plt.close(fig)

    # Figure 2: t-SNE by Environment
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    env_palette = {'E01': '#16a085', 'E02': '#e67e22', 'E03': '#8e44ad', 'E04': '#2c3e50'}
    for env_name, color in env_palette.items():
        mask = (all_envs == env_name)
        if np.any(mask):
            ax.scatter(z_2d[mask, 0], z_2d[mask, 1], c=color, label=env_name, alpha=0.6, s=20)
    ax.legend(title="Environment", loc="upper right")
    ax.set_title('Physical State Representation Z_t Manifold by Environment (Disentanglement)', fontsize=12, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=10)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    env_plot_path = os.path.join(output_dir, "tsne_environments.png")
    fig.savefig(env_plot_path)
    plt.close(fig)

    # Summary results dictionary
    summary = {
        'num_samples_analyzed': len(all_z),
        'linear_probe_accuracy_pct': probe_acc,
        'silhouette_score_cosine': sil_score,
        'intra_action_cosine_sim': intra_sim,
        'inter_action_cosine_sim': inter_sim,
        'semantic_margin': intra_sim - inter_sim,
        'plots': {
            'action_manifold': action_plot_path,
            'environment_manifold': env_plot_path
        }
    }

    json_path = os.path.join(output_dir, "embedding_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Done] Embedding analysis complete! Assets saved to: {output_dir}")
    return summary

def main():
    parser = argparse.ArgumentParser(description="Evaluate Physical State Representation Z_t (Point-MAE)")
    parser.add_argument("--checkpoint", type=str, default="eksperimen_model/checkpoints/pose_estimation/best_model.pth")
    parser.add_argument("--config", type=str, default="eksperimen_model/configs/mmfi_pose_finetune.yaml")
    parser.add_argument("--output_dir", type=str, default="docs/report_training/embedding_analysis")
    parser.add_argument("--max_samples", type=int, default=3000)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use unseen validation subjects for clean representation analysis
    val_subjects = cfg["dataset"]["split"].get("val_subjects", ["S05", "S10", "S18", "S20", "S24", "S28", "S34", "S35"])

    dataset = MMFiDataset(
        root_dir=cfg["dataset"]["root_dir"],
        subjects=val_subjects,
        environments=cfg["dataset"].get("environments", ["E01", "E02", "E03", "E04"]),
        num_points=cfg["model"]["num_points"],
        augment=False,
        normalize=True
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    model = PointMAEPoseEstimator(
        num_points=cfg["model"]["num_points"],
        num_groups=cfg["model"]["num_groups"],
        group_size=cfg["model"]["group_size"],
        embed_dim=cfg["model"]["embed_dim"],
        depth=cfg["model"]["depth"],
        num_heads=cfg["model"]["num_heads"],
        num_joints=cfg["model"]["num_joints"]
    ).to(device)

    if os.path.exists(args.checkpoint):
        print(f"Loading weights from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    else:
        print(f"[Warning] Checkpoint '{args.checkpoint}' not found! Evaluating randomly initialized model.")

    evaluate_embeddings(model, dataloader, device, output_dir=args.output_dir, max_samples=args.max_samples)

if __name__ == "__main__":
    main()
