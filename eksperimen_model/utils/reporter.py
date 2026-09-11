import os
import time
import datetime
import json
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg") # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

SKELETON_BONES = [
    (0, 7), (7, 8), (8, 9), (9, 10),     # Spine to Head
    (0, 4), (4, 5), (5, 6),             # Left Leg
    (0, 1), (1, 2), (2, 3),             # Right Leg
    (8, 11), (11, 12), (12, 13),        # Left Arm
    (8, 14), (14, 15), (15, 16)         # Right Arm
]

JOINT_NAMES = [
    "Pelvis", "R_Hip", "R_Knee", "R_Ankle", "L_Hip", "L_Knee", "L_Ankle",
    "Spine", "Thorax", "Neck/Nose", "Head", "L_Shoulder", "L_Elbow", "L_Wrist",
    "R_Shoulder", "R_Elbow", "R_Wrist"
]

ANATOMICAL_GROUPS = {
    "Central Axis": [0, 7, 8, 9, 10],      # Pelvis, Spine, Thorax, Neck, Head
    "Lower Limbs": [1, 2, 3, 4, 5, 6],     # Hips, Knees, Ankles
    "Upper Limbs": [11, 12, 13, 14, 15, 16] # Shoulders, Elbows, Wrists
}

class ExperimentReporter:
    """
    Automated Reporter for Deep Learning Training Pipelines.
    Creates a dedicated timestamped folder inside docs/report_training/
    and generates publication-ready graphs, 3D visualizations, tables, and reports.
    Supports multi-environment (E01-E04) stratified breakdown and robustness analysis.
    """
    def __init__(
        self,
        exp_name: str,
        config: Dict[str, Any],
        base_dir: str = "docs/report_training"
    ):
        self.exp_name = exp_name
        self.config = config
        self.base_dir = base_dir
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"RUN_{self.timestamp}_{exp_name}"
        self.run_dir = os.path.join(base_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        print(f"[Reporter] Initialized report directory: {self.run_dir}")

    def save_config_snapshot(self) -> str:
        """Saves a clean copy of the configuration YAML used for the experiment."""
        cfg_path = os.path.join(self.run_dir, "config_snapshot.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        return cfg_path

    def plot_loss_curves(self, history: List[Dict[str, Any]]) -> str:
        """Generates loss and learning rate curves over epochs, including per-env val curves if available."""
        epochs = [entry['epoch'] for entry in history]
        train_losses = [entry['train']['loss'] for entry in history]
        val_mpjpes = [entry['val']['mpjpe'] for entry in history]
        lrs = [entry.get('lr', 0.0) for entry in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=300)
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

        # Subplot 1: Loss & MPJPE
        ax1.plot(epochs, train_losses, 'o-', color='#1f77b4', linewidth=2.5, markersize=5, label='Train Loss (m)')
        ax1.plot(epochs, val_mpjpes, 's-', color='#d62728', linewidth=2.5, markersize=5, label='Val MPJPE (Overall)')

        # Plot per-environment curves if available
        env_colors = {'E01': '#16a085', 'E02': '#e67e22', 'E03': '#8e44ad', 'E04': '#2c3e50'}
        for env_key, col in env_colors.items():
            env_vals = []
            has_env_data = True
            for entry in history:
                env_dict = entry['val'].get('env_mpjpe', {})
                if env_key in env_dict and env_dict[env_key] > 0:
                    env_vals.append(env_dict[env_key])
                else:
                    has_env_data = False
                    break
            if has_env_data and len(env_vals) == len(epochs):
                ax1.plot(epochs, env_vals, '--', color=col, linewidth=1.5, alpha=0.8, label=f'Val {env_key}')

        ax1.set_title('Training & Validation Convergence (MPJPE)', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Error (Meters)', fontsize=11)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend(loc='upper right', frameon=True, fontsize=9)

        # Annotate best validation score
        min_val = min(val_mpjpes)
        min_epoch = epochs[val_mpjpes.index(min_val)]
        ax1.scatter([min_epoch], [min_val], color='#2ca02c', s=120, zorder=5)
        ax1.annotate(f'Best: {min_val:.4f}m ({min_val*1000:.1f}mm)',
                     (min_epoch, min_val),
                     textcoords="offset points", xytext=(0, 12), ha='center',
                     fontweight='bold', color='#2ca02c')

        # Subplot 2: Learning Rate
        ax2.plot(epochs, lrs, '^-', color='#9467bd', linewidth=2, markersize=5)
        ax2.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Learning Rate', fontsize=11)
        ax2.set_yscale('log')
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plot_path = os.path.join(self.run_dir, "loss_curve.png")
        fig.savefig(plot_path)
        plt.close(fig)
        return plot_path

    def plot_per_joint_errors(self, per_joint_errors: np.ndarray, joint_names: List[str] = JOINT_NAMES) -> str:
        """Generates per-joint error bar chart color-coded by body anatomy."""
        fig, ax = plt.subplots(figsize=(13, 6), dpi=300)

        # Color coding by anatomical section
        colors = []
        for i in range(len(joint_names)):
            if i in ANATOMICAL_GROUPS["Central Axis"]:
                colors.append('#3498db')
            elif i in ANATOMICAL_GROUPS["Lower Limbs"]:
                colors.append('#2ecc71')
            else:
                colors.append('#e74c3c')

        x_pos = np.arange(len(joint_names))
        bars = ax.bar(x_pos, per_joint_errors, color=colors, edgecolor='black', alpha=0.85, width=0.6)

        # Add error values above bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height*1000:.0f}mm\n({height:.2f}m)',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_title('Mean Per-Joint Position Error (MPJPE) on Unseen Validation Set', fontsize=13, fontweight='bold')
        ax.set_xlabel('Anatomical Joint Keypoints', fontsize=11)
        ax.set_ylabel('Error (Meters)', fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(joint_names, rotation=45, ha='right', fontsize=9)
        ax.grid(True, linestyle='--', axis='y', alpha=0.6)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498db', edgecolor='black', label='Central Body Axis'),
            Patch(facecolor='#2ecc71', edgecolor='black', label='Lower Limbs (Legs)'),
            Patch(facecolor='#e74c3c', edgecolor='black', label='Upper Limbs (Arms)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', frameon=True)

        plt.tight_layout()
        plot_path = os.path.join(self.run_dir, "per_joint_error.png")
        fig.savefig(plot_path)
        plt.close(fig)
        return plot_path

    def plot_per_env_comparison(self, per_env_mpjpe: Dict[str, float]) -> str:
        """Generates bar chart comparing MPJPE across environments E01-E04."""
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        envs = list(per_env_mpjpe.keys())
        errors_mm = [per_env_mpjpe[e] * 1000.0 for e in envs]
        colors = ['#16a085', '#e67e22', '#8e44ad', '#2c3e50'][:len(envs)]

        bars = ax.bar(envs, errors_mm, color=colors, edgecolor='black', alpha=0.85, width=0.5)

        mean_err = np.mean(errors_mm)
        std_err = np.std(errors_mm)
        cv = (std_err / mean_err) if mean_err > 0 else 0.0
        ers = max(0.0, 1.0 - cv)

        ax.axhline(mean_err, color='#e74c3c', linestyle='--', linewidth=2, label=f'Mean MPJPE: {mean_err:.1f} mm')

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f} mm\n({height/1000:.3f}m)',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_title(f'Cross-Environment Generalization (ERS: {ers:.3f})', fontsize=13, fontweight='bold')
        ax.set_xlabel('Environment', fontsize=11)
        ax.set_ylabel('MPJPE (mm)', fontsize=11)
        ax.grid(True, linestyle='--', axis='y', alpha=0.6)
        ax.legend(loc='upper right', frameon=True)

        plt.tight_layout()
        plot_path = os.path.join(self.run_dir, "env_comparison.png")
        fig.savefig(plot_path)
        plt.close(fig)
        return plot_path

    def plot_joint_env_heatmap(
        self,
        joint_env_matrix: np.ndarray,
        envs: List[str] = ["E01", "E02", "E03", "E04"],
        joint_names: List[str] = JOINT_NAMES
    ) -> str:
        """
        Generates 17 x 4 heatmap showing per-joint error across each environment in mm.
        joint_env_matrix shape: (17, 4) in meters
        """
        fig, ax = plt.subplots(figsize=(9, 10), dpi=300)
        matrix_mm = joint_env_matrix * 1000.0

        im = ax.imshow(matrix_mm, cmap='YlOrRd', aspect='auto')

        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('MPJPE (mm)', rotation=-90, va="bottom", fontsize=10)

        ax.set_xticks(np.arange(len(envs)))
        ax.set_yticks(np.arange(len(joint_names)))
        ax.set_xticklabels(envs, fontsize=11, fontweight='bold')
        ax.set_yticklabels(joint_names, fontsize=9)

        # Annotate text values
        for i in range(len(joint_names)):
            for j in range(len(envs)):
                val = matrix_mm[i, j]
                text_color = "white" if val > (matrix_mm.max() * 0.65) else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", color=text_color, fontsize=8, fontweight='bold')

        ax.set_title("17-Joint Error Heatmap Across Environments (mm)", fontsize=13, fontweight='bold', pad=12)
        plt.tight_layout()
        plot_path = os.path.join(self.run_dir, "joint_env_heatmap.png")
        fig.savefig(plot_path)
        plt.close(fig)
        return plot_path

    def plot_error_distribution(self, sample_errors: np.ndarray) -> str:
        """Generates error distribution histogram & cumulative distribution function (CDF)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
        errs_mm = sample_errors * 1000.0

        # Histogram
        ax1.hist(errs_mm, bins=50, color='#3498db', edgecolor='black', alpha=0.7, density=True)
        mean_v = np.mean(errs_mm)
        median_v = np.median(errs_mm)
        p90_v = np.percentile(errs_mm, 90)

        ax1.axvline(mean_v, color='#e74c3c', linestyle='-', linewidth=2, label=f'Mean: {mean_v:.1f} mm')
        ax1.axvline(median_v, color='#2ecc71', linestyle='--', linewidth=2, label=f'Median: {median_v:.1f} mm')
        ax1.axvline(p90_v, color='#f39c12', linestyle=':', linewidth=2, label=f'90th %ile: {p90_v:.1f} mm')
        ax1.set_title('Per-Sample MPJPE Density Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('MPJPE (mm)', fontsize=10)
        ax1.set_ylabel('Density', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='upper right')

        # CDF (PCK curve)
        sorted_errs = np.sort(errs_mm)
        cdf = np.arange(1, len(sorted_errs) + 1) / len(sorted_errs) * 100.0
        ax2.plot(sorted_errs, cdf, color='#8e44ad', linewidth=2.5)
        ax2.axhline(50, color='gray', linestyle=':', alpha=0.7)
        ax2.axhline(80, color='gray', linestyle=':', alpha=0.7)
        ax2.set_xlim(0, max(300, min(1000, p90_v * 1.5)))
        ax2.set_ylim(0, 105)
        ax2.set_title('Cumulative Distribution Function (PCK Curve)', fontsize=12, fontweight='bold')
        ax2.set_xlabel(r'Distance Threshold $\tau$ (mm)', fontsize=10)
        ax2.set_ylabel('Percentage of Samples (%)', fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plot_path = os.path.join(self.run_dir, "error_distribution.png")
        fig.savefig(plot_path)
        plt.close(fig)
        return plot_path

    def plot_3d_skeleton_comparison(
        self,
        pred_joints: np.ndarray,
        gt_joints: np.ndarray,
        sample_title: str = "Frame 3D Skeleton Prediction vs Ground Truth"
    ) -> str:
        """
        Renders a 3D wireframe plot comparing predicted joints with ground truth joints.
        pred_joints, gt_joints shape: (17, 3)
        """
        fig = plt.figure(figsize=(9, 8), dpi=300)
        ax = fig.add_subplot(111, projection='3d')

        # Plot ground truth bones & joints
        ax.scatter(gt_joints[:, 0], gt_joints[:, 1], gt_joints[:, 2], color='#2ecc71', s=50, label='Ground Truth Joints', zorder=5)
        for u, v in SKELETON_BONES:
            ax.plot([gt_joints[u, 0], gt_joints[v, 0]],
                    [gt_joints[u, 1], gt_joints[v, 1]],
                    [gt_joints[u, 2], gt_joints[v, 2]],
                    color='#27ae60', linewidth=2.5, alpha=0.8)

        # Plot predicted bones & joints
        ax.scatter(pred_joints[:, 0], pred_joints[:, 1], pred_joints[:, 2], color='#e74c3c', s=50, marker='^', label='Predicted Joints', zorder=5)
        for u, v in SKELETON_BONES:
            ax.plot([pred_joints[u, 0], pred_joints[v, 0]],
                    [pred_joints[u, 1], pred_joints[v, 1]],
                    [pred_joints[u, 2], pred_joints[v, 2]],
                    color='#c0392b', linewidth=2.0, linestyle='--', alpha=0.85)

        ax.set_title(sample_title, fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('Lateral X (m)', labelpad=8)
        ax.set_ylabel('Depth Y (m)', labelpad=8)
        ax.set_zlabel('Height Z (m)', labelpad=8)
        ax.legend(loc='upper right', frameon=True)

        # Equal aspect ratio
        max_range = np.array([
            gt_joints[:, 0].max() - gt_joints[:, 0].min(),
            gt_joints[:, 1].max() - gt_joints[:, 1].min(),
            gt_joints[:, 2].max() - gt_joints[:, 2].min()
        ]).max() / 2.0
        mid_x = (gt_joints[:, 0].max() + gt_joints[:, 0].min()) * 0.5
        mid_y = (gt_joints[:, 1].max() + gt_joints[:, 1].min()) * 0.5
        mid_z = (gt_joints[:, 2].max() + gt_joints[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.tight_layout()
        plot_path = os.path.join(self.run_dir, "skeleton_3d_comparison.png")
        fig.savefig(plot_path)
        plt.close(fig)
        return plot_path

    def save_metrics_json(self, history: List[Dict[str, Any]], final_eval: Dict[str, Any]) -> str:
        """Saves raw numerical history and evaluation metrics to JSON."""
        metrics_data = {
            'run_id': self.run_id,
            'experiment_name': self.exp_name,
            'timestamp': self.timestamp,
            'history': history,
            'final_evaluation': final_eval
        }
        json_path = os.path.join(self.run_dir, "metrics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2)
        return json_path

    def generate_markdown_report(
        self,
        history: List[Dict[str, Any]],
        final_eval: Dict[str, Any],
        hardware_info: Optional[Dict[str, str]] = None
    ) -> str:
        """Compiles a complete publication-grade report.md embedding graphs, multi-env tables, and anatomical breakdown."""
        report_path = os.path.join(self.run_dir, "report.md")
        best_val = min(entry['val']['mpjpe'] for entry in history)
        best_epoch = next(entry['epoch'] for entry in history if entry['val']['mpjpe'] == best_val)
        final_train_loss = history[-1]['train']['loss']

        hw_gpu = hardware_info.get('gpu', 'NVIDIA GPU') if hardware_info else 'NVIDIA GPU'
        hw_cpu = hardware_info.get('cpu', 'Multi-core CPU') if hardware_info else 'Multi-core CPU'

        # Per environment metrics
        env_mpjpe = final_eval.get('per_env_mpjpe', {})
        env_ers = final_eval.get('env_robustness_score', 0.0)

        # Anatomical group breakdown
        per_joint_arr = np.array(final_eval.get('per_joint_errors', []))
        group_errors = {}
        if len(per_joint_arr) == 17:
            for g_name, indices in ANATOMICAL_GROUPS.items():
                group_errors[g_name] = float(np.mean(per_joint_arr[indices]))

        md_content = f"""# Training Evaluation Report: `{self.run_id}`

- **Experiment Name:** {self.exp_name}
- **Run ID:** `{self.run_id}`
- **Execution Date:** {datetime.datetime.now().strftime('%d %B %Y, %H:%M:%S')}
- **Hardware Profile:** {hw_gpu} | {hw_cpu}
- **Dataset Split:** Stratified Random Cross-Subject (Seed: 42, 6:2:2 ratio across E01-E04)
- **Best Validation MPJPE:** **{best_val:.4f} meters ({best_val*1000:.1f} mm / {best_val*100:.1f} cm)** at Epoch {best_epoch}
- **Environment Robustness Score (ERS):** **{env_ers:.3f}** (1.0 = perfect generalization across all environments)

---

## 1. Executive Summary & Convergence

| Metric | Initial State | Final State | Best Value |
| :--- | :--- | :--- | :--- |
| **Train Loss (MPJPE)** | {history[0]['train']['loss']:.4f} m | {final_train_loss:.4f} m | **{min(e['train']['loss'] for e in history):.4f} m** |
| **Val MPJPE (Overall)** | {history[0]['val']['mpjpe']:.4f} m | {history[-1]['val']['mpjpe']:.4f} m | **{best_val:.4f} m ({best_val*1000:.1f} mm)** |
| **Epochs Completed** | 1 | {len(history)} | Best at Epoch {best_epoch} |
| **Env Robustness Score** | - | - | **{env_ers:.3f}** |

---

## 2. Multi-Environment Generalization Analysis (E01 - E04)

Tabel evaluasi performa per environment pada data validasi unseen subjects:

| Environment | Subjek Uji Validasi | MPJPE (Meters) | MPJPE (mm) | MPJPE (cm) | Relatif terhadap Rerata |
| :---: | :--- | :---: | :---: | :---: | :---: |
"""
        mean_env_err = np.mean(list(env_mpjpe.values())) if env_mpjpe else best_val
        for env_id, val_m in env_mpjpe.items():
            rel_diff = ((val_m - mean_env_err) / mean_env_err * 100.0) if mean_env_err > 0 else 0.0
            sign = "+" if rel_diff > 0 else ""
            subs = ", ".join(self.config.get("dataset", {}).get("split", {}).get("val_subjects", []))
            md_content += f"| **{env_id}** | S-split | {val_m:.4f} m | {val_m*1000:.1f} mm | {val_m*100:.2f} cm | {sign}{rel_diff:.1f}% |\n"

        md_content += f"""
![Cross-Environment Comparison](env_comparison.png)

---

## 3. Training & Validation Curves

![Loss and MPJPE Curves](loss_curve.png)

---

## 4. Anatomical Group Analysis & 17-Joint Breakdown

### Ringkasan Error Berdasarkan Bagian Tubuh:

| Kelompok Anatomi | Sendi yang Termasuk | MPJPE (mm) | MPJPE (m) | Karakteristik Radar |
| :--- | :--- | :---: | :---: | :--- |
| **Central Body Axis** | Pelvis, Spine, Thorax, Neck, Head | {group_errors.get('Central Axis', 0)*1000:.1f} mm | {group_errors.get('Central Axis', 0):.4f} m | Penampang RCS radar terbesar & paling stabil |
| **Lower Limbs (Kaki)** | Hips, Knees, Ankles | {group_errors.get('Lower Limbs', 0)*1000:.1f} mm | {group_errors.get('Lower Limbs', 0):.4f} m | Doppler tinggi saat melangkah / berjalan |
| **Upper Limbs (Tangan)** | Shoulders, Elbows, Wrists | {group_errors.get('Upper Limbs', 0)*1000:.1f} mm | {group_errors.get('Upper Limbs', 0):.4f} m | Ekstremitas paling rentan multipath & oklusi |

![Per-Joint Error Breakdown](per_joint_error.png)

### Tabulasi Nilai Error 17 Sendi Lengkap:

| Index | Joint Name | Anatomical Group | Error (Meters) | Error (mm) | Error (cm) |
| :---: | :--- | :--- | :---: | :---: | :---: |
"""
        for i, (name, err) in enumerate(zip(JOINT_NAMES, per_joint_arr)):
            group = "Central Axis" if i in ANATOMICAL_GROUPS["Central Axis"] else ("Lower Limbs" if i in ANATOMICAL_GROUPS["Lower Limbs"] else "Upper Limbs")
            md_content += f"| {i} | **{name}** | {group} | {err:.4f} m | {err*1000:.1f} mm | {err*100:.2f} cm |\n"

        md_content += f"""
---

## 5. 17-Joint Error Heatmap Across Environments

Heatmap ini menunjukkan distribusi kesalahan per sendi di setiap kondisi lingkungan radar (E01 - E04):

![17-Joint Error Heatmap](joint_env_heatmap.png)

---

## 6. Per-Sample Error Distribution & PCK Analysis

![Error Distribution](error_distribution.png)

---

## 7. 3D Skeleton Prediction vs Ground Truth Visual Sample

Visualisasi perbandingan prediksi pose 3D (merah putus-putus) terhadap ground truth target (hijau solid):

![3D Skeleton Comparison](skeleton_3d_comparison.png)

---

## 8. Artifacts Generated in this Run

- [Configuration Snapshot](config_snapshot.yaml)
- [Raw Metrics Data (JSON)](metrics.json)
- [Loss & Multi-Env Curves Graph](loss_curve.png)
- [Per-Joint Error Breakdown Graph](per_joint_error.png)
- [Environment Comparison Bar Chart](env_comparison.png)
- [Joint x Environment Heatmap](joint_env_heatmap.png)
- [Error Distribution & PCK Curves](error_distribution.png)
- [3D Skeleton Comparison Plot](skeleton_3d_comparison.png)
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return report_path

    def update_master_index(
        self,
        best_val: float,
        total_epochs: int,
        train_loss: float,
        per_env_mpjpe: Optional[Dict[str, float]] = None,
        ers: float = 0.0
    ):
        """Maintains or updates docs/report_training/INDEX.md with comparison entries."""
        index_path = os.path.join(self.base_dir, "INDEX.md")

        header = """# Master Index: Training Experiment Reports

Tabel ini mendokumentasikan seluruh riwayat eksperimen training yang telah dijalankan beserta perbandingan metrik performanya.

| Run ID | Tanggal & Waktu | Eksperimen | Epochs | Final Train Loss | Best Val MPJPE | Env Breakdown (E01/E02/E03/E04) | ERS | Laporan Lengkap |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :---: | :--- |
"""
        env_str = "-"
        if per_env_mpjpe:
            env_str = f"{per_env_mpjpe.get('E01',0)*1000:.0f}/{per_env_mpjpe.get('E02',0)*1000:.0f}/{per_env_mpjpe.get('E03',0)*1000:.0f}/{per_env_mpjpe.get('E04',0)*1000:.0f} mm"

        new_row = f"| `{self.run_id}` | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} | {self.exp_name} | {total_epochs} | {train_loss:.4f} m | **{best_val:.4f} m ({best_val*1000:.1f} mm)** | {env_str} | **{ers:.3f}** | [Lihat Laporan]({self.run_id}/report.md) |\n"

        if not os.path.exists(index_path):
            with open(index_path, "w", encoding="utf-8") as f:
                f.write(header + new_row)
        else:
            with open(index_path, "a", encoding="utf-8") as f:
                f.write(new_row)

        print(f"[Reporter] Master index updated at: {index_path}")

    def finalize_report(
        self,
        history: List[Dict[str, Any]],
        final_eval: Dict[str, Any],
        pred_sample: Optional[np.ndarray] = None,
        gt_sample: Optional[np.ndarray] = None,
        hardware_info: Optional[Dict[str, str]] = None,
        sample_errors: Optional[np.ndarray] = None,
        joint_env_matrix: Optional[np.ndarray] = None
    ) -> str:
        """One-stop method to generate all plots, save metrics, compile report, and update index."""
        print(f"[Reporter] Finalizing comprehensive experiment report for: {self.run_id}...")
        self.save_config_snapshot()
        self.plot_loss_curves(history)

        if 'per_joint_errors' in final_eval:
            self.plot_per_joint_errors(np.array(final_eval['per_joint_errors']))

        if 'per_env_mpjpe' in final_eval and final_eval['per_env_mpjpe']:
            self.plot_per_env_comparison(final_eval['per_env_mpjpe'])

        if joint_env_matrix is not None:
            self.plot_joint_env_heatmap(joint_env_matrix)
        elif 'joint_env_matrix' in final_eval:
            self.plot_joint_env_heatmap(np.array(final_eval['joint_env_matrix']))

        if sample_errors is not None:
            self.plot_error_distribution(sample_errors)
        elif 'sample_errors' in final_eval:
            self.plot_error_distribution(np.array(final_eval['sample_errors']))

        if pred_sample is not None and gt_sample is not None:
            self.plot_3d_skeleton_comparison(pred_sample, gt_sample)

        self.save_metrics_json(history, final_eval)
        report_md = self.generate_markdown_report(history, final_eval, hardware_info)

        best_val = min(e['val']['mpjpe'] for e in history)
        per_env = final_eval.get('per_env_mpjpe', None)
        ers = final_eval.get('env_robustness_score', 0.0)
        self.update_master_index(best_val, len(history), history[-1]['train']['loss'], per_env, ers)

        print(f"[Reporter] All assets successfully generated in: {self.run_dir}")
        print(f"[Reporter] Summary report available at: {report_md}")
        return report_md
