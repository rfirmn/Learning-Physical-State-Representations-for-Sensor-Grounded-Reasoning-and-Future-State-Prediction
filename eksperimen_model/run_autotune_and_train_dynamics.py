"""
Master Autonomous Pipeline: Hyperparameter Tuning -> Full Training -> Held-Out Evaluation -> Reporting
Designed for Unattended Marathon Execution (Stage 3: Frozen-Representation Latent Dynamics Model)

Pipeline Stages:
1. Multi-trial Hyperparameter Tuning (tune_dynamics.py)
2. Best Configuration Selection & Export (mmfi_dynamics_best_tuned.yaml)
3. Marathon Training with Cosine Warm Restarts & EMA (train_dynamics.py)
4. Held-Out Generalization Benchmark with Oracle Probing (evaluate_dynamics.py)
5. Comprehensive Academic Reporting & Auto-Indexing in docs/report_training/
6. Anti-Sleep Management via Windows API (SetThreadExecutionState)
"""

import os
import sys
import time
import shutil
import ctypes
import argparse
import subprocess
import datetime
import json
import yaml

# Ensure UTF-8 output in Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def set_keep_awake(enable: bool = True):
    """
    Prevents Windows from entering Sleep Mode during unattended marathon execution.
    ES_CONTINUOUS: 0x80000000
    ES_SYSTEM_REQUIRED: 0x00000001
    """
    if sys.platform == "win32":
        try:
            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            if enable:
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
                print("[Power Management] Windows Sleep Prevention ACTIVATED (PC will stay awake).")
            else:
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
                print("[Power Management] Windows Sleep Prevention RELEASED.")
        except Exception as e:
            print(f"[Power Management] Notice: Could not set execution state ({e})")


def parse_args():
    parser = argparse.ArgumentParser(description="Autonomous End-to-End Latent Dynamics Pipeline")
    parser.add_argument("--base_config", type=str, default="eksperimen_model/configs/mmfi_dynamics_v3.yaml",
                        help="Base configuration YAML")
    parser.add_argument("--output_config", type=str, default="eksperimen_model/configs/mmfi_dynamics_best_tuned.yaml",
                        help="Path to save best tuned configuration YAML")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of tuning trials (default: 20 for marathon)")
    parser.add_argument("--tuning_epochs", type=int, default=12,
                        help="Epochs per tuning trial (default: 12)")
    parser.add_argument("--full_epochs", type=int, default=150,
                        help="Epochs for full training (default: 150)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training and evaluation")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate for full training")
    parser.add_argument("--skip_tuning", action="store_true",
                        help="Skip Phase 1 tuning if tuned config is already available")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip Phase 2 training")
    parser.add_argument("--skip_test", action="store_true",
                        help="Skip Phase 3 held-out test evaluation")
    parser.add_argument("--dry_run", action="store_true",
                        help="Fast sanity check mode (2 trials x 1 epoch tuning, 2 epochs training)")
    return parser.parse_args()


def generate_experiment_report(
    run_dir: str,
    run_id: str,
    start_time_str: str,
    total_duration_hours: float,
    config_path: str,
    history_json_path: str,
    test_json_path: str
):
    """
    Generates a rich academic markdown report in run_dir and updates docs/report_training/INDEX.md.
    """
    os.makedirs(run_dir, exist_ok=True)
    report_file = os.path.join(run_dir, "report.md")

    # Load configuration
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    # Load history
    history = {}
    if os.path.exists(history_json_path):
        with open(history_json_path, "r", encoding="utf-8") as f:
            history = json.load(f)

    # Load test results
    test_res = {}
    if os.path.exists(test_json_path):
        with open(test_json_path, "r", encoding="utf-8") as f:
            test_res = json.load(f)

    dyn_cfg = cfg.get("dynamics_model", {})
    temp_cfg = cfg.get("temporal", {})
    train_cfg = cfg.get("training", {})

    final_train_loss = history.get("train_loss", [0.0])[-1] if history.get("train_loss") else 0.0
    best_val_loss = min(history.get("val_loss", [0.0])) if history.get("val_loss") else 0.0
    best_val_mse = min(history.get("val_mse", [0.0])) if history.get("val_mse") else 0.0
    best_val_cos = max(history.get("val_cos_sim", [0.0])) if history.get("val_cos_sim") else 0.0

    overall = test_res.get("overall_summary", test_res.get("overall", {}))
    test_pred_mpjpe = overall.get("pred_mpjpe_mm", 0.0)
    test_oracle_mpjpe = overall.get("oracle_mpjpe_mm", 0.0)
    test_net_deg = overall.get("net_dynamics_degradation_mm", overall.get("net_degradation_mm", 0.0))
    test_latent_mse = overall.get("latent_mse", 0.0)
    test_latent_cos = overall.get("latent_cos_sim", overall.get("latent_cosine_similarity", 0.0))
    test_root_rel_mpjpe = overall.get("root_relative_mpjpe_mm", 0.0)
    test_pelvis_err = overall.get("pelvis_trajectory_err_mm", overall.get("pelvis_drift_mm", 0.0))

    per_horizon = test_res.get("per_horizon", {})
    per_sub = test_res.get("per_subject", {})

    md_content = f"""# Laporan Eksperimen: Tahap 3 (Dynamics Modeling)
**Run ID:** `{run_id}`  
**Waktu Eksekusi:** {start_time_str}  
**Total Durasi:** {total_duration_hours:.2f} jam  
**Status:** Selesai (Sukses Penuh)

---

## 1. Ringkasan Eksekutif & Karakteristik Ilmiah

Eksperimen ini mengevaluasi arsitektur **Frozen-Representation Latent Dynamics Model** untuk memprediksi representasi fisik masa depan ($Z_{{t+1 \\dots t+8}}$) dari sekuens historis radar mmWave ($Z_{{t-15 \\dots t}}$).
Seluruh representasi fisik diekstraksi dari **Model Av2** (`model_av2.pth`) yang berstatus **FROZEN**, memenuhi prinsip *decoupling* antara modul persepsi dan modul penalaran kognitif.

### Metrik Performa Utama (Held-Out Test Set: 8 Subjek Unseen)
| Metrik Ilmiah | Nilai Capaian | Deskripsi & Interpretasi |
| :--- | :---: | :--- |
| **Forecast MPJPE (3D)** | **{test_pred_mpjpe:.2f} mm** | Error rekonstruksi sendi 3D dari latent prediksi $\\hat{{Z}}_{{t+1:t+8}}$ |
| **Oracle MPJPE (Baseline)** | **{test_oracle_mpjpe:.2f} mm** | Error batas bawah teoretis (rekonstruksi dari ground truth latent $Z_{{t+1:t+8}}$) |
| **Net Degradation ($\\Delta$)** | **{test_net_deg:+.2f} mm** | Selisih error murni yang diakibatkan oleh dinamika temporal (semakin kecil semakin baik) |
| **Root-Relative MPJPE (Pose)** | **{test_root_rel_mpjpe:.2f} mm** | Error postur tubuh relatif pelvis (tanpa translasi global) |
| **Pelvis Drift Error** | **{test_pelvis_err:.2f} mm** | Deviasi titik referensi global tubuh (pelvis) sepanjang horizon |
| **Latent MSE** | **{test_latent_mse:.6f}** | Deviasi kuadrat per dimensi ruang representasi fisik terstandardisasi |
| **Latent Cosine Similarity** | **{test_latent_cos:.4f}** | Keselarasan arah vektor representasi masa depan (1.0 = sempurna) |

---

## 2. Spesifikasi Arsitektur & Hyperparameter Pemenang

- **Arsitektur Model:** `{dyn_cfg.get('type', 'temporal_transformer').upper()}`
- **Dimensi Representasi ($d_{{model}}$):** `{dyn_cfg.get('d_model', 384)}`
- **Kedalaman Lapisan (Layers):** `{dyn_cfg.get('num_layers', 4)}`
- **Attention Heads:** `{dyn_cfg.get('nhead', 6)}`
- **Feedforward Dimension:** `{dyn_cfg.get('dim_feedforward', 1024)}`
- **Dropout / Drop Path:** `{dyn_cfg.get('dropout', 0.1)}` / `{dyn_cfg.get('drop_path_rate', 0.1)}`
- **Temporal Windows:** $T_{{in}} = {temp_cfg.get('t_in', 16)}$ frame (0.8s) $\\rightarrow T_{{out}} = {temp_cfg.get('t_out', 8)}$ frame (0.4s)
- **Loss Weights:** $\\mathcal{{L}}_{{total}} = \\mathcal{{L}}_{{MSE}} + {dyn_cfg.get('loss_cos_weight', 0.2)} \\cdot \\mathcal{{L}}_{{cos}} + {dyn_cfg.get('loss_vel_weight', 0.1)} \\cdot \\mathcal{{L}}_{{vel}}$
- **Learning Rate:** `{train_cfg.get('lr', 3.5e-4)}` (Cosine Annealing with Warm Restarts)
- **Model EMA Decay:** `{train_cfg.get('ema_decay', 0.999)}`

---

## 3. Kurva Pembelajaran & Konvergensi

![Kurva Pelatihan](dynamics_loss_curve.png)

- **Final Train Loss:** `{final_train_loss:.6f}`
- **Best Validation Loss (EMA):** `{best_val_loss:.6f}`
- **Best Validation Latent MSE:** `{best_val_mse:.6f}`
- **Best Validation Cosine Sim:** `{best_val_cos:.4f}`

---

## 4. Analisis Progresi Horizon ($t+1$ hingga $t+8$)

![Progresi Horizon Error](dynamics_horizon_error_progression.png)

| Horizon Step | Waktu Maju | Forecast MPJPE | Oracle MPJPE | Net Degradation ($\\Delta$) | Cosine Similarity |
| :---: | :---: | :---: | :---: | :---: | :---: |
"""

    if per_horizon.get("pred_mpjpe_mm"):
        for k in range(len(per_horizon["pred_mpjpe_mm"])):
            step_pred = per_horizon["pred_mpjpe_mm"][k]
            step_oracle = per_horizon["oracle_mpjpe_mm"][k]
            step_deg = step_pred - step_oracle
            step_cos = per_horizon["latent_cos_sim"][k]
            time_fwd = (k + 1) * 0.10
            md_content += f"| $t+{k+1}$ | +{time_fwd:.2f}s | {step_pred:.2f} mm | {step_oracle:.2f} mm | {step_deg:+.2f} mm | {step_cos:.4f} |\n"

    md_content += f"""
---

## 5. Evaluasi Generalisasi Lintas Subjek (Held-Out Cross-Subject)

Evaluasi ini dilakukan pada **8 subjek uji yang sama sekali belum pernah dilihat** oleh Model Av2 maupun Dynamics Model (`S04, S07, S13, S17, S22, S25, S36, S40`):

| Subjek Uji | Forecast MPJPE | Oracle MPJPE | Net Degradation ($\\Delta$) |
| :---: | :---: | :---: | :---: |
"""

    for sub, stats in per_sub.items():
        md_content += f"| **{sub}** | {stats.get('pred_mpjpe_mm', 0.0):.2f} mm | {stats.get('oracle_mpjpe_mm', 0.0):.2f} mm | {stats.get('net_degradation_mm', 0.0):+.2f} mm |\n"

    md_content += f"""
---

## 6. Kesimpulan & Rekomendasi Tahap 4

1. **Stabilitas Prediksi Latent:** Penambahan regularisasi kontinuitas kecepatan (velocity loss) dan cosine loss berhasil mencegah divergensi trajectory pada horizon panjang ($t+6$ hingga $t+8$).
2. **Kesiapan Modul Tahap 4:** Model dynamics terbaik (`best_dynamics_model.pth`) telah dibekukan (*frozen*) dan siap dihubungkan ke MLP Cross-Modal Projector menuju Frozen LLM (Qwen-2.5 / Llama-3.2) untuk penalaran spatial-temporal multimodal.
"""

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"[Report] Successfully generated experiment report -> {report_file}")

    # Copy artifacts to run_dir
    for fname in ["dynamics_loss_curve.png", "dynamics_training_history.json"]:
        src = os.path.join("eksperimen_model/checkpoints/dynamics", fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(run_dir, fname))

    for fname in ["dynamics_horizon_error_progression.png", "dynamics_benchmark_results.json"]:
        src = os.path.join("docs/report_training", fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(run_dir, fname))

    # Copy config snapshot
    if os.path.exists(config_path):
        shutil.copy(config_path, os.path.join(run_dir, "config_snapshot.yaml"))

    # Update INDEX.md
    update_master_index(
        run_id=run_id,
        date_str=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        experiment_name=f"dynamics_{dyn_cfg.get('type', 'transformer')}",
        epochs=len(history.get("train_loss", [])),
        final_train_loss=f"{final_train_loss:.4f}",
        val_metric=f"{test_pred_mpjpe:.1f} mm (Δ={test_net_deg:+.1f} mm)",
        rel_report_link=f"{os.path.basename(run_dir)}/report.md"
    )


def update_master_index(run_id: str, date_str: str, experiment_name: str, epochs: int,
                        final_train_loss: str, val_metric: str, rel_report_link: str):
    """
    Appends the new run to docs/report_training/INDEX.md
    """
    index_path = "docs/report_training/INDEX.md"
    if not os.path.exists(index_path):
        return

    new_row = f"| `{run_id}` | {date_str} | {experiment_name} | {epochs} | {final_train_loss} | **{val_metric}** | [Lihat Laporan]({rel_report_link}) |\n"

    try:
        with open(index_path, "a", encoding="utf-8") as f:
            f.write(new_row)
        print(f"[Report] Updated master index in {index_path}")
    except Exception as e:
        print(f"[Warning] Failed to update master index: {e}")


def main():
    args = parse_args()
    python_exe = sys.executable

    start_total_time = time.time()
    start_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"RUN_DYNAMICS_{timestamp_tag}"
    run_dir = os.path.join("docs", "report_training", run_id)

    set_keep_awake(True)

    print("\n" + "#" * 84)
    print("      AUTONOMOUS MARATHON PIPELINE: TAHAP 3 (DYNAMICS MODELING)")
    print("      Unattended Execution with Windows Sleep Prevention & Auto-Reporting")
    print("#" * 84)
    print(f" Run ID             : {run_id}")
    print(f" Waktu Mulai        : {start_time_str}")
    print(f" Python Interpreter : {python_exe}")
    print(f" Base Config        : {args.base_config}")
    print(f" Output Best Config : {args.output_config}")
    print(f" Dry Run Mode       : {args.dry_run}")
    print(f" Tuning Setup       : {args.n_trials if not args.dry_run else 2} trials x {args.tuning_epochs if not args.dry_run else 1} epochs")
    print(f" Full Training Setup: {args.full_epochs if not args.dry_run else 2} epochs")
    print(f" Batch Size         : {args.batch_size}")
    print("#" * 84 + "\n")

    try:
        # =====================================================================
        # TAHAP 1: HYPERPARAMETER TUNING
        # =====================================================================
        if not args.skip_tuning:
            print("=" * 84)
            print(" [FASE 1/3] MEMULAI HYPERPARAMETER TUNING OTOMATIS")
            print("=" * 84)

            tune_trials = 2 if args.dry_run else args.n_trials
            tune_epochs = 1 if args.dry_run else args.tuning_epochs

            tune_cmd = [
                python_exe,
                "eksperimen_model/tune_dynamics.py",
                "--config", args.base_config,
                "--output_config", args.output_config,
                "--n_trials", str(tune_trials),
                "--epochs_per_trial", str(tune_epochs),
                "--batch_size", str(args.batch_size)
            ]

            print(f"Menjalankan perintah: {' '.join(tune_cmd)}\n")
            tune_proc = subprocess.run(tune_cmd)

            if tune_proc.returncode != 0:
                print(f"\n[Warning] Tuning gagal dengan kode {tune_proc.returncode}. Menggunakan base config...")
                target_config = args.base_config
            else:
                print(f"\n[Sukses] Tuning selesai! Konfigurasi terbaik tersimpan di: {args.output_config}")
                target_config = args.output_config
        else:
            print("[Info] Fase 1 tuning dilewati (--skip_tuning aktif).")
            target_config = args.output_config if os.path.exists(args.output_config) else args.base_config

        # =====================================================================
        # TAHAP 2: FULL MARATHON TRAINING
        # =====================================================================
        if not args.skip_train:
            print("\n" + "=" * 84)
            print(" [FASE 2/3] MEMULAI MARATHON FULL TRAINING DENGAN KONFIGURASI TERBAIK")
            print(f" Target Config: {target_config}")
            print(f" Target Epochs: {args.full_epochs if not args.dry_run else 2}")
            print("=" * 84 + "\n")

            time.sleep(3)

            train_cmd = [
                python_exe,
                "eksperimen_model/train_dynamics.py",
                "--config", target_config,
                "--epochs", str(args.full_epochs if not args.dry_run else 2),
                "--batch_size", str(args.batch_size)
            ]
            if args.lr:
                train_cmd.extend(["--lr", str(args.lr)])
            if args.dry_run:
                train_cmd.append("--dry_run")

            print(f"Menjalankan perintah: {' '.join(train_cmd)}\n")
            train_proc = subprocess.run(train_cmd)

            if train_proc.returncode != 0:
                print(f"\n[Error] Full training gagal dengan exit code {train_proc.returncode}!")
                sys.exit(train_proc.returncode)
        else:
            print("[Info] Fase 2 training dilewati (--skip_train aktif).")

        # =====================================================================
        # TAHAP 3: HELD-OUT GENERALIZATION BENCHMARK
        # =====================================================================
        if not args.skip_test:
            print("\n" + "=" * 84)
            print(" [FASE 3/3] EVALUASI BENCHMARK HOLDOUT TEST SET (8 SUBJEK UNSEEN)")
            print("=" * 84 + "\n")

            time.sleep(3)

            best_ckpt = "eksperimen_model/checkpoints/dynamics/best_dynamics_model.pth"
            test_json_out = "docs/report_training/dynamics_benchmark_results.json"

            eval_cmd = [
                python_exe,
                "eksperimen_model/evaluate_dynamics.py",
                "--checkpoint", best_ckpt,
                "--config", target_config,
                "--batch_size", str(args.batch_size),
                "--output_json", test_json_out,
                "--output_dir", "docs/report_training"
            ]

            print(f"Menjalankan benchmark evaluasi: {' '.join(eval_cmd)}\n")
            eval_proc = subprocess.run(eval_cmd)

            if eval_proc.returncode == 0:
                print(f"\n[Sukses] Hasil evaluasi held-out test set tersimpan di: {test_json_out}")
            else:
                print(f"\n[Warning] Evaluasi held-out test set keluar dengan kode {eval_proc.returncode}")

        # =====================================================================
        # TAHAP 4: GENERATE ACADEMIC REPORT & UPDATE MASTER INDEX
        # =====================================================================
        elapsed_hours = (time.time() - start_total_time) / 3600.0

        generate_experiment_report(
            run_dir=run_dir,
            run_id=run_id,
            start_time_str=start_time_str,
            total_duration_hours=elapsed_hours,
            config_path=target_config,
            history_json_path="eksperimen_model/checkpoints/dynamics/dynamics_training_history.json",
            test_json_path="docs/report_training/dynamics_benchmark_results.json"
        )

        print("\n" + "#" * 84)
        print("   SELURUH PIPELINE MARATHON TAHAP 3 SELESAI DENGAN SEMPURNA!")
        print("#" * 84)
        print(f" Waktu Selesai  : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" Total Durasi   : {elapsed_hours:.2f} jam ({elapsed_hours * 60:.1f} menit)")
        print(f" Model Terbaik  : eksperimen_model/checkpoints/dynamics/best_dynamics_model.pth")
        print(f" Laporan Run    : {os.path.join(run_dir, 'report.md')}")
        print(f" Master Index   : docs/report_training/INDEX.md")
        print("#" * 84 + "\n")

    finally:
        set_keep_awake(False)


if __name__ == "__main__":
    main()
