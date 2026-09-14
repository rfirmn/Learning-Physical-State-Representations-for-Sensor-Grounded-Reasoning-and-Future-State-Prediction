"""
Script Pipeline Otomatis (Unattended Execution):
1. Menjalankan Hyperparameter Tuning (Pencarian parameter optimal)
2. Memilih konfigurasi terbaik berdasarkan Validation MPJPE
3. Menyimpan konfigurasi pemenang ke eksperimen_model/configs/mmfi_pose_best_tuned.yaml
4. Langsung melatih model secara penuh (Full Training) hingga konvergen
5. Mencegah Windows masuk ke Sleep Mode selama proses berlangsung
6. Menyimpan checkpoint model_av2.pth dan menghasilkan laporan otomatis lengkap.
"""

import os
import sys
import time
import ctypes
import argparse
import subprocess
import datetime

# Prevent Windows from going to sleep while training is running
def set_keep_awake(enable: bool = True):
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
    parser = argparse.ArgumentParser(description="End-to-End Automated Tuning & Full Training Pipeline")
    parser.add_argument("--base_config", type=str, default="eksperimen_model/configs/mmfi_pose_v2.yaml",
                        help="Path ke base configuration file")
    parser.add_argument("--n_trials", type=int, default=5,
                        help="Jumlah kombinasi hyperparameter yang diuji (default: 5)")
    parser.add_argument("--tuning_epochs", type=int, default=8,
                        help="Jumlah epoch per trial tuning (default: 8)")
    parser.add_argument("--full_epochs", type=int, default=100,
                        help="Jumlah epoch untuk full training setelah parameter terbaik ditemukan (default: 100)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size (default: 128)")
    parser.add_argument("--env", type=str, nargs="+", default=None,
                        help="Filter target environment (misal: E01)")
    parser.add_argument("--train_sub", type=str, nargs="+", default=None,
                        help="Filter train subjects (misal: S01 S02)")
    parser.add_argument("--val_sub", type=str, nargs="+", default=None,
                        help="Filter val subjects (misal: S03)")
    parser.add_argument("--max_train_batches", type=int, default=None,
                        help="Batas batch train per epoch (untuk testing cepat)")
    parser.add_argument("--max_val_batches", type=int, default=None,
                        help="Batas batch val per epoch (untuk testing cepat)")
    parser.add_argument("--skip_tuning", action="store_true",
                        help="Lewati tahap tuning jika konfigurasi terbaik sudah ada")
    parser.add_argument("--output_config", type=str, default="eksperimen_model/configs/mmfi_pose_best_tuned.yaml",
                        help="Path penyimpanan file konfigurasi hasil tuning")
    return parser.parse_args()


def main():
    args = parse_args()
    python_exe = sys.executable

    start_total_time = time.time()
    set_keep_awake(True)

    print("\n" + "#" * 80)
    print("   AUTONOMOUS PIPELINE: HYPERPARAMETER TUNING -> FULL TRAINING")
    print("   Dirancang untuk dijalankan tanpa pengawasan (Unattended Execution)")
    print("#" * 80)
    print(f" Waktu Mulai        : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Python Interpreter : {python_exe}")
    print(f" Base Config        : {args.base_config}")
    print(f" Tuning Setup       : {args.n_trials} trials x {args.tuning_epochs} epochs")
    print(f" Full Training Setup: {args.full_epochs} epochs")
    print(f" Batch Size         : {args.batch_size}")
    print("#" * 80 + "\n")

    try:
        # =====================================================================
        # TAHAP 1: HYPERPARAMETER TUNING
        # =====================================================================
        if not args.skip_tuning:
            print("=" * 80)
            print(" [TAHAP 1/2] MEMULAI HYPERPARAMETER TUNING OTOMATIS")
            print("=" * 80)

            tune_cmd = [
                python_exe,
                "eksperimen_model/tune_pose.py",
                "--base_config", args.base_config,
                "--n_trials", str(args.n_trials),
                "--epochs_per_trial", str(args.tuning_epochs),
                "--batch_size", str(args.batch_size),
                "--output_config", args.output_config
            ]

            if args.env:
                tune_cmd.extend(["--env"] + args.env)
            if args.train_sub:
                tune_cmd.extend(["--train_sub"] + args.train_sub)
            if args.val_sub:
                tune_cmd.extend(["--val_sub"] + args.val_sub)
            if args.max_train_batches:
                tune_cmd.extend(["--max_train_batches", str(args.max_train_batches)])
            if args.max_val_batches:
                tune_cmd.extend(["--max_val_batches", str(args.max_val_batches)])

            print(f"Menjalankan perintah: {' '.join(tune_cmd)}\n")
            tune_proc = subprocess.run(tune_cmd)

            if tune_proc.returncode != 0:
                print(f"\n[Error] Hyperparameter tuning gagal dengan exit code {tune_proc.returncode}!")
                print("Menggunakan base config untuk melanjutkan full training...")
                target_config = args.base_config
            else:
                print(f"\n[Sukses] Tuning selesai! Konfigurasi terbaik tersimpan di: {args.output_config}")
                target_config = args.output_config
        else:
            print("[Info] Tahap tuning dilewati (--skip_tuning aktif).")
            target_config = args.output_config if os.path.exists(args.output_config) else args.base_config

        # =====================================================================
        # TAHAP 2: FULL TRAINING MODEL V2
        # =====================================================================
        print("\n" + "=" * 80)
        print(" [TAHAP 2/2] MEMULAI FULL TRAINING DENGAN KONFIGURASI TERBAIK")
        print(f" Target Config: {target_config}")
        print(f" Target Epochs: {args.full_epochs}")
        print("=" * 80 + "\n")

        # Berikan jeda 5 detik agar sistem I/O & VRAM bersih sempurna
        time.sleep(5)

        train_cmd = [
            python_exe,
            "eksperimen_model/train_pose_v2.py",
            "--config", target_config,
            "--epochs", str(args.full_epochs),
            "--batch_size", str(args.batch_size)
        ]

        if args.env:
            train_cmd.extend(["--env"] + args.env)
        if args.train_sub:
            train_cmd.extend(["--train_sub"] + args.train_sub)
        if args.val_sub:
            train_cmd.extend(["--val_sub"] + args.val_sub)
        if args.max_train_batches:
            train_cmd.extend(["--max_train_batches", str(args.max_train_batches)])
        if args.max_val_batches:
            train_cmd.extend(["--max_val_batches", str(args.max_val_batches)])

        print(f"Menjalankan perintah: {' '.join(train_cmd)}\n")
        train_proc = subprocess.run(train_cmd)

        if train_proc.returncode != 0:
            print(f"\n[Error] Full training gagal dengan exit code {train_proc.returncode}!")
            sys.exit(train_proc.returncode)

        elapsed_hours = (time.time() - start_total_time) / 3600.0

        print("\n" + "#" * 80)
        print("   SELURUH PIPELINE (TUNING + FULL TRAINING) SELESAI DENGAN SUKSES!")
        print("#" * 80)
        print(f" Waktu Selesai  : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" Total Durasi   : {elapsed_hours:.2f} jam")
        print(f" Checkpoint Av2 : eksperimen_model/checkpoints/pose_estimation_v2/model_av2.pth")
        print(f" Master Laporan : docs/report_training/INDEX.md")
        print("#" * 80 + "\n")

    finally:
        set_keep_awake(False)


if __name__ == "__main__":
    main()
