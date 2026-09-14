# AGENTS.md — Panduan Instruksi & SOP untuk Agen AI & Tim Peneliti

Selamat datang di repositori penelitian Tugas Akhir:
> **Learning Physical State Representations for Sensor-Grounded Reasoning and Future-State Prediction**  
> *(Pemodelan Representasi Fisik Spasial-Temporal dan Penyelarasan LLM Berbasis Sinyal Radar mmWave)*

Dokumen ini adalah **panduan instruksi operasional wajib** bagi siapa pun (asisten AI, pengembang baru, kolaborator riset, atau mahasiswa) yang bekerja pada repositori ini. Bacalah dokumen ini secara seksama sebelum menulis kode atau menjalankan eksperimen.

---

## 1. Visi & Arsitektur Utama Sistem

Prinsip fundamental riset ini adalah **pemisahan tegas (*decoupling*) antara modul persepsi fisik dan modul kognitif bahasa**:

```text
mmWave Radar Mentah (x, y, z, vd, SNR)
         │
         ▼
[Sensor Encoder (Point-MAE)]  <-- Fase 1 & 2 (Tahap saat ini)
         │
         ▼ Physical State (Z_t: 384d)
[Dynamics Model (Temporal Transformer / GRU)]  <-- Fase 3
         │
         ▼ Future State Prediction (Z_t+1:t+h)
[MLP Cross-Modal Projector]  <-- Fase 4
         │
         ▼ Pseudo-tokens Fisik
[Frozen LLM (Llama / Qwen)]  <-- Fase 4
         │
         ▼
Penalaran Bahasa Alami (Spatial, Temporal, Counterfactual Reasoning)
```

> [!IMPORTANT]
> **Aturan Arsitektur Penting:**
> 1. **LLM TIDAK PERNAH DILATIH DARI AWAL** dan tetap dalam kondisi **FROZEN** (hanya proyektor MLP yang dilatih di Tahap 4).
> 2. **Encoder dibekukan (frozen)** setelah Tahap 2 selesai (*Model Av2*) sebelum masuk ke Tahap 3.
> 3. **Dynamics Model dibekukan (frozen)** setelah Tahap 3 selesai sebelum masuk ke Tahap 4.

---

## 2. Peta Kurikulum Pelatihan (4 Tahapan)

| Tahap | Fokus | Input | Output / Target | Status |
| :---: | :--- | :--- | :--- | :---: |
| **Tahap 1** | Inisialisasi Bobot 3D | ShapeNet CAD | Bobot Point-MAE Transformer Encoder (**Model A**) | **Selesai** ([models/Point-MAE/pretrain.pth](models/Point-MAE/pretrain.pth)) |
| **Tahap 2** | Adaptasi Domain Radar & Pose Estimation | Point Cloud Radar MM-Fi ($N=128$) | Estimasi 17 Joint 3D Skeleton (**Model Av2**) | **Aktif** ([eksperimen_model/train_pose.py](eksperimen_model/train_pose.py)) |
| **Tahap 3** | Pemodelan Dinamika Temporal | Sekuens Status ($Z_{t-k \dots t}$) | Prediksi Masa Depan ($Z_{t+1 \dots t+h}$) | *Next Stage* |
| **Tahap 4** | Penyelarasan Kognitif ke LLM | Vektor $Z_t$ & $Z_{t+1:t+h}$ | *Pseudo-tokens* untuk penalaran LLM | *Final Stage* |

---

## 3. Peta Direktori Workspace

```text
Tugas_Akhir/
├── AGENTS.md                         <-- [Anda berada di sini] Panduan operasional agen & staff
├── requirements.txt                  <-- Spesifikasi dependensi PyTorch CUDA, uv, & libraries
├── .venv/                            <-- Virtual environment Python 3.12 (dikelola oleh uv)
├── models/
│   └── Point-MAE/
│       └── pretrain.pth              <-- Bobot pretrained foundation model ShapeNet (348 MB)
├── datasets/
│   └── MM-Fi Dataset/
│       ├── MMFi_action_segments.csv  <-- 1.082 batas repetisi gerakan frame awal-akhir
│       └── filtered_mmwave/          <-- 262.297 frame radar .bin + 1.080 file ground_truth.npy
│           └── E01..E04/S01..S40/A01..A27/
│               ├── frame001.bin ... frameXXX.bin
│               └── ground_truth.npy  <-- GT 3D Skeleton (F, 17, 3) dalam satuan meter
├── docs/
│   ├── onboarding.md                 <-- Pengenalan konseptual riset
│   ├── eksperimen_model.md           <-- Rancangan teknis 4 tahap kurikulum
│   ├── laporan_eksperimen_1.md       <-- Analisis detail evaluasi eksperimen pertama
│   ├── sections-proposal/            <-- 35 file modular naskah proposal
│   └── report_training/              <-- FOLDER OTOMATISASI LAPORAN PELATIHAN
│       ├── INDEX.md                  <-- Master tabel pembanding seluruh eksperimen
│       └── RUN_YYYYMMDD_HHMMSS_.../  <-- Folder otomatis hasil setiap kali training selesai
│           ├── report.md
│           ├── loss_curve.png
│           ├── per_joint_error.png
│           ├── skeleton_3d_comparison.png
│           ├── metrics.json
│           └── config_snapshot.yaml
└── eksperimen_model/                 <-- KODE IMPLEMENTASI PYTORCH
    ├── configs/                      <-- File konfigurasi YAML (pose, mae, classifier)
    ├── datasets/                     <-- Custom Dataset (mmfi_dataset.py) & transforms.py
    ├── models/                       <-- Pure PyTorch Point-MAE, Encoder, dan Loss functions
    ├── utils/                        <-- Checkpoint loader, logger, dan ExperimentReporter
    ├── test_pipeline.py              <-- Script sanity check otomatis
    ├── train_pose.py                 <-- Script training Tahap 2 (3D Pose Estimation)
    ├── evaluate_pose.py              <-- Script evaluasi detail error 17 sendi tubuh
    └── train_mae.py                  <-- Script self-supervised domain adaptation
```

---

## 4. Konfigurasi Lingkungan & Hardware

### Hardware yang Digunakan:
- **GPU:** NVIDIA GeForce RTX 3060 (12 GB GDDR6 VRAM, CUDA 12.4 / 12.6, Driver 560.94)
- **CPU:** AMD Ryzen 7 7700 (8 Core, 16 Thread)
- **RAM:** 16 GB DDR5
- **Storage:** Drive C: (>140 GB Free)

### Cara Menjalankan Perintah (Wajib Menggunakan `.venv` atau `uv`):
Seluruh eksekusi Python harus menggunakan interpreter lingkungan virtual:
- Di PowerShell:
  ```powershell
  & ".venv\Scripts\python.exe" <script.py> [argumen]
  ```
  atau
  ```powershell
  uv run python <script.py> [argumen]
  ```

---

## 5. Standar Operasional Prosedur (SOP) untuk Agen & Staff

### Aturan 1: Wajib Menjalankan Sanity Check Sebelum Training Skala Penuh
Sebelum memulai training berdurasi panjang, selalu jalankan script sanity check untuk memastikan data, GPU, dan alur komputasi gradien tidak mengalami *bug*:
```powershell
& ".venv\Scripts\python.exe" eksperimen_model/test_pipeline.py
```

### Aturan 2: Pertahankan Implementasi Pure-PyTorch
Modul Point-MAE pada proyek ini sengaja dirancang menggunakan **Pure-PyTorch** (`farthest_point_sampling`, `knn_group`, `ChamferDistanceLoss`) untuk menghindari kompilasi ekstensi C++/CUDA eksternal (`pointnet2_ops`, `knn_cuda`) yang sering gagal di lingkungan Windows. Jangan mengganti modul ini dengan dependensi kompilasi native tanpa persetujuan.

### Aturan 3: Otomatisasi Laporan di `docs/report_training/`
Setiap training yang dijalankan melalui `train_pose.py` **otomatis menghasilkan folder laporan baru** di `docs/report_training/RUN_.../` dan memperbarui `docs/report_training/INDEX.md`.
- **DILARANG** menghapus folder di `docs/report_training/` kecuali eksperimen tersebut berstatus gagal (*corrupted*).
- Setiap kali Anda selesai melatih model, selalu cek tautan laporannya di [docs/report_training/INDEX.md](docs/report_training/INDEX.md).

### Aturan 4: Ground Truth Fisik Tidak Boleh Dimanipulasi
- File `ground_truth.npy` di setiap subfolder aksi memuat array NumPy berdimensi `(F, 17, 3)` dalam satuan **meter**.
- Indeks frame $k$ (dari `frameXXX.bin`) berkorespondensi langsung dengan `gt[k - 1]`.
- Metrik ilmiah utama untuk evaluasi Tahap 2 adalah **MPJPE (Mean Per-Joint Position Error)** dalam meter/milimeter.

---

## 6. Cheat Sheet Perintah Penting

### 1. Menjalankan Sanity Check Pipeline:
```powershell
& ".venv\Scripts\python.exe" eksperimen_model/test_pipeline.py
```

### 2. Menjalankan Training Estimasi 3D Pose (Tahap 2):
- **Otomatis Penuh (Tuning + Full Training Unattended, PC Tetap Terjaga):**
  ```powershell
  & ".venv\Scripts\python.exe" eksperimen_model/run_autotune_and_train.py --n_trials 5 --tuning_epochs 8 --full_epochs 100 --batch_size 128
  ```
- **Training Model v2 Teroptimasi Langsung (60–100 Epochs):**
  ```powershell
  & ".venv\Scripts\python.exe" eksperimen_model/train_pose_v2.py --config eksperimen_model/configs/mmfi_pose_v2.yaml --epochs 100
  ```
- **Hyperparameter Tuning Mandiri:**
  ```powershell
  & ".venv\Scripts\python.exe" eksperimen_model/tune_pose.py --n_trials 6 --epochs_per_trial 5
  ```
- **Training Model v1 Baseline (Legacy):**
  ```powershell
  & ".venv\Scripts\python.exe" eksperimen_model/train_pose.py --epochs 30 --batch_size 32
  ```

### 3. Mengevaluasi Checkpoint Model per Sendi:
```powershell
& ".venv\Scripts\python.exe" eksperimen_model/evaluate_pose.py --checkpoint eksperimen_model/checkpoints/pose_estimation_v2/best_model.pth --config eksperimen_model/configs/mmfi_pose_v2.yaml --split val
```

### 4. Menjalankan Domain Adaptation Self-Supervised (MAE):
```powershell
& ".venv\Scripts\python.exe" eksperimen_model/train_mae.py --epochs 20 --batch_size 32
```

---

## 7. Tugas Berikutnya (*Next Milestones*)

Jika Anda baru ditugaskan ke repositori ini, berikut adalah prioritas langkah selanjutnya:

1. **Memaksimalkan Konvergensi Tahap 2:**
   - Jalankan `train_pose.py` selama 20–30 epoch pada seluruh subjek latih (`S01`–`S30`) untuk menekan MPJPE validasi mendekati akurasi sentimeter.
   - Pastikan checkpoint terbaik tersimpan sebagai `model_av2.pth`.
2. **Memulai Tahap 3 (Dynamics Modeling):**
   - Rancang modul `DynamicsModel` di `eksperimen_model/models/dynamics_model.py` (Temporal Transformer atau GRU).
   - Bekukan (*freeze*) bobot `model_av2.pth`.
   - Gunakan urutan status fisik historis ($Z_{t-k \dots t}$) yang diiris berdasarkan [datasets/MM-Fi Dataset/MMFi_action_segments.csv](datasets/MM-Fi%20Dataset/MMFi_action_segments.csv) untuk memprediksi status masa depan ($Z_{t+1 \dots t+h}$).
3. **Memulai Tahap 4 (Penyelarasan LLM):**
   - Buat proyektor linear `PhysicalToLLMAlignmentMLP` (`Linear(384, 1024) -> GELU -> Linear(1024, LLM_DIM)`).
   - Sambungkan ke LLM *frozen* (misal Qwen-2.5-3B atau Llama-3.2-3B).
