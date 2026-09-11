# Panduan & Tutorial Lengkap Eksekusi Pelatihan Tahap 2
> **3D Human Skeleton Estimation via Point-MAE on MM-Fi Radar Dataset**  
> *Riset Tugas Akhir: Spatial-Temporal Physical State Representation for LLM Reasoning*

Dokumen ini memuat panduan langkah demi langkah untuk menjalankan pelatihan, memantau proses secara *real-time*, memahami deteksi *stall*, serta memeriksa laporan otomatis dan evaluasi ilmiah siap publikasi.

---

## 1. Ringkasan Perubahan Konfigurasi & Arsitektur

Pipeline pelatihan telah diperbarui secara menyeluruh sesuai dengan rancangan ilmiah yang disetujui:

1. **Stratified Random Cross-Subject Split (Seed: 42, Rasio 6:2:2)**:
   - **4 Environment** (E01, E02, E03, E04) masing-masing memuat 10 subjek unik.
   - **Data Latih (Train, 24 Subjek / 154.401 Frame)**:
     - E01: `S01`, `S02`, `S03`, `S06`, `S08`, `S09`
     - E02: `S11`, `S12`, `S14`, `S15`, `S16`, `S19`
     - E03: `S21`, `S23`, `S26`, `S27`, `S29`, `S30`
     - E04: `S31`, `S32`, `S33`, `S37`, `S38`, `S39`
   - **Data Validasi (Val, 8 Subjek / 55.024 Frame)**:
     - E01: `S05`, `S10` | E02: `S18`, `S20` | E03: `S24`, `S28` | E04: `S34`, `S35`
   - **Data Uji Akhir (Test, 8 Subjek / 52.872 Frame - Terisolasi Penuh)**:
     - E01: `S04`, `S07` | E02: `S13`, `S17` | E03: `S22`, `S25` | E04: `S36`, `S40`
2. **Leakage Guard Otomatis**:
   - `train_pose.py` akan langsung menghentikan eksekusi dengan pesan galat jika terdapat tumpang-tindih (*overlap*) antara subjek latih, validasi, dan uji.
3. **Pencatatan Performa Multi-Environment Per-Epoch**:
   - Validasi menghitung MPJPE keseluruhan serta MPJPE spesifik untuk masing-masing lingkungan (E01, E02, E03, E04) pada setiap epoch.

---

## 2. Langkah 1: Sanity Check Lingkungan & GPU

Sebelum memulai pelatihan berdurasi panjang, selalu jalankan verifikasi cepat (hanya butuh waktu ~3 detik):

```powershell
& ".venv\Scripts\python.exe" eksperimen_model/test_pipeline.py
```

**Hasil yang Diharapkan:**
- Seluruh 5 pengujian bertanda lolos (`ALL TESTS PASSED! PIPELINE IS READY FOR TRAINING`).
- CUDA terdeteksi pada NVIDIA GeForce RTX 3060.
- 156 *backbone layers* Point-MAE ShapeNet terpasang dengan benar.

---

## 3. Langkah 2: Menjalankan Pelatihan (Training)

### Opsi A: Pelatihan Skala Penuh (Rekomendasi Utama — 30 Epochs)
Perintah ini akan melatih model pada seluruh 24 subjek latih di 4 environment:
```powershell
& ".venv\Scripts\python.exe" eksperimen_model/train_pose.py --epochs 30 --batch_size 32
```

### Opsi B: Pelatihan Cepat / Verifikasi Awal (Contoh: 5 Epochs)
Jika Anda ingin memastikan seluruh siklus berjalan cepat sebelum melatih 30 epoch penuh:
```powershell
& ".venv\Scripts\python.exe" eksperimen_model/train_pose.py --epochs 5 --batch_size 32
```

### Opsi C: Pelatihan dengan Penyesuaian `num_workers` (Jika Disk I/O Lambat)
Jika proses *loading* data terasa berat atau terjadi *warning* stall:
```powershell
& ".venv\Scripts\python.exe" eksperimen_model/train_pose.py --epochs 30 --batch_size 32 --num_workers 2
```

---

## 4. Memahami Progress Bar & Fitur Deteksi Stall (*Stall Notice*)

### Tampilan Progress Bar Real-Time
Saat pelatihan berjalan, progress bar `tqdm` menampilkan informasi komprehensif:
```text
Epoch 01/30 [Train]: 100%|█████████████████████████| 4825/4825 [15:20<00:00, 5.24it/s, loss=0.1850m, mpjpe=185.0mm, avg=210.4mm, lr=2.0e-04, vram=3820MB]
```
- **`loss` & `mpjpe`**: Nilai MPJPE pada *batch* saat ini (dalam satuan meter dan milimeter).
- **`avg`**: Rerata MPJPE kumulatif pada epoch yang sedang berjalan.
- **`lr`**: *Learning rate* aktual yang diperbarui oleh *Cosine Annealing Scheduler*.
- **`vram`**: Pemakaian memori GPU secara *real-time* (biasanya ~3.8 GB dari 12 GB VRAM).

### Ringkasan Tiap Akhir Epoch
Di akhir setiap epoch, ringkasan performa per lingkungan akan tercetak:
```text
 >>> [Epoch 01/30] Train: 210.4mm (0.2104m) | Val: 118.2mm (0.1182m) | LR: 1.98e-04
     Environment Breakdown -> E01: 112.4mm | E02: 122.1mm | E03: 115.8mm | E04: 122.5mm
     🌟 NEW BEST MODEL! Val MPJPE: 0.1182 m (118.2 mm)
```

### Cara Kerja Watchdog & *Notice Jika Stack* (Stall Notice)
Pelatihan pada sistem operasi Windows yang membaca ribuan file biner radar kecil (`.bin`) rentan terhadap lonjakan latensi I/O disk atau *thread serialization*.

Kami telah menambahkan **`StallWatchdog`** yang berjalan di *background thread*:
1. Jika proses *batch* tidak selesai dalam waktu **>35 detik**, sebuah peringatan berbingkai akan muncul di layar:
   ```text
   ╔══════════════════════════════════════════════════════════════════════════════════════════╗
   ║ ⚠️  STALL NOTICE: Training loop has not completed a batch for 38.2s (> 35s threshold)     ║
   ╠══════════════════════════════════════════════════════════════════════════════════════════╣
   ║ Current Step: Train Epoch 01/30 | Batch 142/4825                                         ║
   ║ Possible Causes & Diagnostics:                                                           ║
   ║  1. MM-Fi Disk I/O: Reading radar .bin files may be throttled by disk read speed.       ║
   ║  2. Windows DataLoader Workers: Multi-worker IPC can occasionally serialize or pause.   ║
   ║     -> If this persists across batches, rerun with `--num_workers 0`.                    ║
   ║  3. GPU Kernel / VRAM: PyTorch CUDA execution or memory paging might be saturated.       ║
   ║ Action: Process is NOT terminated. Still waiting for batch completion...                 ║
   ╚══════════════════════════════════════════════════════════════════════════════════════════╝
   ```
2. **Proses TIDAK dihentikan (*NOT killed*)**: Watchdog hanya memberi peringatan diagnostik agar Anda tidak menebak-nebak apakah program hang atau masih membaca data.
3. Begitu pembacaan selesai, sistem otomatis mencetak:
   ```text
   >>> [STALL RECOVERED] Loop resumed activity after 40.1s delay. Continuing normal execution...
   ```

---

## 5. Memeriksa Laporan Otomatis (*Reporting Pipeline*)

Segera setelah seluruh epoch selesai, modul `ExperimentReporter` otomatis:
1. Menyimpan checkpoint terbaik sebagai:
   - `eksperimen_model/checkpoints/pose_estimation/best_model.pth`
   - `eksperimen_model/checkpoints/pose_estimation/model_av2.pth` (Bobot final Tahap 2 untuk Tahap 3)
2. Membuat folder laporan baru di:
   ```text
   docs/report_training/RUN_YYYYMMDD_HHMMSS_point_mae_mmfi_pose_estimation/
   ```
3. Menghasilkan **6 visualisasi ilmiah beresolusi tinggi (300 DPI)**:
   - `loss_curve.png`: Kurva konvergensi train vs val MPJPE beserta kurva tiap environment (E01–E04).
   - `per_joint_error.png`: Diagram batang error 17 sendi dengan pengkodean warna anatomi (*Central Axis*, *Lower Limbs*, *Upper Limbs*).
   - `env_comparison.png`: Diagram perbandingan antar-lingkungan beserta nilai *Environment Robustness Score* (ERS).
   - `joint_env_heatmap.png`: Heatmap matriks $17 \times 4$ error per sendi di tiap lingkungan (mm).
   - `error_distribution.png`: Histogram densitas error dan kurva kumulatif PCK.
   - `skeleton_3d_comparison.png`: Visualisasi perbandingan wireframe 3D skeleton prediksi terhadap ground truth.
4. Memperbarui tabel komparasi master pada:
   [docs/report_training/INDEX.md](file:///c:/Users/Rio%20Aslab/Documents/pemrograman/Tugas_Akhir/docs/report_training/INDEX.md)

---

## 6. Langkah 3: Evaluasi Ilmiah Lanjutan untuk Publikasi (Paper/Konferensi)

Setelah pelatihan selesai dan bobot `model_av2.pth` tersimpan, jalankan alat evaluasi berikut:

### 1. Evaluasi pada Test Set yang Belum Pernah Dilihat (Strictly Isolated Test Split)
Menghitung PA-MPJPE (Procrustes), PCK@30/50/100/150mm, N-MPJPE, per-kategori gerakan, dan 95% Confidence Interval:
```powershell
& ".venv\Scripts\python.exe" eksperimen_model/evaluate_pose.py --split test --output_json docs/test_benchmark_results.json
```

### 2. Analisis Representasi Status Fisik $Z_t \in \mathbb{R}^{384}$
Mengevaluasi apakah representasi $Z_t$ telah mengodekan semantik gerakan secara invariansi lingkungan (disentanglement) untuk persiapan Tahap 3 & 4:
```powershell
& ".venv\Scripts\python.exe" eksperimen_model/evaluate_embedding.py
```
*Output*: Visualisasi 2D t-SNE klaster gerakan (`tsne_actions.png`), t-SNE lingkungan (`tsne_environments.png`), akurasi linear probe classifier, dan *silhouette score*.

### 3. Uji Ketahanan Fisik (*Physical Robustness Testing*)
Menguji ketahanan model terhadap oklusi radar (point dropout) dan pantulan multipath (noise injection):
```powershell
& ".venv\Scripts\python.exe" eksperimen_model/evaluate_robustness.py
```
*Output*: Kurva degradasi performa siap publikasi (`robustness_curves.png`).
