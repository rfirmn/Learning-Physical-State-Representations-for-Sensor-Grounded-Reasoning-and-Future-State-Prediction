# Final Implementation Plan (Revisi Akademik): Pipeline Pemrosesan Data Sekuens Temporal (Tahap 3 — Dynamics Modeling)

Dokumen ini merupakan **rencana implementasi akademik lengkap dan definitif** untuk pra-pemrosesan data (*data preprocessing*), ekstraksi representasi fisik laten, dan pembentukan dataset deret waktu (*temporal sequence dataset*) guna melatih **Dynamics Model** (Tahap 3). 

Rencana ini merevisi draf sebelumnya secara mendasar dengan mengintegrasikan **15 poin evaluasi kritis akademik**: meluruskan nomenklatur benchmark MM-Fi, memperjelas silsilah (*provenance*) model, memperbaiki formulasi matematis *loss*, memperkenalkan *oracle decoder baseline* untuk probe pose, menjamin pelacakan *source frame ID* asli, dan memisahkan audit integritas saintifik dari tolak ukur rekayasa (*engineering benchmark*).

---

## 1. Landasan Konseptual & Koreksi Nomenklatur Ilmiah

### A. Terminologi Arsitektur: Frozen-Representation Latent Dynamics
- **Bukan Canonical JEPA:** Arsitektur canonical JEPA (I-JEPA, V-JEPA) melatih *context encoder* dan *target encoder* secara bersamaan melalui mekanisme *asymmetric/EMA target update* secara *end-to-end self-supervised*.
- **Definisi Presisi untuk Riset Ini:** Sistem yang dibangun pada Tahap 3 adalah **"Frozen-Representation Latent Dynamics Model"** *(JEPA-inspired latent predictive dynamics)*. Model mempelajari prediktor temporal di atas ruang *embedding* yang sudah dibekukan (*fixed feature space* $Z_t \in \mathbb{R}^{384}$), memprediksi evolusi dinamika langsung di ruang representasi sensor tanpa merekonstruksi point cloud radar mentah.

### B. Koreksi Nomenklatur Protokol MM-Fi
Dalam literatur resmi MM-Fi (Yang et al., NeurIPS 2023), istilah *Protocol* dan *Setting* memiliki arti yang berbeda dan tidak boleh dicampuradukkan:
- **Kategori Aksi (*Protocol*):**
  - **Protocol 1 (P1):** 14 aktivitas harian (*daily activities*, A01–A14).
  - **Protocol 2 (P2):** 13 latihan rehabilitasi (*rehab exercises*, A15–A27).
  - **Protocol 3 (P3):** Seluruh 27 aktivitas (A01–A27).
- **Strategi Pemisahan Data (*Setting*):**
  - **Setting 1 (S1):** *Random split* antar-frame.
  - **Setting 2 (S2):** *Cross-Subject split* (generalisasi subjek baru).
  - **Setting 3 (S3):** *Cross-Environment split* (generalisasi ruangan baru).

> [!IMPORTANT]
> **Definisi Protokol Penelitian Ini:**  
> Riset ini menggunakan **Protocol 3 (P3: Seluruh 27 Aksi)** dengan konfigurasi pemisahan **Leakage-Controlled Cross-Subject Split**.

---

## 2. Analisis Kritis Provenance Checkpoint & Strategi Split Subjek

### A. Audit Silsilah (*Provenance*) Model Av2 (`model_av2.pth`)
Untuk menjamin klaim evaluasi tidak gugur saat sidang/publikasi, riwayat data yang pernah dilihat oleh `model_av2.pth` pada Tahap 2 dianalisis secara transparan:
- Pada Tahap 2 (`mmfi_pose_best_tuned.yaml` / `laporan_teknis_training_dan_evaluasi_v2.md`):
  - **Subjek Latih (24 Subjek):** `S01, S02, S03, S06, S08, S09, S11, S12, S14, S15, S16, S19, S21, S23, S26, S27, S29, S30, S31, S32, S33, S37, S38, S39`
  - **Subjek Validasi (8 Subjek):** `S05, S10, S18, S20, S24, S28, S34, S35` (digunakan untuk *early stopping*)
  - **Subjek Uji Held-Out (8 Subjek):** `S04, S07, S13, S17, S22, S25, S36, S40` (hanya dievaluasi 1 kali pada benchmark akhir)

### B. Perbandingan terhadap Official MM-Fi Setting 2 (S2)
Dalam benchmark resmi MM-Fi S2:
- **Official Test Subjects (8 Subjek):** `S05, S10, S15, S20, S25, S30, S35, S40` (kelipatan 5).
- **Official Train Subjects (32 Subjek):** Seluruh subjek lainnya.

**Temuan Kritis:**  
Model Av2 yang sudah ada saat ini (`model_av2.pth`) **pernah melihat** `S15` dan `S30` pada data latihnya, serta menggunakan `S05, S10, S20, S35` untuk validasi. Maka:
1. Jika kita memaksakan evaluasi pada *Official S2 Test* (`S05..S40`), klaim *strict unseen-subject* gugur karena bobot encoder Av2 telah terpengaruh oleh subjek tersebut.
2. Sebaliknya, terhadap himpunan 8 subjek:
   $$\mathcal{S}_{\text{test}}^{\text{held-out}} = \{\text{S04, S07, S13, S17, S22, S25, S36, S40}\}$$
   Model Av2 **SAMA SEKALI BELUM PERNAH MELIHAT** subjek-subjek ini, baik saat pelatihan maupun validasi *early stopping*.

### C. Keputusan Metodologis: Protokol Split yang Diterapkan
Untuk mempertahankan bobot Model Av2 yang telah terlatih tanpa melanggar etika ilmiah:
- **Nama Protokol:** **Custom Stratified 60/20/20 Cross-Subject Protocol (P3)**.
- **Klaim Akademik:** **"Leakage-Controlled Experimental Protocol"** (bukan klaim absolut tanpa syarat).
- **Partisi Subjek Terkunci:**
  - **Train Split (24 Subjek):** `S01, S02, S03, S06, S08, S09, S11, S12, S14, S15, S16, S19, S21, S23, S26, S27, S29, S30, S31, S32, S33, S37, S38, S39`
  - **Validation Split (8 Subjek):** `S05, S10, S18, S20, S24, S28, S34, S35`
  - **Test Split (8 Subjek Held-Out):** `S04, S07, S13, S17, S22, S25, S36, S40`
- Seluruh komponen berikutnya (Dynamics Model, Normalizer, MLP Projector Tahap 4) **wajib menghormati partisi ini secara mutlak**.

---

## 3. Spesifikasi Jendela Temporal & Agregasi Statistik

### A. Konfigurasi Jendela Temporal Riset Ini
Berdasarkan frekuensi sampling MM-Fi sebesar **10 Hz (100 ms per frame)**:
- **Konteks Pengamatan Historis ($T_{\text{in}} = 16$ frame / $1{,}6$ detik):** Rentang waktu yang cukup untuk menangkap momentum, fase percepatan, dan arah pergerakan anggota tubuh.
- **Horison Prediksi Masa Depan ($T_{\text{out}} = 8$ frame / $0{,}8$ detik):** Horison prediksi jangka pendek-menengah (*short-to-medium term dynamic forecast*) yang realistis sebelum akumulasi entropi gerak membesar.
- **Langkah Pergeseran (*Stride*):**
  - $\text{stride} = 2$ frame pada data latih (augmentasi pergeseran temporal).
  - $\text{stride} = 4$ atau $8$ frame pada evaluasi.
- **Justifikasi *Continuous Action Windowing*:**  
  Riset ini memodelkan dinamika pergerakan manusia kontinu di dalam rentang rekaman aksi (~27–30 detik per berkas), bukan mengklasifikasikan repetisi diskret.

### B. Penanganan Korelasi Sampel (*Correlated Windows vs. Independent Samples*)
Karena penggunaan *sliding window* menghasilkan jendela-jendela yang saling tumpang tindih (*overlapping*), **jendela-jendela tersebut tidak boleh dilaporkan sebagai observasi independen**.
- **Protokol Agregasi Pelaporan Ilmiah:**
  $$\text{Window-Level Metric} \longrightarrow \text{Action-Level Aggregation} \longrightarrow \text{Subject-Level Reporting}$$
- Metrik pengujian utama akan dilaporkan dalam bentuk **Mean $\pm$ Standard Deviation across Subjects** serta **95% Bootstrap Confidence Intervals**.

---

## 4. Integritas Data Frame: Pelacakan `source_frame_ids`

Untuk mencegah *false continuity bug*:
- Point cloud filtering (`sample_or_pad_points`) hanya membuang titik-titik sentinel di dalam satu frame, **bukan membuang berkas frame**.
- Namun, jika terdapat frame yang hilang (*missing frame*, misal frame 101, 102, 104), indeks frame tidak boleh di-reset menjadi array artifisial `[1, 2, 3]`.
- Setiap berkas cache `.pt` **wajib menyimpan `source_frame_ids` asli** yang diparsing langsung dari nama berkas (`frame(\d+).bin`).
- Pengecekan kontinuitas pada dataset loader dilakukan dengan:
  $$\Delta f_k = \text{source\_frame\_ids}[k+1] - \text{source\_frame\_ids}[k] == 1$$
  Jika $\Delta f_k \neq 1$, jendela sekuens otomatis diputus dan dimulai ulang dari titik diskontinuitas.

---

## 5. Formulasi Matematis Fungsi Loss & Regularisasi

### A. Latent Mean Squared Error yang Tepat
Untuk representasi laten berdimensi $D = 384$ sepanjang horison prediksi $T_{\text{out}}$:
$$\mathcal{L}_{\text{MSE}} = \frac{1}{T_{\text{out}}} \sum_{k=1}^{T_{\text{out}}} \left[ \frac{1}{D} \sum_{d=1}^{D} \left(\hat{Z}_{t+k, d} - Z_{t+k, d}\right)^2 \right] = \frac{1}{T_{\text{out}}} \sum_{k=1}^{T_{\text{out}}} \frac{1}{D} \|\hat{Z}_{t+k} - Z_{t+k}\|_2^2$$

### B. Total Fungsi Loss Pelatihan Dynamics Model
$$\mathcal{L}_{\text{dynamics}} = \frac{1}{T_{\text{out}}} \sum_{k=1}^{T_{\text{out}}} \left[ \frac{1}{D} \|\hat{Z}_{t+k} - Z_{t+k}\|_2^2 + \lambda_{\text{cos}} \left(1 - \frac{\hat{Z}_{t+k} \cdot Z_{t+k}}{\|\hat{Z}_{t+k}\|_2 \|Z_{t+k}\|_2}\right) \right]$$
Formula ini konsisten secara matematis dengan implementasi `F.mse_loss(pred, target) + lambda_cos * (1 - F.cosine_similarity(pred, target)).mean()`.

### C. Karakteristik Augmentasi *Latent Perturbation Noise*
- *Gaussian Noise* berskala kecil ($\sigma = 0{,}01$) diinjeksikan pada `hist_z` **hanya pada subset Train** dan **hanya setelah normalisasi standar (*standardized latent space*)**.
- **Rasional Ilmiah:** Bertindak sebagai *latent perturbation robustness* (regularisasi terhadap *jitter* sinyal sensor), bukan untuk akumulasi error autoregresif, karena model menggunakan arsitektur *direct multi-horizon forecasting*.
- Ablasi parameter disiapkan: $\sigma \in \{0{,}0, 0{,}01, 0{,}05\}$.

---

## 6. Protokol Evaluasi Probe Skeleton 3D & Oracle Decoder Baseline

### A. Formulasi Oracle Decoder Baseline
Supervisi skeleton 3D **tidak dijadikan target optimasi saat pelatihan Dynamics Model** untuk menjaga komputasi tetap fokus pada ruang laten. Namun, untuk membuktikan bahwa $\hat{Z}_{t+k}$ mempertahankan struktur fisik tubuh, digunakan probe pasif dengan *frozen* Pose Head Av2.

Untuk memisahkan degradasi akibat model dinamika vs keterbatasan representasi encoder awal, wajib dievaluasi dua metrik komparatif:
1. **Oracle Representation Error (Batas Atas Kualitas Av2):**
   $$S_{\text{oracle}, t+k} = \text{PoseHead}_{\text{frozen}}(Z_{\text{true}, t+k})$$
   $$\text{MPJPE}_{\text{oracle}} = \frac{1}{17} \sum_{j=1}^{17} \|S_{\text{oracle}, t+k}^{(j)} - S_{\text{GT}, t+k}^{(j)}\|_2$$
2. **Dynamics Prediction Error:**
   $$\hat{S}_{t+k} = \text{PoseHead}_{\text{frozen}}\left(\hat{Z}_{\text{denorm}, t+k}\right)$$
   $$\text{MPJPE}_{\text{pred}} = \frac{1}{17} \sum_{j=1}^{17} \|\hat{S}_{t+k}^{(j)} - S_{\text{GT}, t+k}^{(j)}\|_2$$
3. **Net Dynamics Error Degradation:**
   $$\Delta_{\text{dynamics}} = \text{MPJPE}_{\text{pred}} - \text{MPJPE}_{\text{oracle}}$$
   *Interpretasi:* Jika $\Delta_{\text{dynamics}}$ kecil (misal < 15–25 mm), terbukti bahwa model dinamika berhasil meramalkan masa depan dengan akurasi tinggi mendekati representasi asli.

### B. Invers Normalisasi Sebelum Probe Pose Head
Karena Pose Head dari Model Av2 dilatih pada ruang laten asli (tanpa standardisasi z-score temporal), vektor prediksi yang telah dinormalisasi harus dikembalikan ke ruang aslinya sebelum diumpankan ke Pose Head:
$$\hat{Z}_{\text{denorm}, t+k} = \hat{Z}'_{t+k} \odot \sigma_{\text{train}} + \mu_{\text{train}}$$

### C. Taksonomi Metrik MPJPE yang Dilaporkan
1. **Root-Relative MPJPE (mm):** Koordinat sendi dihitung relatif terhadap sendi Pelvis ($j=0$), mengukur akurasi konfigurasi postur tubuh tanpa terdistorsi pergeseran posisi global.
2. **Global Pelvis Trajectory Error (mm):** Error posisi sendi pelvis, mengukur kualitas estimasi pergerakan/translasi global subjek di dalam ruangan.
3. **Procrustes-Aligned MPJPE (PA-MPJPE, mm):** Akurasi bentuk rigid setelah penyelarasan rotasi dan translasi optimal via SVD.

---

## 7. Transparansi Data Attrition & Frame Provenance

Untuk menjawab perbedaan antara total frame sinkron MM-Fi (~320.000 frame) dan total frame mmWave aktif di dataset filtered:
- MM-Fi mencakup 4 modalitas (WiFi, mmWave, LiDAR, RGB). Sebagian sesi memiliki *packet loss* atau *sensor buffer timeout* pada modul radar TI IWR6843AOP.
- Folder `filtered_mmwave/` memuat tepat **1.080 sesi rekaman aksi valid** (dari total teoritis $4 \text{ env} \times 40 \text{ sub} \times 27 \text{ act} = 4.320$ kombinasi, MM-Fi hanya menjadwalkan 1.080 sesi rekaman resmi).
- Total frame radar aktual yang berhasil diindeks: **262.297 frame**.
- Rata-rata frame per sesi: $262.297 / 1.080 \approx 242{,}8$ frame ($\approx 24{,}3$ detik per aksi).

Tabel ringkasan data attrition akan dimasukkan ke laporan akhir:
| Tahapan Pemrosesan | Jumlah Frame | Keterangan |
| :--- | :---: | :--- |
| **Raw Synchronized Frames MM-Fi** | ~320.000 | Seluruh modalitas gabungan |
| **Filtered mmWave Binary Frames** | 262.297 | Frame radar 5 kanal yang tersedia di `filtered_mmwave/` |
| **Cleaned Valid Radar Frames** | 262.297 | Sentinel filtered (titik non-finite dibersihkan, 0 frame hilang) |
| **Final Extracted Latent Vectors ($Z_t$)**| 262.297 | Tersimpan dalam cache terpartisi |

---

## 8. Pemisahan Audit: Scientific Integrity vs. Engineering Benchmark

Audit otomatis dipisahkan secara tegas agar tidak mencampuradukkan validitas ilmiah dengan performa perangkat keras:

### A. Scientific & Data Integrity Audit (Penentu Keabsahan Metodologis)
1. **Mathematical Subject Disjointness:** $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
2. **Physical Disk Partition Audit:** Tidak ada berkas subjek uji yang berada di folder latih/validasi.
3. **Action Boundary Isolation:** Jendela sekuens tidak pernah melintasi batas rekaman ($E_{xx}, S_{xx}, A_{xx}$).
4. **Temporal Contiguity Verification:** Seluruh frame di dalam jendela memiliki delta indeks asli $\Delta f = 1$.
5. **Numerical Finiteness & Dimension Check:** Bebas dari NaN/Inf, dimensi tepat ($T_{\text{in}} \times 384$ dan $T_{\text{out}} \times 384$).
6. **Normalization Provenance:** Statistik $\mu_{\text{train}}$ dan $\sigma_{\text{train}}$ terbukti dihitung eksklusif dari folder `train/`.

### B. Engineering & System Benchmark (Metrik Efisiensi Komputasi)
1. DataLoader throughput (sampel/detik).
2. Pemanfaatan VRAM GPU selama ekstraksi offline.
3. Total jejak penyimpanan disk cache ($\approx 480$ MB).

---

## 9. Rencana Modifikasi Berkas & Komponen

### Komponen 1: Konfigurasi Master
#### [MODIFY] [mmfi_dynamics_v3.yaml](file:///c:/Users/Rio%20Aslab/Documents/pemrograman/Tugas_Akhir/eksperimen_model/configs/mmfi_dynamics_v3.yaml)
- Memperbarui metadata eksperimen: penamaan protokol P3 + Cross-Subject, penambahan parameter ablasi noise, parameter inverse normalisasi, dan bobot loss cosine.

### Komponen 2: Engine Ekstraksi Offline
#### [MODIFY] [extract_physical_features.py](file:///c:/Users/Rio%20Aslab/Documents/pemrograman/Tugas_Akhir/eksperimen_model/datasets/extract_physical_features.py)
- Menyimpan kunci `source_frame_ids` secara eksplisit dari integer nama berkas asli.
- Menyimpan `original_filenames` untuk penelusuran data mentah.
- Memastikan `normalization_stats.pt` memuat `mean_z` dan `std_z` yang siap untuk invers transformasi.

### Komponen 3: PyTorch Dataset Deret Waktu
#### [MODIFY] [temporal_dataset.py](file:///c:/Users/Rio%20Aslab/Documents/pemrograman/Tugas_Akhir/eksperimen_model/datasets/temporal_dataset.py)
- Memperbaiki dokumentasi bentuk tensor: membedakan bentuk sampel `__getitem__` (`(T_in, 384)`) dari bentuk batched DataLoader (`(B, T_in, 384)`).
- Menambahkan method `denormalize_z(z_tensor)` untuk mendukung evaluasi probe pose head.
- Memastikan noise augmentasi diterapkan setelah standardisasi.

### Komponen 4: Script Audit Ilmiah Independen
#### [MODIFY] [verify_temporal_data.py](file:///c:/Users/Rio%20Aslab/Documents/pemrograman/Tugas_Akhir/eksperimen_model/verify_temporal_data.py)
- Memisahkan laporan menjadi 2 seksi terpisah: **SECTION 1: SCIENTIFIC INTEGRITY AUDIT** dan **SECTION 2: ENGINEERING BENCHMARKS**.
- Menambahkan audit verifikasi `source_frame_ids` asli.

---

## 10. Verification Plan

### Langkah 1: Uji Audit Konseptual & Data (Dry-Run Check)
```powershell
& ".venv\Scripts\python.exe" eksperimen_model/verify_temporal_data.py --config eksperimen_model/configs/mmfi_dynamics_v3.yaml
```
*Target:* Lulus seluruh 6 tes integritas saintifik.

### Langkah 2: Ekstraksi Fitur Penuh (Full Dataset Caching)
```powershell
& ".venv\Scripts\python.exe" eksperimen_model/datasets/extract_physical_features.py --config eksperimen_model/configs/mmfi_dynamics_v3.yaml --batch_size 128
```
*Target:* 1.080 aksi berhasil diekstrak dan tersimpan rapi di `datasets/MM-Fi_features_v2/{train,val,test}`.
