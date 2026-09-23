28. Tahapan kurikulum eksekusi penelitian

Penelitian dieksekusi secara terstruktur melalui kurikulum 4 tahap yang modular dan terisolasi:

### Tahap 1 — Inisialisasi Bobot Geometri 3D (Transfer Learning)
```text
ShapeNet CAD Point Clouds
          ↓
Point-MAE Transformer Pretraining
          ↓
Inisialisasi Bobot Encoder 3D (Model A)
```
* **Target:** Memperoleh *spatial prior* representasi geometri 3D yang kaya dari data CAD berskala besar sebelum mengolah sinyal radar yang memiliki rasio derau tinggi.

---

### Tahap 2 — Adaptasi Domain Radar & Estimasi Status Fisik (Physical State Estimation)
```text
Point Cloud mmWave Radar MM-Fi (N=128)
          ↓
Point-MAE Encoder (Av2)
          ↓
Estimasi 17 Joint 3D Skeleton (Regresi Pose)
```
* **Target:** Mengadaptasikan encoder ke domain sinyal radar MM-Fi dan memvalidasi bahwa representasi laten $Z_t \in \mathbb{R}^{384}$ berhasil mengabstraksikan struktur fisik tubuh subjek.
* **Aturan Pembekuan:** Setelah tahap ini tuntas, modul encoder dibekukan (*frozen*).

---

### Tahap 3 — Pemodelan Dinamika Temporal (Dynamics Modeling)
```text
Sekuens Status Laten Historis Z_{t-15:t} (1.6s @ 10 Hz)
          ↓
Dynamics Model (Temporal Transformer)
          ↓
Prediksi Status Laten Masa Depan Z_{t+1:t+8} (0.8s @ 10 Hz)
```
* **Target:** Membuktikan bahwa representasi sensorik memiliki kontinuitas temporal dan kapasitas prediktif masa depan yang mengungguli baseline persistensi statis.
* **Aturan Pembekuan:** Setelah tahap ini tuntas, model dinamika dibekukan (*frozen*).

---

### Tahap 4 — Penyelarasan Kognitif ke Frozen SLM & Uji Grounding (Cognitive Alignment)
```text
Token Fisik (Z_hist: 16 token, Z_future: 8 token)
          ↓
Two-Layer MLP Projector (~1.97M param)
          ↓
Frozen Small Language Model (Qwen2.5-1.5B-Instruct)
          ↓
Sensor-Grounded Physical Reasoning & Explanation
```
* **Target:** Melatih adapter proyektor untuk menerjemahkan status fisik menjadi *pseudo-tokens* bahasa tanpa mengubah bobot SLM.
* **Uji Grounding Ilmiah:** Melakukan *Sensor Shuffling Controls* (Cross-Action & Within-Action Shuffling) pada *held-out test set* untuk membuktikan secara empiris bahwa model bahasa bernalar berdasarkan sinyal fisik sensor, bukan halusinasi teks.
