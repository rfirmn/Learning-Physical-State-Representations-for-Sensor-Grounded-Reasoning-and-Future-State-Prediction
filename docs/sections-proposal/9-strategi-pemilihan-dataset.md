9. Strategi pemilihan dataset

Penelitian berfokus pada kedalaman investigasi representasi satu modalitas sensorik yang terukur (*depth over breadth*), alih-alih menyebarkan fokus pada banyak dataset yang superficial.

### Alur Strategi Dataset:

```text
ShapeNet 3D CAD (Point-MAE Pretraining)
   ↓ (Inisialisasi Bobot Geometri 3D)
MM-Fi Dataset — Protocol 3 (mmWave Radar)
   ↓ (Adaptasi Domain Sensor & Estimasi Status Fisik)
Dinamika Temporal & Prediksi Masa Depan
   ↓ (Sekuens Laten Z_t-k:t -> Z_t+1:t+h)
Grounded Reasoning & Cognitive Alignment
```

### Rasionalisasi Ilmiah:
1. **Inisialisasi Geometri yang Kuat:** Menggunakan bobot *foundation model* Point-MAE yang dilatih pada ShapeNet CAD memberikan *spatial prior* 3D yang kaya, mengatasi kendala kelangkaan label spasial pada domain radar mmWave.
2. **Karakteristik MM-Fi Protocol 3:** MM-Fi (NeurIPS 2023) menyediakan 262.297 frame point cloud radar mmWave sinkron dengan ground truth 3D skeleton dari 40 subjek, 4 lingkungan berbeda (*cross-environment* E01–E04), dan 27 variasi aksi.
3. **Protokol Evaluasi Cross-Subject yang Ketat:** Data dibagi secara independen per subjek:
   - **Training Set (24 Subjek):** Pembelajaran representasi dan model dinamika.
   - **Validation Set (8 Subjek: S05, S10, S18, S20, S24, S28, S34, S35):** Tuning hyperparameter dan *early stopping*.
   - **Held-Out Test Set (8 Subjek: S03, S08, S12, S15, S23, S27, S31, S39):** Evaluasi ketahanan generalisasi pada orang yang belum pernah dilihat model sama sekali.

Satu dataset multimodal berskala besar yang dieksplorasi secara mendalam dengan protokol *cross-subject* terisolasi memberikan validitas ilmiah yang jauh lebih kokoh dibandingkan mengombinasikan dataset yang tidak terstandarisasi.
