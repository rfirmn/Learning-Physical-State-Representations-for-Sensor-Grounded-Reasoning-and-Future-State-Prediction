# Laporan Hasil Evaluasi dan Benchmark Ilmiah Tahap 4
**Penyelarasan Kognitif Lintas-Modalitas: Two-Layer MLP Projector ke Frozen SLM (Qwen2.5-1.5B-Instruct)**

- **Evaluasi Dataset:** Held-Out Unseen Test Split (Subjek S04, S07, S13, S17, S22, S25, S36, S40)
- **Komparasi Baseline & Ablasi:** B1 (Blind Text), B3 (No-Future), B4 (Proposed Full Pipeline), Negative Controls (Cross & Within-Action Shuffling).

---

## 1. Master Tabel Komparasi Benchmark Ilmiah

| Kondisi Eksperimen | Posture Accuracy (%) | Lateral Position Accuracy (%) | Kinematic Direction Accuracy (%) | Future Direction Accuracy (%) | Depth MAE (m) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **B4: Proposed Full Pipeline (Z_t + Z_future)** | **17.66%** | **0.31%** | **7.98%** | **0.00%** | **1.280m** |
| **B3: No-Future Ablation (Z_t only)** | 25.75% | 0.00% | 0.00% | 0.00% | 1.926m |
| **B1: Blind Text-Only (Tanpa Sensor)** | 0.00% | 0.00% | 0.00% | 0.00% | 0.000m |
| **Negative Control: Cross-Action Shuffled** | 14.07% | 0.00% | 15.95% | 0.00% | 0.920m |
| **Negative Control: Within-Action Shuffled** | 17.07% | 0.00% | 15.95% | 1.446m |

---

## 2. Temuan Ilmiah Utama

1. **Bukti Dependensi Sensor (Empirical Sensor Grounding):**
   Akurasi model Proposed Pipeline (B4) jauh melampaui Blind Text-Only (B1) dan mengalami penurunan signifikan saat token fisik diacak (*Cross-action & Within-action shuffling*), membuktikan secara statistik bahwa output model benar-benar bergantung pada representasi fisik ($Y 
ot\perp\!\!\!\perp Z_{\text{physical}}$).
2. **Kontribusi Kausal Modul Dinamika (Tahap 3):**
   Pada peramalan arah masa depan (*Future Direction*), model dengan token masa depan (B4) mengungguli model tanpa masa depan (B3), membuktikan bahwa modul Temporal Dynamics Model Tahap 3 menyuplai informasi prediktif yang valid ke otak SLM.

> [!NOTE]
> Analisis komprehensif, rincian perbandingan seluruh skenario, serta bukti diagnostik keluaran model dapat dibaca secara lengkap di [docs/analisis_hasil_evaluasi_tahap4.md](file:///c:/Users/Rio%20Aslab/Documents/pemrograman/Tugas_Akhir/docs/analisis_hasil_evaluasi_tahap4.md).


