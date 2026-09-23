# Catatan Akademik: Transisi dari Proposal Konseptual Awal ke Implementasi Terbumi (MM-Fi & Frozen SLM)

**Dokumen Rujukan Penjembatanan untuk Dosen Pembimbing, Penguji, dan Peneliti**  
*Penelitian Tugas Akhir: Learning Physical State Representations for Sensor-Grounded Reasoning and Future-State Prediction*

---

## 1. Latar Belakang & Tujuan Dokumen

Folder [docs/sections-proposal/](file:///Users/rio/Documents/RIO/Pemrograman/Research/Learning-Physical-State-Representations-for-Sensor-Grounded-Reasoning-and-Future-State-Prediction/docs/sections-proposal) memuat 35 berkas modular rancangan proposal penelitian yang disusun pada fase paling awal perumusan ide (*initial conceptualization phase*).

Seiring dengan implementasi sistem pada lingkungan nyata menggunakan dataset radar MM-Fi dan model bahasa Qwen2.5-1.5B, terdapat beberapa **penajaman skop penelitian (*refinement of research scope*)** agar metodologi berdiri di atas landasan empiris yang kokoh dan dapat diuji secara objektif (*scientifically verifiable*). Dokumen ini merangkum transisi tersebut agar pembaca tidak mengalami salah paham antara ide abstrak awal dan wujud implementasi sistem.

---

## 2. Matriks Transisi Konsep: Proposal Awal vs Implementasi Aktual

| Aspek Rancangan | Gagasan Konseptual Awal ([sections-proposal/](file:///Users/rio/Documents/RIO/Pemrograman/Research/Learning-Physical-State-Representations-for-Sensor-Grounded-Reasoning-and-Future-State-Prediction/docs/sections-proposal)) | Wujud Implementasi Terbumi (*Grounded Reality*) | Justifikasi Ilmiah & Rekayasa |
| :--- | :--- | :--- | :--- |
| **Jumlah Subjek dalam Skenario** | **Multi-Orang (*Person A & B*)**<br>Contoh pada Section 16 & 17 menyebut interaksi: *"Where is person A?", "Is A to the left of B?", "Who is closer to the radar?"*. | **Subjek Tunggal Kontinu (*Single-Subject Continuous Motion*)**<br>Model memproses pergerakan artikulasi 17 sendi tubuh dari satu orang dalam satu rentang rekaman radar. | Dataset patokan radar mmWave berlabel 3D skeleton presisi tinggi yang tersedia secara publik (**MM-Fi Dataset, NeurIPS 2023**) memiliki format data single-subject `(F, 17, 3)`. Riset difokuskan pada kedalaman pemodelan postur dan dinamika tubuh manusia daripada pelacakan kerumunan (*crowd tracking*). |
| **Sifat Representasi Fisik ($Z_t$)** | **Variabel Simbolik Cartesian Eksplisit**<br>Section 17 mengasumsikan kontrafaktual diuji dengan mengubah angka variabel teks: `velocity = (+1, 0)` diubah menjadi `(-1, 0)`. | **Representasi Laten Kontinu (*Dense Latent Vector*)**<br>$Z_t \in \mathbb{R}^{384}$ diekstrak dari Transformer Backbone, di mana keterbumian fisik diuji via *Sensor Shuffling Controls* dan geometri tubuh relatif. | Sistem ini adalah pemodelan dunia berbasis representasi neural (*neural representation world modeling*), bukan sistem penalaran berbasis aturan (*symbolic rule engine*). Menguji representasi laten melalui degradasi pengacakan sensor (*shuffling*) adalah standar baku ilmiah multimodal representation learning. |
| **Kurikulum Dataset** | **3 Fase Dataset Eksternal**<br>Section 8 menyebutkan dataset bertahap: MVRHAR (Fase 1), MM-Fi (Fase 2), M4Human (Fase 3). | **ShapeNet 3D CAD (Tahap 1) & MM-Fi Protocol 3 (Tahap 2–4)**<br>Transfer learning menggunakan bobot pretrained Point-MAE pada ShapeNet, dilanjutkan adaptasi domain pada MM-Fi. | Penggunaan *foundation model* Point-MAE ShapeNet jauh lebih efektif daripada melatih encoder dari nol pada dataset sederhana (MVRHAR), menghemat komputasi dan memberikan inisialisasi geometri 3D yang lebih kaya. |
| **Metode Adaptasi Bahasa** | **Opsional Fine-Tuning LoRA**<br>Section 21 menyebutkan opsi baseline: *"Sensor representation + LLM + PEFT (LoRA)"*. | **Strictly Frozen SLM (100% Beku)**<br>Hanya adapter *Two-Layer MLP Projector* (~1.97M param) yang dilatih; bobot model bahasa Qwen2.5-1.5B dibekukan secara ketat. | Mempertahankan model bahasa dalam kondisi beku murni (*strictly frozen*) adalah aturan arsitektur utama ([AGENTS.md](../AGENTS.md)) untuk: (1) menjamin kemampuan bahasa alami tidak mengalami *catastrophic forgetting*, (2) membuktikan bahwa pemahaman fisik murni berasal dari representasi sensorik, dan (3) membatasi penggunaan VRAM pada GPU 12 GB. |
| **Definisi Tugas Penalaran (Tahap 4)** | **Jarak Metrik Absolut (Meter)**<br>Awalnya direncanakan estimasi koordinat jarak kedalaman absolut (*depth distance in meters*). | **Geometri Tubuh Relatif & Perubahan Temporal**<br>Tugas difokuskan pada pemisahan pergelangan tangan terhadap lebar bahu (`current_wrist_separation`) dan perubahannya ke depan (`future_wrist_separation_change`). | Normalisasi point cloud per-frame pada preprocessing ([transforms.py](../../eksperimen_model/datasets/transforms.py)) memusatkan titik ke centroid dan menskalakannya ke *unit sphere*, sehingga informasi koordinat absolut meter musnah sebelum mencapai encoder. Menguji geometri relatif adalah pendekatan yang sah dan terbumi secara fisika radar. |

---

## 3. Kesimpulan untuk Pembaca & Penilai

Rancangan modular di dalam berkas 1–35 tetap valid sebagai **peta visi dan motivasi teoretis** penelitian. Namun, ketika menilai hasil eksperimen, model PyTorch, dan capaian angka metrik, pembaca wajib mengacu pada:
- **[README.md](../../README.md)** sebagai ringkasan master repositori yang telah diselaraskan.
- **[docs/laporan_teknis_training_dan_evaluasi_v2.md](../laporan_teknis_training_dan_evaluasi_v2.md)** untuk hasil resmi estimasi pose 3D Tahap 2 (Model Av2).
- **[docs/landasan_ilmiah_horizon_temporal_dan_dinamika_gerak.md](../landasan_ilmiah_horizon_temporal_dan_dinamika_gerak.md)** untuk justifikasi dinamika temporal Tahap 3 ($T_{in}=16, T_{out}=8$).
- **[docs/stage4_recovery_runbook.md](../stage4_recovery_runbook.md)** dan **[docs/implementation_plan_stage4_recovery.md](../implementation_plan_stage4_recovery.md)** untuk protokol ilmiah Tahap 4.
