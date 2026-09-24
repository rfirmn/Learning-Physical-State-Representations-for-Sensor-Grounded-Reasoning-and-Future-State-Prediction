# Proposal Penelitian Tugas Akhir
> **Learning Physical State Representations for Sensor-Grounded Reasoning and Future-State Prediction**  
> *(Pemodelan Representasi Fisik Spasial-Temporal dan Penyelarasan LLM Berbasis Sinyal Radar mmWave)*

---

## Informasi Dokumen
- **Topik Penelitian:** Sensor Representation Learning, Physical Dynamics, and Cognitive Alignment
- **Modalitas Sensor:** mmWave Radar Point Cloud ($N=128$, fitur: $x, y, z, v_d, \text{SNR}$)
- **Model Bahasa:** Small Language Model (Qwen2.5-1.5B-Instruct, **100% Strictly Frozen**)
- **Antarmuka Penyelaras:** Two-Layer MLP Projector (~1.97M parameter)
- **Dataset Empiris:** MM-Fi Dataset (Protocol 3, 40 subjek, 4 lingkungan, 27 aksi) + Pretrained Point-MAE ShapeNet
- **Metode Kompilasi:** Digabungkan secara otomatis dari 36 berkas modular di [`docs/sections-proposal/`](sections-proposal/)

---

## Daftar Isi

- [Catatan Akademik: Transisi dari Proposal Konseptual Awal ke Implementasi Terbumi (MM-Fi & Frozen SLM)](#catatan-akademik-transisi-dari-proposal-konseptual-awal-ke-implementasi-terbumi-mm-fi-frozen-slm)
- [1. Judul penelitian](#1-judul-penelitian)
- [2. Posisi penelitian](#2-posisi-penelitian)
- [3. Latar belakang](#3-latar-belakang)
- [4. Konsep utama penelitian](#4-konsep-utama-penelitian)
- [5. Pertanyaan penelitian](#5-pertanyaan-penelitian)
- [6. Hipotesis](#6-hipotesis)
- [7. Sensor yang digunakan](#7-sensor-yang-digunakan)
- [8. Dataset dan kurikulum pelatihan](#8-dataset-dan-kurikulum-pelatihan)
- [9. Strategi pemilihan dataset](#9-strategi-pemilihan-dataset)
- [10. Apakah data sudah dalam format LLM?](#10-apakah-data-sudah-dalam-format-llm)
- [11. Arsitektur yang diusulkan](#11-arsitektur-yang-diusulkan)
- [12. Sensor encoder](#12-sensor-encoder)
- [13. Physical state representation](#13-physical-state-representation)
- [14. Dynamics modeling](#14-dynamics-modeling)
- [15. Future-state prediction](#15-future-state-prediction)
- [16. Task reasoning](#16-task-reasoning)
- [17. Counterfactual reasoning & sensor controls](#17-counterfactual-reasoning-sensor-controls)
- [18. Evaluasi](#18-evaluasi)
- [19. Grounding evaluation](#19-grounding-evaluation)
- [20. Robustness evaluation](#20-robustness-evaluation)
- [21. Baseline](#21-baseline)
- [22. Ablation study](#22-ablation-study)
- [23. Novelty](#23-novelty)
- [24. Novelty utama](#24-novelty-utama)
- [25. Hubungan dengan world modeling](#25-hubungan-dengan-world-modeling)
- [26. Scope penelitian](#26-scope-penelitian)
- [27. Workflow penelitian](#27-workflow-penelitian)
- [28. Tahapan kurikulum eksekusi penelitian](#28-tahapan-kurikulum-eksekusi-penelitian)
- [29. Risiko penelitian](#29-risiko-penelitian)
- [30. Resource requirement](#30-resource-requirement)
- [31. Expected result](#31-expected-result)
- [32. Batas antara TA dan world model](#32-batas-antara-ta-dan-world-model)
- [33. Posisi final penelitian](#33-posisi-final-penelitian)
- [34. Kesimpulan proposal](#34-kesimpulan-proposal)
- [35. Ringkasan satu kalimat](#35-ringkasan-satu-kalimat)

---

## Catatan Akademik: Transisi dari Proposal Konseptual Awal ke Implementasi Terbumi (MM-Fi & Frozen SLM)

**Dokumen Rujukan Penjembatanan untuk Dosen Pembimbing, Penguji, dan Peneliti**  
*Penelitian Tugas Akhir: Learning Physical State Representations for Sensor-Grounded Reasoning and Future-State Prediction*

---

### 1. Latar Belakang & Tujuan Dokumen

Folder [docs/sections-proposal/](file:///Users/rio/Documents/RIO/Pemrograman/Research/Learning-Physical-State-Representations-for-Sensor-Grounded-Reasoning-and-Future-State-Prediction/docs/sections-proposal) memuat 35 berkas modular rancangan proposal penelitian yang disusun pada fase paling awal perumusan ide (*initial conceptualization phase*).

Seiring dengan implementasi sistem pada lingkungan nyata menggunakan dataset radar MM-Fi dan model bahasa Qwen2.5-1.5B, terdapat beberapa **penajaman skop penelitian (*refinement of research scope*)** agar metodologi berdiri di atas landasan empiris yang kokoh dan dapat diuji secara objektif (*scientifically verifiable*). Dokumen ini merangkum transisi tersebut agar pembaca tidak mengalami salah paham antara ide abstrak awal dan wujud implementasi sistem.

---

### 2. Matriks Transisi Konsep: Proposal Awal vs Implementasi Aktual

| Aspek Rancangan | Gagasan Konseptual Awal ([sections-proposal/](file:///Users/rio/Documents/RIO/Pemrograman/Research/Learning-Physical-State-Representations-for-Sensor-Grounded-Reasoning-and-Future-State-Prediction/docs/sections-proposal)) | Wujud Implementasi Terbumi (*Grounded Reality*) | Justifikasi Ilmiah & Rekayasa |
| :--- | :--- | :--- | :--- |
| **Jumlah Subjek dalam Skenario** | **Multi-Orang (*Person A & B*)**<br>Contoh pada Section 16 & 17 menyebut interaksi: *"Where is person A?", "Is A to the left of B?", "Who is closer to the radar?"*. | **Subjek Tunggal Kontinu (*Single-Subject Continuous Motion*)**<br>Model memproses pergerakan artikulasi 17 sendi tubuh dari satu orang dalam satu rentang rekaman radar. | Dataset patokan radar mmWave berlabel 3D skeleton presisi tinggi yang tersedia secara publik (**MM-Fi Dataset, NeurIPS 2023**) memiliki format data single-subject `(F, 17, 3)`. Riset difokuskan pada kedalaman pemodelan postur dan dinamika tubuh manusia daripada pelacakan kerumunan (*crowd tracking*). |
| **Sifat Representasi Fisik ($Z_t$)** | **Variabel Simbolik Cartesian Eksplisit**<br>Section 17 mengasumsikan kontrafaktual diuji dengan mengubah angka variabel teks: `velocity = (+1, 0)` diubah menjadi `(-1, 0)`. | **Representasi Laten Kontinu (*Dense Latent Vector*)**<br>$Z_t \in \mathbb{R}^{384}$ diekstrak dari Transformer Backbone, di mana keterbumian fisik diuji via *Sensor Shuffling Controls* dan geometri tubuh relatif. | Sistem ini adalah pemodelan dunia berbasis representasi neural (*neural representation world modeling*), bukan sistem penalaran berbasis aturan (*symbolic rule engine*). Menguji representasi laten melalui degradasi pengacakan sensor (*shuffling*) adalah standar baku ilmiah multimodal representation learning. |
| **Kurikulum Dataset** | **3 Fase Dataset Eksternal**<br>Section 8 menyebutkan dataset bertahap: MVRHAR (Fase 1), MM-Fi (Fase 2), M4Human (Fase 3). | **ShapeNet 3D CAD (Tahap 1) & MM-Fi Protocol 3 (Tahap 2–4)**<br>Transfer learning menggunakan bobot pretrained Point-MAE pada ShapeNet, dilanjutkan adaptasi domain pada MM-Fi. | Penggunaan *foundation model* Point-MAE ShapeNet jauh lebih efektif daripada melatih encoder dari nol pada dataset sederhana (MVRHAR), menghemat komputasi dan memberikan inisialisasi geometri 3D yang lebih kaya. |
| **Metode Adaptasi Bahasa** | **Opsional Fine-Tuning LoRA**<br>Section 21 menyebutkan opsi baseline: *"Sensor representation + LLM + PEFT (LoRA)"*. | **Strictly Frozen SLM (100% Beku)**<br>Hanya adapter *Two-Layer MLP Projector* (~1.97M param) yang dilatih; bobot model bahasa Qwen2.5-1.5B dibekukan secara ketat. | Mempertahankan model bahasa dalam kondisi beku murni (*strictly frozen*) adalah aturan arsitektur utama ([AGENTS.md](../AGENTS.md)) untuk: (1) menjamin kemampuan bahasa alami tidak mengalami *catastrophic forgetting*, (2) membuktikan bahwa pemahaman fisik murni berasal dari representasi sensorik, dan (3) membatasi penggunaan VRAM pada GPU 12 GB. |
| **Definisi Tugas Penalaran (Tahap 4)** | **Jarak Metrik Absolut (Meter)**<br>Awalnya direncanakan estimasi koordinat jarak kedalaman absolut (*depth distance in meters*). | **Geometri Tubuh Relatif & Perubahan Temporal**<br>Tugas difokuskan pada pemisahan pergelangan tangan terhadap lebar bahu (`current_wrist_separation`) dan perubahannya ke depan (`future_wrist_separation_change`). | Normalisasi point cloud per-frame pada preprocessing ([transforms.py](../../eksperimen_model/datasets/transforms.py)) memusatkan titik ke centroid dan menskalakannya ke *unit sphere*, sehingga informasi koordinat absolut meter musnah sebelum mencapai encoder. Menguji geometri relatif adalah pendekatan yang sah dan terbumi secara fisika radar. |

---

### 3. Kesimpulan untuk Pembaca & Penilai

Rancangan modular di dalam berkas 1–35 tetap valid sebagai **peta visi dan motivasi teoretis** penelitian. Namun, ketika menilai hasil eksperimen, model PyTorch, dan capaian angka metrik, pembaca wajib mengacu pada:
- **[README.md](../../README.md)** sebagai ringkasan master repositori yang telah diselaraskan.
- **[docs/laporan_teknis_training_dan_evaluasi_v2.md](../laporan_teknis_training_dan_evaluasi_v2.md)** untuk hasil resmi estimasi pose 3D Tahap 2 (Model Av2).
- **[docs/landasan_ilmiah_horizon_temporal_dan_dinamika_gerak.md](../landasan_ilmiah_horizon_temporal_dan_dinamika_gerak.md)** untuk justifikasi dinamika temporal Tahap 3 ($T_{in}=16, T_{out}=8$).
- **[docs/stage4_recovery_runbook.md](../stage4_recovery_runbook.md)** dan **[docs/implementation_plan_stage4_recovery.md](../implementation_plan_stage4_recovery.md)** untuk protokol ilmiah Tahap 4.


---

## 1. Judul penelitian

Pilihan utama

Learning Physical State Representations for Sensor-Grounded Reasoning and Future-State Prediction

Alternatif

Sensor-Grounded Spatial-Temporal Reasoning and Future-State Prediction in Large Language Models

Learning Sensor-Grounded Physical Representations for Reasoning and Future-State Prediction

Sensor-Grounded Physical State Modeling: Representation, Reasoning, and Future-State Prediction

Rekomendasi

Saya paling menyarankan:

Learning Physical State Representations for Sensor-Grounded Reasoning and Future-State Prediction

Alasannya, judul ini tidak mengunci penelitian hanya pada LLM. Fokus utamanya adalah physical state representation, yang merupakan fondasi lebih fundamental untuk pengembangan world model.

LLM tetap dapat digunakan sebagai reasoning layer, tetapi penelitian tidak bergantung pada klaim bahwa LLM itu sendiri merupakan world model.


---

## 2. Posisi penelitian

Penelitian ini berada pada persimpangan:

* Sensor Representation Learning
* Multimodal / Sensor-Language Learning
* Spatial-Temporal Reasoning
* Physical State Modeling
* Future-State Prediction
* Large Language Models

Posisinya:

Sensor Perception
       ↓
Sensor Representation
       ↓
Physical State Modeling
       ↓
Dynamics Modeling
       ↓
Future-State Prediction
       ↓
Sensor-Grounded Reasoning
       ↓
       ─────────────
       World Modeling
       ─────────────
       ↓
Planning / Embodied AI

Penelitian ini tidak bertujuan membangun general-purpose world model.

Sebaliknya, penelitian membangun dan mengevaluasi salah satu fondasi pentingnya:

learning a representation of the physical world that preserves state and dynamics information sufficiently for reasoning and prediction.


---

## 3. Latar belakang

Large Language Models telah menunjukkan kemampuan reasoning yang kuat pada informasi linguistik. Namun, kemampuan tersebut tidak secara otomatis berarti bahwa model memahami keadaan dan dinamika dunia fisik.

Sensor memberikan informasi yang berbeda dari bahasa.

Misalnya mmWave radar dapat memberikan:

x
y
z
velocity / Doppler
timestamp
SNR

Informasi tersebut menggambarkan physical observation secara langsung.

Namun, terdapat beberapa tahapan yang berbeda antara observation dan reasoning:

Sensor Observation
       ↓
Physical State
       ↓
State Transition / Dynamics
       ↓
Future State
       ↓
Reasoning

Model yang hanya mampu melakukan:

“What activity is being performed?”

belum tentu mampu melakukan:

“Bagaimana postur dan konfigurasi tubuh subjek saat ini?”

“Apakah kedua tangan subjek sedang terangkat atau merapat ke torso?”

“Bagaimana geometri artikulasi tubuh akan berubah dalam rentang waktu berikutnya?”

“Di mana posisi relatif sendi pergelangan tangan setelah 0.8 detik?”

“Bagaimana perubahan penalaran jika urutan dinamika fisik sensorik diintervensi?”

Pertanyaan tersebut membutuhkan pemahaman terhadap:

* spatial information;
* temporal information;
* velocity;
* trajectory;
* object relations;
* state transitions;
* physical dynamics.

Oleh karena itu, penelitian ini tidak hanya menanyakan apakah sensor dapat dimasukkan ke dalam LLM.

Pertanyaan yang lebih fundamental adalah:

Apakah learned sensor representations mampu mempertahankan informasi physical state dan dynamics yang diperlukan untuk reasoning dan future-state prediction?

Pertanyaan tersebut merupakan langkah menuju world modeling.


---

## 4. Konsep utama penelitian

Penelitian menggunakan konsep:

Observation → State → Dynamics → Prediction → Reasoning

Observation

Apa yang diterima sensor?

Radar point cloud

State

Apa keadaan fisik yang direpresentasikan?

position
velocity
orientation
skeleton
object relation

Dynamics

Bagaimana keadaan tersebut berubah?

S_t → S_(t+1)

Prediction

Apa kemungkinan keadaan berikutnya?

S_t → Ŝ_(t+1:t+k)

Reasoning

Apa yang dapat disimpulkan dari state dan dynamics?

Bagaimana konfigurasi postur tubuh subjek saat ini?
Apakah kedua tangan sedang terangkat atau merapat ke torso?
Bagaimana perubahan konfigurasi tubuh pada langkah waktu berikutnya?

Struktur ini menjadi dasar eksperimen.


---

## 5. Pertanyaan penelitian

Main Research Question

Can learned sensor representations preserve sufficient physical state and dynamics information for grounded reasoning and future-state prediction?

Kemudian dibagi menjadi:

RQ1 — Physical Representation

How effectively can learned sensor representations encode spatial-temporal physical states from mmWave radar observations?

RQ2 — Dynamics

Can the learned physical representation support prediction of future physical states and motion trajectories?

RQ3 — Reasoning

Can a large language model use the learned physical representation to perform sensor-grounded spatial-temporal reasoning?

RQ4 — Grounding

Does the model’s reasoning and prediction change consistently when the underlying physical state is changed?

RQ5 — Robustness

Does the learned representation preserve physical information under subject and environmental shifts?


---

## 6. Hipotesis

H1

Learned sensor representations that preserve spatial-temporal information will perform better on physical state estimation than simple structured or linguistic representations.

H2

Representations that preserve temporal and velocity information will provide better future-state prediction than representations based only on instantaneous spatial information.

H3

A physical representation capable of predicting future states will provide stronger sensor-grounded reasoning than a representation optimized only for classification.

H4

Sensor-language alignment will improve the ability of an LLM to reason over physical information compared with directly providing raw/structured sensor observations.

H5

A genuinely sensor-grounded model will produce consistent changes in reasoning and future-state prediction when physical observations are modified counterfactually.

H6

Physical representations that capture underlying dynamics will be more robust to subject and environmental changes than representations that primarily learn activity-specific patterns.


---

## 7. Sensor yang digunakan

mmWave Radar

mmWave radar menjadi modality utama.

Data yang digunakan dapat mencakup:

x
y
z
Doppler
range
SNR
timestamp

Keunggulan modality ini adalah sensor secara langsung memberikan informasi yang relevan terhadap physical dynamics:

Position
+
Velocity
+
Time

Hal tersebut membuat mmWave radar lebih sesuai untuk penelitian physical state dan future-state prediction dibandingkan sensor yang hanya memberikan semantic labels.


---

## 8. Dataset dan kurikulum pelatihan

Berikut adalah bagaimana 4 tahapan kurikulum pelatihan terintegrasi menjadi satu kesatuan sistem end-to-end:

1. **Tahap 1: Inisialisasi Bobot Geometri 3D (Transfer Learning)**
   - **Data & Model:** Bobot *pretrained* Point-MAE Transformer yang telah dilatih secara *self-supervised* pada dataset geometri 3D berskala besar (ShapeNet CAD, 348 MB).
   - **Hasil:** Menghasilkan modul Sensor Encoder dasar (**Model A**) yang memiliki pemahaman *prior* tentang geometri dan struktur spasial 3D tanpa harus melatih dari nol pada sinyal radar yang berderau.

2. **Tahap 2: Adaptasi Domain Radar mmWave & Estimasi Status Fisik (MM-Fi)**
   - **Data:** Dataset utama MM-Fi Protocol 3 (262.297 frame radar $N=128$ point cloud tersinkronisasi dengan 1.080 sekuens ground truth 17 sendi 3D skeleton).
   - **Hasil:** Model Encoder diadaptasikan ke domain sinyal radar melalui *task* regresi 3D pose estimation (**Model Av2**). Representasi laten $Z_t \in \mathbb{R}^{384}$ terbukti berhasil mempertahankan konfigurasi fisik tubuh subjek sebelum modul ini dibekukan (*frozen*).

3. **Tahap 3: Pemodelan Dinamika Temporal (Dynamics Model)**
   - **Data:** Sekuens status laten $Z_{t-15 \dots t}$ ($T_{in}=16$ frame atau rentang 1.6 detik @ 10 Hz).
   - **Hasil:** Melatih model dinamika temporal berbasis *autoregressive / sequence-to-sequence* untuk memproyeksikan lintasan laten masa depan $\hat{Z}_{t+1 \dots t+8}$ ($T_{out}=8$ frame atau 0.8 detik ke depan). Modul ini dibekukan setelah mencapai konvergensi representasi dinamika.

4. **Tahap 4: Penyelarasan Kognitif ke Frozen SLM (Two-Layer MLP Projector)**
   - **Data & Arsitektur:** Pasangan token representasi fisik ($Z_{hist}$ dan $\hat{Z}_{future}$) yang diproyeksikan ke *embedding space* model bahasa kecil (Qwen2.5-1.5B-Instruct) yang **100% dibekukan** (*strictly frozen*).
   - **Hasil:** Adapter proyektor MLP (~1.97M parameter) menghubungkan ruang laten fisik sensor dengan pemahaman bahasa alami, memungkinkan SLM melakukan penalaran spasial-temporal tanpa merusak kapabilitas linguistik dasarnya.

### Wujud Akhir Saat Digunakan (Inference)
Ketika seluruh modul terintegrasi, sistem beroperasi dalam satu alur terarah (*decoupled pipeline*):
$$\text{mmWave Radar Cloud } (X_{t-k:t}) \longrightarrow \mathbf{[ \text{Encoder } (\text{Av2}) ]} \longrightarrow \mathbf{[ \text{Dynamics Model} ]} \longrightarrow \mathbf{[ \text{MLP Projector} ]} \longrightarrow \mathbf{[ \text{Frozen SLM} ]} \longrightarrow \text{Jawaban Reasoning}$$
*Catatan:* Pemisahan tegas (*decoupling*) ini menjamin bahwa setiap komponen persepsi fisik, pemodelan dinamika, dan penalaran kognitif dapat diisolasi, diverifikasi, dan diuji secara independen.


---

## 9. Strategi pemilihan dataset

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


---

## 10. Apakah data sudah dalam format LLM?

Tidak.

Dan hal tersebut justru merupakan bagian penting penelitian.

Data awal:

timestamp
x
y
z
doppler
SNR

bukan:

"Subjek sedang merentangkan kedua tangan ke samping."

dan bukan pula:

LLM embedding

Pipeline penelitian:

Raw / Processed Sensor Data
          ↓
Sensor Representation
          ↓
Physical State
          ↓
Dynamics
          ↓
LLM Reasoning

Dengan demikian, penelitian dapat mengisolasi bagaimana informasi fisik berubah selama proses representasi.


---

## 11. Arsitektur yang diusulkan

Arsitektur utama:
mmWave Radar
                    │
                    ▼
           Sensor Preprocessing
                    │
                    ▼
              Sensor Encoder
                    │
                    ▼
       Physical State Representation (Z_t)
                    │
           ┌────────┴────────┐
           ▼                 ▼
    State Estimation    Dynamics Model
    (Pos, Vel, dll)   (Z_t ➔ Z_{t+1:t+k})
           │                 │
           └────────┬────────┘
                    ▼
         Future State Prediction 
        (Trajectory / Next States)
                    │
                    ▼
          [ Alignment Module / MLP ]  <-- (Lapisan Penyelaras ala SensorLLM)
                    │
                    ▼
              Frozen LLM
                    │
           ┌────────┴────────┐
           ▼                 ▼
       Reasoning         Explanation
   (Spatial/Relational) (Natural Language)


LLM tidak harus menjadi bagian dari physical dynamics model.

Ia berfungsi sebagai reasoning layer yang menggunakan physical representation.


---

## 12. Sensor encoder

Beberapa kandidat:

* Temporal Transformer;
* Point Transformer;
* PointNet/PointNet++;
* lightweight Transformer;
* 1D CNN untuk structured temporal representation.

Pemilihan model tidak menjadi kontribusi utama.

Tujuannya adalah mendapatkan representation:

[
z_t = f(X_{t-k:t})
]

di mana:

* (X) = sensor observation;
* (z_t) = learned physical representation.


---

## 13. Physical state representation

Representation harus diuji apakah mampu mempertahankan informasi seperti:

Position
Velocity
Direction
Trajectory
Object Relation

Misalnya:

Radar sequence
      ↓
     z_t
      ↓
 ┌────┼────┐
 ↓    ↓    ↓
pos  vel  relation

Dengan demikian, representation tidak hanya dinilai berdasarkan kemampuan classification.

Ia dinilai berdasarkan informasi fisik yang dapat dipulihkan atau diprediksi dari representation tersebut.


---

## 14. Dynamics modeling

Ini merupakan komponen yang membedakan penelitian dari sekadar sensor-language alignment.

Model mempelajari:

[
z_t \rightarrow z_{t+1}
]

atau:

[
S_t \rightarrow S_{t+1}
]

dan untuk horizon lebih panjang:

[
S_t \rightarrow S_{t+1},S_{t+2},…,S_{t+k}
]

Tujuannya adalah mengetahui apakah learned representation mengandung informasi mengenai bagaimana dunia fisik berubah dari waktu ke waktu.


---

## 15. Future-state prediction

Task utama:

Given observations from (t-k) to (t), predict the physical state at (t+\Delta).

Contoh:

Input:
Radar(t-10 ... t)
Output:
Position(t+1)
Velocity(t+1)
Trajectory(t+1 ... t+k)

Untuk human motion:

Radar sequence
      ↓
Encoder
      ↓
Physical state
      ↓
Dynamics model
      ↓
Future skeleton / trajectory

Ini menjadi komponen predictive modeling dalam penelitian.


---

## 16. Task reasoning

Setelah representasi fisik berhasil dipelajari dan disejajarkan melalui modul proyektor, model bahasa (Frozen SLM) digunakan untuk melakukan *sensor-grounded physical reasoning* terhadap subjek yang diamati radar:

### Task 1 — State & Posture Reasoning
* Bagaimana konfigurasi postur tubuh subjek saat ini (berdiri, duduk, membungkuk)?
* Di mana posisi relatif lengan terhadap torso dan kepala?
* Apakah kedua pergelangan tangan berada di atas atau di bawah bahu?

### Task 2 — Spatial Geometry Reasoning (Body-Relative Geometry)
* Berapa rasio jarak antar-pergelangan tangan relatif terhadap lebar bahu (*current wrist separation*: narrow, medium, wide)?
* Apakah lengan kiri berada lebih tinggi atau lebih rendah daripada lengan kanan?
* Apakah orientasi tubuh condong ke arah kiri atau kanan relatif terhadap posisi radar?

### Task 3 — Temporal & Kinematic Reasoning
* Apakah anggota tubuh sedang bergerak mendekat (*positive Doppler*) atau menjauh (*negative Doppler*) terhadap sensor?
* Apakah gerakan tangan mengalami percepatan atau deselerasi antara frame $t_1$ dan $t_2$?
* Berapa fase siklus pergerakan yang telah dilalui subjek selama rentang observasi 1.6 detik?

### Task 4 — Relational Kinematics Reasoning
* Apakah kedua tangan bergerak secara simetris atau asimetris?
* Apakah jarak antara tangan dan torso sedang merenggang atau menyempit?
* Bagaimana koordinasi pergerakan antara tungkai bawah dan lengan atas?

### Task 5 — Future-State Reasoning
* Bagaimana perubahan separasi pergelangan tangan dalam rentang 0.8 detik ke depan (*future wrist separation change*: narrowing, unchanged, widening)?
* Apakah subjek diprediksi akan menyelesaikan gerakan mengangkat tangan atau kembali ke posisi istirahat?
* Apakah lintasan dinamika masa depan konsisten dengan momentum dan inersia yang diobservasi pada jendela historis?


---

## 17. Counterfactual reasoning & sensor controls

Evaluasi kontrafaktual dan kontrol pengacakan (*sensor shuffling controls*) merupakan pilar utama untuk membuktikan bahwa model bahasa benar-benar bertumpu pada sinyal fisik sensor (**sensor grounding**), bukan sekadar menebak berdasarkan bias linguistik atau hafalan teks (*language hallucination*).

Karena status fisik dalam riset ini direpresentasikan oleh vektor laten kontinu $Z_t \in \mathbb{R}^{384}$ hasil ekstraksi Point-MAE dari point cloud radar, intervensi kontrafaktual dilakukan secara empiris melalui gangguan terukur pada domain sensor:

### 1. Cross-Action Shuffling Control (Intervensi Fisik Antar-Aksi)
* **Skenario:** Mengganti sekuens point cloud radar aktual dengan sekuens dari aksi fisik yang sangat berbeda (misalnya: radar dari gerakan melompat dimasukkan ke dalam pertanyaan tentang merentangkan tangan).
* **Ekspektasi Grounding:** Jika model benar-benar membaca representasi sensor, akurasi penalaran harus **turun drastis (*performance drop*)**. Jika akurasi tetap tinggi, berarti LLM menjawab hanya dari prior teks pertanyaan tanpa memedulikan sinyal radar.

### 2. Within-Action Temporal Shuffling / Reversal (Intervensi Dinamika Kausalitas)
* **Skenario:** Membalikkan urutan waktu frame radar ($t_k \rightarrow t_1$) atau mengacak urutan temporal frame dalam satu aksi.
* **Ekspektasi Grounding:** Pada pertanyaan kinematika dan prediksi masa depan (*future-state prediction*), pembalikan urutan waktu harus membalikkan arah prediksi (misalnya dari *increasing separation* menjadi *decreasing separation*).

### Metrik Pembuktian: Shuffling Degradation Gap
Tingkat keterikatan fisik (*physical grounding*) diukur melalui selisih degradasi kinerja:
$$\Delta_{\text{Grounding}} = \text{Accuracy}_{\text{Original}} - \text{Accuracy}_{\text{Shuffled}}$$

Semakin besar nilai $\Delta_{\text{Grounding}}$, semakin kuat bukti ilmiah bahwa penalaran model benar-benar didorong oleh representasi fisik sensorik mmWave radar.


---

## 18. Evaluasi

Evaluasi penelitian dirancang secara terstruktur mencakup setiap tahap representasi, dinamika, hingga penalaran bahasa:

### A. Evaluasi Status Fisik (Tahap 2 — 3D Pose Estimation)
1. **MPJPE (Mean Per-Joint Position Error):**
   Mengukur rata-rata jarak euclidean antara prediksi posisi 17 sendi skeleton dengan ground truth (dalam satuan milimeter):
   $$\text{MPJPE} = \frac{1}{J}\sum_{j=1}^{J} \|\mathbf{p}_j - \hat{\mathbf{p}}_j\|_2$$
2. **PA-MPJPE (Procrustes-Aligned MPJPE):**
   Mengukur error posisi sendi setelah dilakukan *rigid alignment* (translasi, rotasi, skala) melalui Procrustes Analysis untuk mengevaluasi kualitas bentuk pose murni secara terisolasi.
3. **PCK@100 (Percentage of Correct Keypoints):**
   Persentase sendi yang memiliki jarak error di bawah ambang batas toleransi 100 mm.

---

### B. Evaluasi Pemodelan Dinamika Temporal (Tahap 3 — Dynamics Forecasting)
1. **Latent MSE:**
   Mengukur galat kuadrat rata-rata pada ruang representasi laten $Z \in \mathbb{R}^{384}$ antara status masa depan prediksi $\hat{Z}_{t+h}$ dan ground truth $Z_{t+h}$ untuk horizon $h \in [1, 8]$:
   $$\text{MSE}_{\text{latent}} = \frac{1}{h \cdot D} \sum_{i=1}^{h} \|\mathbf{z}_{t+i} - \hat{\mathbf{z}}_{t+i}\|_2^2$$
2. **Latent Cosine Similarity:**
   Mengukur konsistensi arah pergerakan vektor representasi dalam ruang multidimensi:
   $$\text{CosSim}(\mathbf{z}, \hat{\mathbf{z}}) = \frac{\mathbf{z} \cdot \hat{\mathbf{z}}}{\|\mathbf{z}\| \|\hat{\mathbf{z}}\|}$$
3. **Horizon Trajectory Drift & Persistence Margin:**
   Mengukur deviasi galat per-langkah waktu ($t+1$ hingga $t+8$) dan memverifikasi apakah model dinamika menghasilkan error yang lebih rendah dibandingkan model *persistence* statis ($\hat{Z}_{t+h} = Z_t$).

---

### C. Evaluasi Penalaran Bahasa & Sensor Grounding (Tahap 4 — Cognitive Alignment)
1. **Body-Relative Geometry Macro-F1:**
   Mengukur akurasi penalaran terdistribusi seimbang pada kategori geometri tubuh yang telah dinormalisasi terhadap dimensi subjek (misalnya: `current_wrist_separation` [narrow, medium, wide] dan `future_wrist_separation_change` [narrowing, unchanged, widening]).
   *(Catatan metodologis: Evaluasi menggunakan geometri tubuh relatif alih-alih koordinat absolut ruangan karena normalisasi sentralisasi point cloud radar per frame mengeliminasi translasi absolut global subjek).*
2. **Valid Parse Rate:**
   Persentase luaran bahasa dari model yang berhasil diparsing secara konsisten sesuai format jawaban terstruktur target.
3. **Shuffling Degradation Gap ($\Delta_{\text{Grounding}}$):**
   Mengukur penurunan performa penalaran saat representasi sensor diacak (Cross-Action Shuffling atau Within-Action Shuffling). Nilai degradasi yang signifikan membuktikan bahwa model bahasa benar-benar bertumpu pada sinyal fisik sensor dan terbebas dari halusinasi teks.


---

## 19. Grounding evaluation

Salah satu pertanyaan terpenting:

Apakah model benar-benar menggunakan sensor?

Bukan hanya:

Apakah model memberikan jawaban benar?

Eksperimen:

Original sensor
      ↓
Prediction A
Modified sensor
      ↓
Prediction B

Kemudian dibandingkan:

Physical change
       ↓
Expected prediction change
       ↓
Actual prediction change

Jika input berubah tetapi prediction tidak berubah:

model kemungkinan tidak menggunakan physical information secara efektif.


---

## 20. Robustness evaluation

Training:

Subjects 1–N
Environment A

Testing:

New subjects
Environment B

Kemudian:

[
Robustness\ Drop =
Performance_{in-domain}

Performance_{out-domain}
]

Representasi yang baik diharapkan mengalami penurunan yang lebih kecil.

Eksperimen dapat meliputi:

* subject shift;
* environment shift;
* sensor configuration shift;
* noise;
* missing points;
* perubahan density point cloud.


---

## 21. Baseline

Baseline penelitian dibangun secara bertingkat dan terisolasi (**B0 hingga B5**) untuk mengukur kontribusi masing-masing komponen:

| Kode Baseline | Konfigurasi Model | Sensor Encoder (Point-MAE) | Dynamics Model | Adapter / Projector | Large Language Model | Keterangan / Tujuan Ilmiah |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **B0** | Text-Only Zero-Shot | — | — | — | Frozen SLM | Mengukur bias linguistik dan halusinasi teks murni tanpa sensor. |
| **B1** | Text-Only Few-Shot | — | — | — | Frozen SLM | Menilai pengaruh contoh *in-context* tanpa informasi sensorik. |
| **B2** | Direct Task Probes | ✓ ($Z_t$) | — | Linear / MLP Probe | — (Tanpa LLM) | Menguji apakah informasi fisik dapat diekstrak langsung tanpa model bahasa. |
| **B3** | Sensor Historical Alignment | ✓ ($Z_{t-15:t}$) | — | Two-Layer MLP | **Strictly Frozen** | Memetakan 16 token fisik historis ke ruang bahasa tanpa prediksi masa depan. |
| **B3P** | Persistence Future Control | ✓ ($Z_{t-15:t} + Z_t^{\times 8}$) | — (Statis) | Two-Layer MLP | **Strictly Frozen** | Mengontrol *token budget* identik (24 token fisik) menggunakan pengulangan status terakhir. |
| **B4** | **Full Proposed Pipeline** | ✓ ($Z_{t-15:t}$) | ✓ ($\hat{Z}_{t+1:t+8}$) | Two-Layer MLP | **Strictly Frozen** | Menggabungkan 16 token historis + 8 token dinamika masa depan menuju SLM beku. |
| **B5** | Oracle / Upper Bound | ✓ ($Z_{t-15:t}$) | Oracle ($Z_{t+1:t+8}^{\text{GT}}$) | Two-Layer MLP | **Strictly Frozen** | Mengukur batas atas teoretis jika model dinamika memiliki akurasi sempurna 100%. |

### Aturan Arsitektur: Strictly Frozen SLM
Model bahasa (Qwen2.5-1.5B-Instruct) berada dalam kondisi **100% beku (*strictly frozen*)** pada seluruh baseline B3, B3P, B4, dan B5. Modifikasi parameter LLM seperti LoRA atau PEFT ditiadakan untuk memastikan:
1. Kemampuan bahasa alami tidak terdegradasi (*zero catastrophic forgetting*).
2. Peningkatan penalaran fisik murni berasal dari kualitas representasi sensor dan model dinamika, bukan akibat penyesuaian bobot teks.
3. Efisiensi komputasi terpenuhi pada hardware riset (12 GB VRAM).


---

## 22. Ablation study

Studi ablasi dirancang untuk mengisolasi dan memahami kontribusi masing-masing komponen fisik, temporal, dan arsitektural:

```text
Full Pipeline (B4)
    │
    ├── 1. Ablasi Kanal Sensor (Input Modality)
    │     ├── Tanpa Doppler (Menghapus kecepatan radial v_d)
    │     ├── Tanpa SNR (Menghapus intensitas pantulan radar)
    │     └── Spatial-Only (Hanya koordinat 3D x, y, z)
    │
    ├── 2. Ablasi Pemodelan Dinamika Masa Depan
    │     ├── B4 (Predicted Dynamics) vs B3P (Persistence Baseline)
    │     └── B4 vs B3 (Historical Only, tanpa token masa depan)
    │
    ├── 3. Ablasi Konteks Temporal & Horizon
    │     ├── Variasi Konteks Historis T_in ∈ {8, 16} (0.8s vs 1.6s)
    │     └── Variasi Horizon Prediksi T_out ∈ {4, 8} (0.4s vs 0.8s)
    │
    └── 4. Ablasi Arsitektur Proyektor Penyelaras
          ├── Linear Projection tunggal (W * z)
          └── Two-Layer MLP Projector dengan aktivasi non-linear GELU
```

### Tujuan Ilmiah:
Tujuannya bukan sekadar mengejar angka akurasi tertinggi, melainkan mengungkap:
1. **Peran Sinyal Fisik Radar:** Sejauh mana kecepatan Doppler berkontribusi dalam membedakan pergerakan artikulasi cepat dibandingkan fitur posisi spasial murni?
2. **Kebutuhan Model Dinamika:** Apakah proyeksi masa depan eksplisit dari Dynamics Model memberikan keuntungan penalaran yang signifikan di atas model *persistence* statis?
3. **Kapasitas Pemetaan Antarmuka:** Apakah adapter proyektor non-linear (Two-Layer MLP) cukup menjembatani diskrepansi representasi tanpa perlu mengubah bobot model bahasa (LLM)?


---

## 23. Novelty

Novelty tidak diklaim sebagai:

“A new SensorLLM architecture.”

atau:

“A new world model.”

Klaim tersebut terlalu besar.

Novelty yang lebih defensible:

Novelty 1 — Physical information preservation

Menganalisis apakah learned sensor representations mempertahankan physical state information.

Novelty 2 — Predictive physical representation

Menguji apakah representation tersebut mampu mendukung future-state prediction, bukan hanya recognition.

Novelty 3 — Sensor-grounded reasoning

Menguji apakah LLM dapat menggunakan physical representation untuk spatial-temporal reasoning.

Novelty 4 — Counterfactual grounding

Menguji apakah reasoning dan prediction berubah secara konsisten ketika physical state diubah.

Novelty 5 — Robustness

Menguji apakah physical representation tetap berguna ketika terjadi subject/environment shift.


---

## 24. Novelty utama

Jika harus diringkas menjadi satu:

Investigating whether learned sensor representations preserve the physical state and dynamics information required for grounded reasoning and future-state prediction.

Dengan kata lain:

Penelitian bukan sekadar:

Can sensors talk to LLMs?

tetapi:

Does the sensor representation preserve enough
information about the physical world
for a model to reason about it
and predict how it will change?

Ini merupakan pertanyaan yang lebih dekat dengan fundamental world modeling.


---

## 25. Hubungan dengan world modeling

Penelitian ini tidak mengklaim membangun general-purpose world model.

Namun, pipeline yang dibangun merupakan fondasi:

Sensor
 ↓
Physical State
 ↓
Dynamics
 ↓
Future State

yang merupakan inti dari predictive world modeling.

Tahap lanjutan setelah penelitian ini dapat berupa:

Physical State
      +
Dynamics
      +
Action
      ↓
Action-conditioned Future Prediction
      ↓
Planning
      ↓
Embodied Agent

Dengan demikian penelitian dapat menjadi tahap pertama menuju:

* world models;
* embodied AI;
* robotics;
* physical agents;
* sensor-driven autonomous systems.


---

## 26. Scope penelitian

Agar tetap feasible, terfokus, dan terukur secara ketat sebagai Tugas Akhir:

1 modalitas sensorik
        ↓
mmWave Radar Point Cloud ($N=128$, fitur $x, y, z, v_d, \text{SNR}$)

1 domain fisik
        ↓
Artikulasi pergerakan tubuh manusia (*human motion dynamics*)

1 dataset empiris utama
        ↓
MM-Fi Dataset — Protocol 3 (40 subjek, 4 lingkungan, 27 aksi) + Pretrained Point-MAE ShapeNet

1 model bahasa kecil (Strictly Frozen)
        ↓
Qwen2.5-1.5B-Instruct (100% parameter beku)

1 antarmuka representasi fisik
        ↓
Point-MAE Transformer Encoder ($Z_t \in \mathbb{R}^{384}$) + Two-Layer MLP Projector

Batasan lingkup (Tidak perlu / Di luar cakupan):
* melatih LLM dari nol;
* fine-tuning bobot LLM (LoRA / PEFT ditiadakan);
* multimodal foundation model berbobot raksasa;
* penggabungan ragam modalitas sensor lain (kamera RGB, LiDAR, wearable IMU);
* implementasi perangkat keras robotika fisik;
* deployment real-time embedded;
* pembangunan general-purpose world model skala besar.


---

## 27. Workflow penelitian

Literature Review
                    │
                    ▼
             Dataset Selection
                    │
                    ▼
             Radar Preprocessing
                    │
                    ▼
            Sensor Representation
                    │
                    ▼
            Physical State Learning
                    │
                    ▼
             Dynamics Modeling
                    │
                    ▼
          Future-State Prediction
                    │
                    ▼
             Sensor-Language
                Alignment
                    │
                    ▼
             LLM Reasoning
                    │
         ┌──────────┼──────────┐
         ▼          ▼          ▼
      Spatial    Temporal   Relational
         │          │          │
         └──────────┼──────────┘
                    ▼
          Counterfactual Testing
                    │
                    ▼
           Robustness Evaluation
                    │
                    ▼
              Ablation Study
                    │
                    ▼
              Final Analysis


---

## 28. Tahapan kurikulum eksekusi penelitian

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


---

## 29. Risiko penelitian

Risiko 1 — Physical state prediction buruk

Solusi:

* sederhanakan target;
* gunakan subset dataset;
* mulai dari position/velocity;
* jangan langsung predict full skeleton.

--------

Risiko 2 — Dynamics prediction terlalu sulit

Solusi:

gunakan horizon pendek terlebih dahulu:

t → t+1

kemudian:

t → t+1...t+k

--------

Risiko 3 — LLM tidak dapat menggunakan representation

Solusi:

gunakan structured physical representation sebagai intermediate baseline.

Radar
 ↓
Physical State
 ↓
Text / Structured Representation
 ↓
LLM

Jika pipeline ini berhasil, baru bandingkan dengan learned representation.

--------

Risiko 4 — LLM terlalu dominan

Solusi:

gunakan frozen LLM sebagai baseline dan pisahkan kontribusi sensor-side dengan LLM-side.

--------

Risiko 5 — Dataset terlalu sederhana atau terbatas pada satu lingkungan

Solusi:

mulai dari lingkungan terkontrol (*Environment E01*), kemudian uji generalisasi dan robustness model pada variasi lingkungan berbeda (*Cross-Environment E02–E04* pada dataset MM-Fi).


---

## 30. Resource requirement

Kebutuhan komputasi dirancang sangat efisien dan dapat dieksekusi penuh pada lingkungan workstation tunggal (*consumer-grade GPU*):

### Tidak Membutuhkan:
* training LLM from scratch;
* fine-tuning bobot LLM (LoRA / PEFT ditiadakan) — model bahasa 100% dibekukan (*strictly frozen*);
* GPU cluster berskala industri;
* robot fisik / hardware deployment khusus;
* anotasi manual baru (menggunakan ground truth 3D mocap MM-Fi).

### Arsitektur Hardware & Software yang Digunakan:
* **Dataset:** MM-Fi Dataset (Protocol 3) + Bobot *pretrained* Point-MAE ShapeNet.
* **Komputasi:** 1x GPU Consumer (NVIDIA GeForce RTX 3060, 12 GB VRAM, CUDA 12.4).
* **Modul Terlatih:**
  - Sensor Encoder (Point-MAE Model Av2, output laten 384d).
  - Dynamics Model (Temporal Transformer untuk prediksi 8 horizon laten).
  - Two-Layer MLP Cross-Modal Projector (~1.97M parameter).
* **Model Bahasa:** Frozen Small Language Model (Qwen2.5-1.5B-Instruct, presisi bfloat16).

Komputasi difokuskan sepenuhnya pada pembelajaran representasi fisik sensorik dan pemodelan dinamika, bukan pada pelatihan model bahasa. Hal ini menjaga konsumsi memori VRAM tetap di bawah batas 12 GB sepanjang seluruh tahapan eksperimen.


---

## 31. Expected result

Ada beberapa kemungkinan hasil yang semuanya valid.

Hasil A — Representation + dynamics berhasil

Sensor representation
        ↓
Good future prediction
        ↓
Better physical reasoning

Kesimpulan:

Learned representation mampu mempertahankan physical state dan dynamics yang berguna untuk reasoning.

--------

Hasil B — Representation bagus untuk state tetapi buruk untuk future

State estimation ↑
Prediction ↓

Ini menunjukkan:

representation menangkap physical state tetapi belum menangkap underlying dynamics secara memadai.

Ini merupakan hasil yang sangat relevan untuk world modeling.

--------

Hasil C — Dynamics bagus tetapi LLM tidak mampu menggunakannya

Physical prediction ↑
LLM reasoning ↓

Kesimpulan:

physical representation dapat memodelkan dynamics, tetapi sensor-language interface masih menjadi bottleneck.

--------

Hasil D — Counterfactual gagal

Jika:

Physical state changes
        ↓
Prediction barely changes

maka:

model mungkin mempelajari correlation daripada physical dynamics.

Ini juga merupakan hasil penelitian yang valid.

--------

Hasil E — Robustness rendah

Jika:

In-domain ↑
Out-of-domain ↓↓↓

maka:

representation belum cukup invariant terhadap environmental changes.

Ini membuka penelitian lanjutan mengenai robust physical representation.


---

## 32. Batas antara TA dan world model

Penelitian ini:

                  WORLD MODEL
                       │
              Action-conditioned
                 prediction
                       │
                    Planning
                       │
                 Embodied AI
                       │
              ─────────┼─────────
                       │
               Future prediction
                       │
               Dynamics modeling
                       │
             Physical state model
                       │
                ← PENELITIAN →
                       │
             Sensor representation
                       │
                Sensor perception

Dengan demikian:

TA ini tidak mencoba menyelesaikan world modeling.

TA ini mencoba menjawab:

Bagaimana memperoleh representasi physical state dari sensor yang cukup informatif untuk memahami dan memprediksi perubahan dunia fisik?

Itulah fondasi yang nantinya dapat digunakan untuk membangun world model yang lebih lengkap.


---

## 33. Posisi final penelitian

Domain

AI / Multimodal Learning / Sensor Intelligence

Modality

mmWave Radar

Physical domain

Human motion

Core problem

Physical state representation and dynamics modeling

Reasoning problem

Sensor-grounded spatial-temporal reasoning

Predictive problem

Future-state / trajectory prediction

LLM role

Reasoning layer

Language Interface

Two-Layer MLP Projector (LLM is Strictly Frozen)

Main contribution

Analysis of physical-information preservation and predictive capability of learned sensor representations

Strong evaluation

Sensor Shuffling Controls (Cross-Action / Within-Action) + Cross-Subject / Cross-Environment Held-Out Evaluation

Dataset

MM-Fi Dataset (Protocol 3, mmWave Radar Point Cloud) with ShapeNet 3D CAD Pretrained Point-MAE Initialization


---

## 34. Kesimpulan proposal

Penelitian ini bertujuan menginvestigasi bagaimana data sensor mmWave radar dapat dipelajari menjadi representasi physical state yang mempertahankan informasi spatial-temporal dan dynamics yang diperlukan untuk memahami serta memprediksi keadaan dunia fisik. Penelitian tidak berfokus pada pembangunan LLM baru maupun general-purpose world model, tetapi pada fondasi representasional yang memungkinkan sistem AI melakukan reasoning terhadap physical information. Radar observation diproses melalui sensor encoder untuk menghasilkan physical representation, kemudian digunakan untuk physical state estimation dan future-state prediction. Representasi tersebut selanjutnya dapat dihubungkan dengan Large Language Model untuk melakukan spatial, temporal, dan relational reasoning terhadap keadaan fisik. Kemampuan model dievaluasi menggunakan physical-state error, trajectory prediction metrics, reasoning accuracy, counterfactual consistency, dan robustness terhadap subject maupun environment shift. Kontribusi utama penelitian adalah menganalisis apakah learned sensor representations mampu mempertahankan physical state dan dynamics information yang cukup untuk menghasilkan grounded reasoning dan future-state prediction. Dengan demikian, penelitian ini tidak mengklaim sebagai pembangunan world model penuh, tetapi menyediakan salah satu fondasi penting menuju predictive world modeling, physical AI, dan embodied intelligence.


---

## 35. Ringkasan satu kalimat

Jika harus menjelaskan penelitian ini kepada dosen dalam satu kalimat:

Saya ingin meneliti apakah data sensor dapat dipelajari menjadi representasi keadaan fisik yang tidak hanya menjelaskan apa yang sedang terjadi, tetapi juga mempertahankan dynamics sehingga model dapat memprediksi apa yang akan terjadi dan menggunakan informasi tersebut untuk melakukan physical reasoning.

Atau secara lebih singkat:

Sensor → Physical State → Dynamics → Future State → Reasoning.

Itulah inti penelitian.


---

## Penutup & Integritas Dokumen
Dokumen ini merupakan kompilasi terpadu dari seluruh rancangan seksi proposal penelitian. Seluruh parameter teknis, diagram arsitektur, dan protokol evaluasi yang termuat di dalamnya telah diselaraskan dengan implementasi kode aktif di repositori dan hasil pengujian empiris yang tervalidasi.
