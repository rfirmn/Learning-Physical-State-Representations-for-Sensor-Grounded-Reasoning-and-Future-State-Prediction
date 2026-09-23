# Rencana Implementasi Komprehensif (Master Plan): Penyelarasan Kognitif Lintas-Modalitas (MLP Projector & Sensor-Grounded SLM Inference)

> [!WARNING]
> **STATUS RENCANA: DIREVISI OLEH RENCANA PEMULIHAN (REVISI 2)**  
> Desain awal pada dokumen ini (khususnya skema 13 token dan 5 kategori tugas metrik absolut) telah direvisi secara mendasar. Melalui evaluasi matematis pada preprocessing encoder, estimasi metrik absolut (kedalaman meter) ditunda karena normalisasi per-frame memusatkan point cloud. Metodologi aktif yang diadopsi saat ini berlandaskan pada geometri tubuh relatif dan penyetaraan token 16/24 (B3/B3P/B4). Silakan merujuk ke **[docs/implementation_plan_stage4_recovery.md](implementation_plan_stage4_recovery.md)** untuk rencana implementasi terbaru.

Dokumen ini adalah **panduan implementasi teknis dan ilmiah lengkap** untuk mengeksekusi **Tahap 4 (Penyelarasan Kognitif Lintas-Modalitas)** pada penelitian Tugas Akhir:
> **Learning Physical State Representations for Sensor-Grounded Reasoning and Future-State Prediction**  
> *(Fondasi Menuju Predictive World Modeling dan Embodied AI Berbasis Sinyal Radar mmWave)*

Dokumen ini mencakup pembaruan riset model bahasa berskala kecil (*Small Language Models / SLM*) generasi terkini (termasuk keluarga **Phi-4** dan **Qwen3/Qwen2.5**), penetapan model final resmi, jaminan komputasi 12 GB VRAM, formulasi matematis, serta protokol evaluasi berstandar publikasi internasional (*top-tier academic rigor*).

---

## 1. Posisi Tahap 4 dalam Kurikulum Pelatihan

Prinsip utama penelitian ini adalah **pemisahan tegas (*decoupling*) antara modul persepsi fisik dan modul kognitif bahasa**. Sistem tidak melatih LLM dari awal, melainkan menghubungkan representasi fisik terstruktur ke dalam model bahasa yang telah memiliki pemahaman semantik dunia.

```text
                                    KURIKULUM RISET 4 TAHAP
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ TAHAP 1: Inisialisasi Bobot Geometris 3D (ShapeNet CAD)                                          │
│ └── Output: Bobot Pretrained Point-MAE Transformer Encoder (Model A) [SELESAI]                   │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ TAHAP 2: Pematangan Persepsi & Estimasi 17 Sendi 3D Skeleton (Domain Radar MM-Fi)               │
│ └── Output: Sensor Encoder Av2 (MPJPE < 95mm, Frozen Backbone) [SELESAI]                         │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ TAHAP 3: Pemodelan Dinamika Temporal Deret Waktu (Temporal Transformer / Residual GRU)           │
│ └── Output: Latent Dynamics Model (Memprediksi Z_{t+1:t+8}, Frozen) [SELESAI]                    │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ TAHAP 4: Penyelarasan Kognitif Lintas-Modalitas (MLP Projector Alignment ke Frozen SLM)          │
│ └── Status: AKTIF (Fokus Perencanaan Dokumen Ini)                                                │
│ └── Output: Model Av2 (Frozen) + Dynamics (Frozen) + MLP Projector (Trainable) + SLM (Frozen)    │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Riset Model SLM Terkini (Generasi 2024–2026) & Penetapan Model Final

Sesuai arahan riset lanjutan, kami mengevaluasi perkembangan keluarga model SLM generasi mutakhir (termasuk **Microsoft Phi-4** dan **Alibaba Qwen3 / Qwen2.5**) untuk menyeimbangkan antara kecerdasan inferensial dan batasan memori GPU lokal (**NVIDIA RTX 3060 12 GB VRAM** di lingkungan Windows OS).

### A. Lanskap Model Generasi Terkini:
1. **Microsoft Phi-4 Family (2025):**
   - **Phi-4 (14B):** Model dense berdaya nalar STEM luar biasa, namun ukuran 14B membutuhkan ~28 GB (BF16) atau ~8 GB (4-bit). Tidak memungkinkan untuk pelatihan proyektor dengan perambatan graf aktivasi pada 12 GB VRAM.
   - **Phi-4-mini (3.8B):** Menggunakan $d_{\text{model}} = 3072$. Bobot model BF16 memakan **~7.6 GB**. Mengingat sistem operasi Windows dengan desktop display manager (DWM) secara konstan menyerap ~0.8–1.2 GB VRAM, ruang tersisa pada RTX 3060 hanya sekitar ~3.2 GB. Alur *backpropagation* aktivasi melintasi 32 transformer layer berisiko tinggi memicu *CUDA Out of Memory (OOM)*.
2. **Alibaba Qwen Family (Qwen3 & Qwen2.5):**
   - **Qwen3 (2025/2026):** Memperkenalkan arsitektur *hybrid thinking/non-thinking* dan varian MoE (seperti 30B-A3B). Varian dense ultra-kecil (0.6B) memiliki kapasitas penalaran teks yang terlalu terbatas untuk interpretasi koordinat spasial multi-atribut, sedangkan varian menengah (4B / 8B) memiliki isu VRAM serupa dengan Phi-4-mini.
   - **Qwen2.5 (2024/2025):** Varian **1.5B** dan **3B** tetap menjadi *gold standard* open-weights paling stabil, matang, dan teruji secara global untuk integrasi adaptor multimodal (*connector-based models*), dengan dukungan ekosistem HuggingFace yang sangat solid.
3. **Meta Llama-3.2 (1B & 3B):**
   - Model ringan yang sangat efisien, namun skor penalaran terstruktur (JSON, matematika, logika spasial) pada varian 1B masih berada di bawah Qwen2.5-1.5B.

### B. Matriks Komparasi Ilmiah Kandidat SLM Sub-4B

| Model SLM | Rilis | Parameter | Hidden Dim ($d_{\text{model}}$) | Ukuran Kosakata | VRAM Bobot (BF16) | Skor MMLU / GSM8k / ARC | Headroom VRAM (RTX 3060 12GB) | Status Evaluasi |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-1.5B-Instruct** | 2024 | **1.54B** | **1536** | **151.643** | **~3.08 GB** | **68.2 / 74.8 / 55.4** | **~7.8 GB (Sangat Longgar & Aman)** | 🏆 **TERPILIH: MODEL FINAL UTAMA** |
| **Qwen2.5-3B-Instruct** | 2024 | **3.09B** | **2048** | **151.643** | **~6.18 GB** | **74.1 / 82.0 / 64.2** | ~4.6 GB (Cukup aman dengan micro-batch 1/2) | ⭐ **Varian Scaling Komparasi Opsional** |
| **Phi-4-mini-Instruct** | 2025 | 3.82B | 3072 | 100.352 | ~7.64 GB | 72.0 / 84.5 / 62.0 | ~3.1 GB (Sangat sempit, risiko OOM tinggi) | Tidak Disarankan untuk Training 12GB |
| **Llama-3.2-1B-Instruct** | 2024 | 1.23B | 2048 | 128.256 | ~2.46 GB | 49.3 / 44.4 / 41.2 | ~8.4 GB (Sangat longgar) | Skor penalaran terstruktur lebih rendah |
| **Gemma-2-2B-IT** | 2024 | 2.61B | 2304 | 256.000 | ~5.22 GB | 56.1 / 55.0 / 48.0 | ~5.6 GB (Cukup) | Logit soft-capping menambah kompleksitas |

---

### C. Keputusan Model Final Resmi (Final Model Decision)

Berdasarkan pertimbangan ilmiah dan rekayasa perangkat keras yang objektif:
> **MODEL FINAL RESMI YANG DIGUNAKAN:**  
> **`Qwen/Qwen2.5-1.5B-Instruct`**

#### Justifikasi Ilmiah & Komputasi:
1. **Safety Margin VRAM Sempurna di Lingkungan Windows (12 GB RTX 3060):**
   Dengan bobot model hanya ~3.08 GB, tersedia sisa memori **>7.5 GB**. Ini memberikan jaminan mutlak bahwa saat melatih proyektor (di mana gradien $\frac{\partial \mathcal{L}}{\partial \psi}$ harus melakukan *backpropagation* melintasi graf aktivasi 28 layer Transformer SLM), proses komputasi tidak akan pernah mengalami crash *CUDA Out-of-Memory*.
2. **Proporsi Dimensi Proyektor yang Elegan:**
   Dimensi representasi fisik kita adalah $384\text{d}$. Memetakan $384\text{d} \rightarrow 1024\text{d} \rightarrow 1536\text{d}$ menghasilkan lapisan MLP yang kompak (~1.97 Juta parameter), mencegah *overfitting* pada dataset instruksi dan mempercepat konvergensi hanya dalam 3–5 epoch.
3. **Kemampuan Format Terstruktur (JSON / Key-Value):**
   Qwen2.5-1.5B secara spesifik dioptimasi untuk penurutan instruksi terstruktur (*structured output generation*), sangat krusial untuk menghasilkan evaluasi objektif berbasis metrik (MAE, RMSE, F1).
4. **Opsi Scaling Eksperimental:**
   Sebagai bagian dari diskusi Tugas Akhir, kami juga menyiapkan konfigurasi modular yang memungkinkan pengujian scaling ke `Qwen/Qwen2.5-3B-Instruct` sebagai analisis tambahan (*ablation on model capacity*).

---

## 3. Arsitektur Teknis Sistem End-to-End

### A. Diagram Alur Data & Komponen

```text
                                         RANAH FISIK (FROZEN)
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ mmWave Radar Frame: P_t in R^{128 x 5} (x, y, z, Doppler, SNR)                                    │
│                                │                                                                   │
│                                ▼                                                                   │
│ [Model Av2: Point-MAE Backbone (Frozen)] ─────────▶ z_t (Current Physical Latent: 1 x 384)        │
│                                │                                                                   │
│                                ▼                                                                   │
│ [Temporal Sequence Buffer: t-15..t (16 frames)]                                                    │
│                                │                                                                   │
│                                ▼                                                                   │
│ [Temporal Dynamics Model (Frozen)] ───────────────▶ \hat{z}_{t+1:t+8} (Future Horizon: 8 x 384)   │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼ Token Packing
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ SPATIO-TEMPORAL PHYSICAL TOKENIZER:                                                                │
│ - [HISTORICAL MOTION TOKENS] : 4 tokens (ringkasan kinematika masa lalu dari encoder dynamics)     │
│ - [CURRENT STATE TOKEN]      : 1 token  (status fisik sekarang z_t dari Model Av2)                 │
│ - [FUTURE PREDICTION TOKENS] : 8 tokens (prediksi horizon masa depan \hat{z}_{t+1:t+8})            │
│ TOTAL INPUT FISIK            : 13 tokens in R^{13 x 384}                                           │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ TWO-LAYER MLP PROJECTOR WITH LAYER NORMALIZATION (TRAINABLE ~1.97M params)                         │
│   Linear(384, 1024) ──▶ LayerNorm(1024) ──▶ GELU ──▶ Dropout(0.1) ──▶ Linear(1024, 1536) ──▶ LN   │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼ 13 Pseudo-Tokens in R^{13 x 1536}
                                 │
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ FROZEN SLM (Qwen2.5-1.5B-Instruct, d_model = 1536)                                                │
│                                                                                                    │
│ Sequence Construction (Existing Vocabulary Delimiters):                                            │
│ <|im_start|>system                                                                                 │
│ You are a sensor-grounded physical reasoning assistant.<|im_end|>                                  │
│ <|im_start|>user                                                                                   │
│ [Physical Observations]: [13 Continuous Pseudo-Tokens Injected Directly]                           │
│ Question: "What is the subject's current posture and estimated depth distance from the sensor?"     │
│ <|im_end|>                                                                                         │
│ <|im_start|>assistant                                                                              │
│ Target Output (Structured Key-Value + Natural Language Explanation):                               │
│ posture: standing, depth_m: 2.73, active_limbs: right_arm ...                                      │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### B. Spesifikasi Two-Layer MLP Projector
```python
import torch
import torch.nn as nn

class PhysicalToLLMProjector(nn.Module):
    """
    Two-Layer MLP Projector with Layer Normalization and GELU activation.
    Maps 384-dimensional physical latent representations to the Qwen2.5-1.5B embedding space (1536d).
    Trainable parameters: ~1,971,968 (~1.97M)
    """
    def __init__(self, in_dim: int = 384, hidden_dim: int = 1024, out_dim: int = 1536, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch_Size, Num_Tokens=13, in_dim=384)
        # returns: (Batch_Size, Num_Tokens=13, out_dim=1536)
        return self.net(x)
```

### C. Delimiter Modalitas Tanpa Merusak Status Frozen SLM
- **Metode Baku:** Memanfaatkan token pembatas berbasis teks yang sudah ada dalam kosakata bawaan Qwen2.5:
  ```text
  [Physical Observations]: <13 continuous pseudo-tokens injected directly into input_embeds>
  ```
- **Keunggulan Metodologis:** Tidak mengubah ukuran matriks embedding (`token_embeddings`), tidak menambahkan bobot acak yang terlantar, dan menjamin seluruh parameter SLM berstatus beku murni (*strictly frozen 100%*).

---

## 4. Formalisasi Geometris: Sensor-Rig Coordinate Frame

Sistem menetapkan konvensi metrik berbasis **Sensor-Rig Coordinate System** yang terkalibrasi:
$$\mathbf{p} = (x, y, z) \in \mathbb{R}^3 \quad \text{(satuan meter)}$$

```text
                         Elevasi Vertikal (+Y)
                                  ▲
                                  │   Subjek Manusia
                                  │      O  [Kepala: Y ~ +0.6m]
                                  │     /|\ [Pelvis Root: X_p, Y_p, Z_p]
                                  │     / \
                                  │
                                  └─────────────────► Sisi Kanan (+X)
                                 /  Garis Tengah (X = 0)
                                /
                               ▼ Jarak Kedalaman Sensor (+Z: 1.5m s/d 4.5m)
                 ┌───────────────────────────┐
                 │    PLATFORM RIG SENSOR    │
                 │   [Radar mmWave + Kinect] │
                 └───────────────────────────┘
```

1. **Sumbu $Z$ (Depth / Jarak Kedalaman Sensor):**
   - Jarak ortogonal dari platform sensor ke subjek ($z > 0$).
   - Jarak kedalaman pusat tubuh: $z_{\text{depth}} = z_{\text{pelvis}}$.
   - Kecepatan radial translasi tubuh:
     $$v_z = \frac{z_{\text{pelvis}}(t) - z_{\text{pelvis}}(t-\Delta t)}{\Delta t}$$
   - **Aturan Arah Radial:**
     - Jika $v_z < -0.10\text{ m/s} \implies$ **Approaching** (bergerak mendekati radar).
     - Jika $v_z > +0.10\text{ m/s} \implies$ **Receding** (bergerak menjauhi radar).
     - Jika $|v_z| \le 0.10\text{ m/s} \implies$ **Stationary / In-place** (posisi radial konstan).
2. **Sumbu $X$ (Lateral / Horisontal):**
   - Posisi lateral terhadap garis tengah sensor ($x = 0$).
   - **Aturan Posisi Lateral:**
     - Jika $x_{\text{pelvis}} < -0.15\text{ m} \implies$ **Left side** (sisi kiri sensor).
     - Jika $|x_{\text{pelvis}}| \le 0.15\text{ m} \implies$ **Center** (tengah sensor).
     - Jika $x_{\text{pelvis}} > +0.15\text{ m} \implies$ **Right side** (sisi kanan sensor).
3. **Sumbu $Y$ (Elevasi Vertikal):**
   - Ketinggian tubuh relatif terhadap sensor ($y > 0$ mengarah ke atas/kepala, $y < 0$ mengarah ke lantai/kaki).
   - Postur berdiri vs jongkok ditentukan secara kuantitatif melalui elevasi vertikal pelvis $y_{\text{pelvis}}$ dan sudut fleksi lutut.

---

## 5. Metodologi Dataset Grounded-QA Otomatis (`generate_grounded_qa.py`)

Dataset instruksi teks diturunkan secara **deterministik dan matematis** langsung dari data fisik MM-Fi (`gt_skeleton`, `pred_skeleton`, `action_idx`, dan sampling temporal 10 Hz):

### A. Rincian 5 Kategori Tugas Inferensi

#### 1. Task 1 — Physical State Understanding (Postur & Ekstremitas Aktif)
- **Definisi Sendi Aktif:** Kecepatan temporal sendi ke-$j$:
  $$v_j(t) = \frac{\|\mathbf{p}_j(t) - \mathbf{p}_j(t-\Delta t)\|_2}{\Delta t}$$
  Sendi aktif jika $v_j(t) > \tau_{\text{active}}$, di mana threshold ditentukan strictly dari persentil empiris **training set**:
  $$\tau_{\text{active}} = P_{75}(v_j^{\text{train}})$$
- **Format Target:** `posture: <standing/squatting/lunging>, active_limbs: <upper_limbs/lower_limbs/both/none>`

#### 2. Task 2 — Spatial & Metric State Understanding (Koordinat Metrik Riil)
- Estimasi kedalaman: $z_{\text{depth}} = z_{\text{pelvis}}$ (satuan meter, 2 desimal).
- Posisi lateral: $x_{\text{lateral}} = x_{\text{pelvis}}$ (kiri / tengah / kanan).
- **Format Target:** `depth_m: <float>, lateral_position: <left/center/right>`

#### 3. Task 3 — Temporal Kinematics Inference (Arah Gerak & Kecepatan)
- Vektor kecepatan pusat tubuh: $\mathbf{v}_t = \frac{\mathbf{p}_{\text{pelvis}}(t) - \mathbf{p}_{\text{pelvis}}(t-\Delta t)}{\Delta t}$.
- Vektor percepatan: $\mathbf{a}_t = \frac{\mathbf{v}_t - \mathbf{v}_{t-\Delta t}}{\Delta t}$.
- **Format Target:** `radial_direction: <approaching/receding/stationary>, radial_velocity_mps: <float>`

#### 4. Task 4 — Inter-Limb Spatial Configuration Reasoning (Konfigurasi Antar-Sendi)
- Jarak antar pergelangan tangan: $d_{\text{wrists}} = \|\mathbf{p}_{\text{wrist\_L}}(t) - \mathbf{p}_{\text{wrist\_R}}(t)\|_2$.
- Jarak antar pergelangan kaki: $d_{\text{ankles}} = \|\mathbf{p}_{\text{ankle\_L}}(t) - \mathbf{p}_{\text{ankle\_R}}(t)\|_2$.
- **Format Target:** `wrist_distance_m: <float>, arm_configuration: <extended/contracted/wide>`

#### 5. Task 5 — Future-State Predictive Physical Reasoning (Peramalan Masa Depan)
- Memprediksi status kinematika pada horizon masa depan $t+8$ (0.8 detik ke depan) berdasarkan perpaduan $z_t$ dan $\hat{z}_{t+1:t+8}$.
- Displasemen masa depan: $\Delta \mathbf{p}_{\text{future}} = \mathbf{p}_{\text{pelvis}}(t+8) - \mathbf{p}_{\text{pelvis}}(t)$.
- Input model: $\hat{z}_{t+1:t+8}$ (prediksi Dynamics Model). Target evaluasi: $GT_{t+8}$ (ground truth aktual).
- **Format Target:** `future_radial_direction: <approaching/receding/stationary>, future_depth_m: <float>`

### B. Kueri Komposisional & Generalisasi Linguistik (Unseen Templates)
1. **Multi-Attribute Compositional Queries:** Pertanyaan gabungan yang memerlukan evaluasi multi-status sekaligus (misal: perbandingan kecepatan lengan kanan terhadap pergeseran pelvis saat tubuh bergerak mendekat).
2. **Disjoint Linguistic Templates (Train vs Test):** Struktur kalimat dan variasi frasa pada test set sepenuhnya independen dari data latih guna menguji generalisasi bahasa dan mencegah hafalan kata.

---

## 6. Protokol Partisi Subjek: Jaminan Nol Kebocoran Data (Zero-Leakage)

Pipeline penelitian ini mematuhi protokol **strictly cross-subject end-to-end**:

```text
                           ALOKASI PARTISI SUBJEK KONSISTEN
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 24 TRAINING SUBJECTS (Digunakan oleh Encoder Av2, Dynamics Model, dan Projector Stage 4):       │
│ S01, S02, S03, S06, S08, S09, S11, S12, S14, S15, S16, S19,                                     │
│ S21, S23, S26, S27, S29, S30, S31, S32, S33, S37, S38, S39                                     │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 8 VALIDATION SUBJECTS (Hanya untuk penentuan epoch terbaik dan early stopping):                 │
│ S05, S10, S18, S20, S24, S28, S34, S35                                                          │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 8 HELD-OUT TEST SUBJECTS (TIDAK PERNAH DILIHAT SAMA SEKALI OLEH ENCODER, DYNAMICS, & PROJECTOR): │
│ S04, S07, S13, S17, S22, S25, S36, S40                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Protokol Komputasi & Mekanika Gradien VRAM (RTX 3060 12GB)

### A. Alur Gradien pada Frozen SLM
- Bobot SLM dibekukan (`param.requires_grad = False`).
- Graf aktivasi forward pass SLM **tetap dipertahankan** agar gradien loss dapat mengalir mundur ke proyektor:
  $$\frac{\partial \mathcal{L}}{\partial \psi} = \frac{\partial \mathcal{L}}{\partial \mathbf{E}_{\text{phys}}} \cdot \frac{\partial \mathbf{E}_{\text{phys}}}{\partial \psi}$$
  *(Dilarang menggunakan `torch.no_grad()` pada forward pass SLM saat pelatihan proyektor).*

### B. Proteksi VRAM & Estimasi Rekayasa:
1. **Gradient Checkpointing:** Diaktifkan pada SLM (`model.gradient_checkpointing_enable()`), memangkas memori aktivasi >60%.
2. **Conservative Micro-Batching:** Micro-batch $B = 2$ dipadukan dengan **Gradient Accumulation Steps = 8** (*effective batch size* = 16).
3. **Presisi:** `bfloat16`.
4. **Pre-Experiment Engineering Estimate:**
   - Bobot Model Qwen2.5-1.5B (BF16): ~3.08 GB
   - Memori Aktivasi (Gradient Checkpointing, $B=2$, seq=180): ~2.50 GB
   - Bobot Proyektor & Optimizer AdamW (2M params): ~0.05 GB
   - CUDA Context & Static Buffers: ~1.20 GB
   - **Estimasi Total Peak Memory:** **~6.83 GB s/d ~7.00 GB**
   - *Catatan:* Nilai riil memori puncak (*actual peak VRAM*) akan diukur secara empiris menggunakan `torch.cuda.max_memory_allocated()` saat *smoke test*.

---

## 8. Matriks Baseline Ilmiah & Protokol Evaluasi Objektif

### A. Matriks 5 Model Pembanding (B1 s/d B5)

| Kode | Nama Model / Varian | Input Representasi Fisik | Modul Penalaran / Classifier | Peran Ilmiah & Pertanyaan Riset yang Dijawab |
| :---: | :--- | :---: | :---: | :--- |
| **B1** | **Blind Text-Only SLM** | Tanpa input fisik | Frozen SLM (Zero-shot) | Seberapa besar bias tebakan teks murni tanpa sensor? |
| **B2** | **Direct Linear/MLP Probe** | $\mathbf{z}_t$ (384d) | Linear / 2-Layer MLP Probe | Apakah SLM memberikan nilai tambah komposisional bahasa dibanding pengklasifikasi numerik biasa? |
| **B3** | **No-Future Pipeline** | $\mathbf{z}_{\text{hist}} + \mathbf{z}_t$ | Projector + Frozen SLM | Bagaimana performa inferensi jika modul prediksi masa depan Tahap 3 ditiadakan? |
| **B4** | **Proposed Full Pipeline** | $\mathbf{z}_{\text{hist}} + \mathbf{z}_t + \mathbf{\hat{z}}_{\text{future}}$ | Projector + Frozen SLM | **Model Utama yang Diusulkan (Tahap 4)** |
| **B5** | **Oracle Upper Bound** | $\mathbf{z}_{\text{hist}} + \mathbf{z}_t + \mathbf{z}_{\text{future}}^{GT}$ | Projector + Frozen SLM | **Batas Atas Teoretis:** Mengisolasi apakah error inferensi masa depan berasal dari Dynamics Model atau dari Projector/SLM |

### B. Uji Dependensi Grounding (Empirical Verification of Sensor Dependence)
Menguji ketergantungan empiris prediksi model terhadap representasi sensor ($Y \not\!\perp\!\!\!\perp Z_{\text{physical}}$ secara statistical dependence):
1. **Cross-Action Token Shuffling (Kontrol Kasar):** Menyuntikkan token fisik dari rekaman aksi yang berlawanan (*Squat* vs *Punch*).
2. **Within-Action Token Shuffling (Kontrol Halus):** Menyuntikkan token fisik dari subjek lain yang melakukan aksi yang sama (*S04 Squat* vs *S07 Squat* yang berbeda metrik kecepatan dan kedalaman).

### C. Metrik Evaluasi Objektif Terstruktur
- **Kategorikal:** *Classification Accuracy (%)* dan *Macro-F1 Score*.
- **Estimasi Metrik:** *Mean Absolute Error (MAE)* dan *Root Mean Squared Error (RMSE)* dalam meter atau m/s.
- **Peramalan Horizon:** *Directional F1 Score* dan *Average Displacement Error (ADE)* pada horizon $t+8$.
- **Resampling Statistik:** Bootstrap 95% Confidence Interval dihitung pada **Sequence / Subject Level**.

---

## 9. Rencana Berkas & Arsitektur Kode Baru

```text
eksperimen_model/
├── configs/
│   └── mmfi_projector_qwen.yaml          <-- [BARU] Konfigurasi lengkap Stage 4 (Qwen2.5-1.5B)
├── datasets/
│   └── generate_grounded_qa.py           <-- [BARU] Skrip generator otomatis dataset QA deterministik
├── models/
│   └── projector.py                      <-- [BARU] Modul PhysicalToLLMProjector & PhysicalSLMWrapper
├── train_projector.py                    <-- [BARU] Skrip pelatihan Stage 4 (MLP Projector training)
├── evaluate_reasoning.py                 <-- [BARU] Skrip evaluasi komprehensif benchmark B1-B5 & Shuffling
└── test_projector_pipeline.py            <-- [BARU] Smoke test sanity check & VRAM benchmarking
```

---

## 10. Protokol Verifikasi Eksekusi Bertahap

Setelah persetujuan pengguna diberikan, alur kerja eksekusi teknis dilakukan secara berurutan:

1. **Langkah 1: Pembuatan Skrip Generator Dataset QA (`generate_grounded_qa.py`)**
   - Menghasilkan file JSONL: `mmfi_grounded_qa_train.jsonl`, `val.jsonl`, dan `test.jsonl`.
2. **Langkah 2: Setup Dependensi & Smoke Test (`test_projector_pipeline.py`)**
   - Menginstal `transformers>=4.45.0` dan `accelerate>=0.34.0` di `.venv`.
   - Menguji pemuatan Qwen2.5-1.5B-Instruct, verifikasi koneksi graf gradien ke proyektor, aktivasi gradient checkpointing, dan mengukur *actual peak VRAM* menggunakan `torch.cuda.max_memory_allocated()`.
3. **Langkah 3: Pelatihan Direct Probe Baseline (B2)**
   - Melatih MLP classifier/regressor sederhana langsung dari $Z_t$ sebagai pembanding performa dasar.
4. **Langkah 4: Pelatihan MLP Projector Tahap 4 (`train_projector.py`)**
   - Melatih proyektor selama 3–5 epoch dengan micro-batching $B=2$, gradient accumulation 8, dan scheduler Cosine Annealing dengan warmup.
5. **Langkah 5: Evaluasi Lengkap & Ablation Benchmark (`evaluate_reasoning.py`)**
   - Menjalankan pengujian berjenjang pada test set unseen (B1, B2, B3, B4, B5).
   - Menjalankan uji kontrol negatif (Cross-action & Within-action shuffling).
   - Menghasilkan tabel metrik akademik lengkap dan kurva konvergensi.
