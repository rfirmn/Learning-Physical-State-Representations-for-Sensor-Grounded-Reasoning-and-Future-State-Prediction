# Optimasi Pelatihan Tahap 2: Menerobos Plateau MPJPE 243mm

## Latar Belakang & Diagnosis Masalah

Eksperimen terbaru (`RUN_20260911_172432`) mencapai **Best Val MPJPE = 243.2 mm** pada Epoch 16, namun kemudian stagnan di plateau 243–249 mm selama 14 epoch tersisa. Generalization gap melebar dari 49 mm (Epoch 16) → 80 mm (Epoch 30), menandakan **overfitting progresif** tanpa peningkatan generalisasi.

### Akar Penyebab yang Teridentifikasi dari Analisis Mendalam:

| # | Bottleneck | Bukti | Dampak Estimasi |
|---|---|---|---|
| 1 | **Pose Head terlalu dangkal** — hanya 1 hidden layer (384→512→51) tanpa bottleneck per-joint | Semua 17 sendi dipaksa keluar dari 1 vektor global, tidak ada mekanisme per-joint attention | Tinggi |
| 2 | **Hanya mean-pooling** — `Z_t = mean(tokens)` membuang informasi spasial lokal antar patch | Wrist error 349 mm vs Pelvis 208 mm; model gagal memanfaatkan token lokal untuk sendi perifer | Tinggi |
| 3 | **Augmentasi terlalu lemah** — hanya rotasi ±15°, jitter σ=0.01, scale 0.95–1.05 | Generalization gap 80 mm, E04 degradasi +31% | Tinggi |
| 4 | **Tidak ada regularisasi serius** — Dropout hanya 0.1 di pose head, tidak ada di encoder | Gap train-val melebar monoton setelah Epoch 8 | Sedang-Tinggi |
| 5 | **LR Schedule tanpa warmup efektif** — Cosine langsung dari LR maks tanpa warmup bertahap | Epoch 1–3 sudah di LR tinggi, potensi merusak bobot pretrained yang halus | Sedang |
| 6 | **Loss function MPJPE murni** — tidak ada insentif struktural/anatomis untuk menjaga koherensi skeleton | Wrist/Elbow bisa melompat independen dari Shoulder tanpa penalti | Sedang |
| 7 | **Tidak memanfaatkan fitur Doppler & SNR** — input hanya xyz, padahal radar menyediakan 5 channel (x,y,z,vd,snr) | Informasi kecepatan radial (Doppler) sangat berguna untuk membedakan sendi yang bergerak | Sedang |

---

## User Review Required

> [!IMPORTANT]
> **Estimasi Waktu Pelatihan:** Dengan konfigurasi yang diusulkan (100 epoch, batch 32, ~180 detik/epoch), total pelatihan akan memakan waktu **~5 jam**. Apakah ini dapat diterima?

> [!IMPORTANT]
> **Strategi Fitur Doppler/SNR:** Saya merencanakan penambahan channel input dari 3 (xyz) menjadi 5 (xyz + doppler + snr). Ini memerlukan modifikasi pada `MiniPointNetPatchEmbed` (parameter `in_channel=5`). Karena bobot pretrained ShapeNet hanya untuk `in_channel=3`, layer pertama Conv1d(3→128) harus di-reinisialisasi sebagian menjadi Conv1d(5→128). Bobot 3 channel lama akan dipertahankan, 2 channel baru diinisialisasi secara Xavier. **Apakah Anda menyetujui pendekatan ini, atau lebih memilih hanya menggunakan xyz?**

## Open Questions

> [!NOTE]
> **Pilihan per-Joint Attention vs. Multi-Head Regression:**
> Saya akan mengimplementasikan **Joint-Query Cross-Attention Head** — di mana 17 learnable query (satu per sendi) melakukan cross-attention terhadap 16 patch tokens. Pendekatan ini lebih powerful dibanding MLP linear karena setiap sendi bisa "memilih" patch mana yang paling relevan (misal: Wrist akan attend ke patch di area tangan). Pendekatan alternatif adalah multi-head regression biasa dengan hidden layer lebih dalam. Mana yang Anda preferensikan?

---

## Proposed Changes

Perubahan dikelompokkan dalam **7 komponen** yang saling terkait, diurutkan dari dependensi terendah ke tertinggi.

---

### Komponen 1: Enhanced Data Augmentation Pipeline

Augmentasi saat ini terlalu lemah dan menjadi penyebab utama overfitting. Kita perlu augmentasi yang lebih agresif dan realistis secara fisika radar.

#### [MODIFY] [transforms.py](file:///c:/Users/Rio%20Aslab/Documents/pemrograman/Tugas_Akhir/eksperimen_model/datasets/transforms.py)

Peningkatan pada class `PointCloudAugment`:

| Augmentasi | Sekarang | Sesudah | Alasan Fisika |
|---|---|---|---|
| **Rotasi Z-axis** | ±15° | ±45° | Subjek bisa menghadap berbagai arah relatif terhadap radar |
| **Rotasi X-axis (tilt)** | Tidak ada | ±5° | Simulasi posisi sensor yang sedikit miring |
| **Gaussian Jitter** | σ=0.01, clip ±0.03 | σ=0.02, clip ±0.05 | Lebih realistis untuk noise fase radar mmWave |
| **Anisotropic Scale** | Uniform 0.95–1.05 | Per-axis [0.90–1.10] | Perbedaan postur tinggi/lebar tubuh subjek |
| **Random Point Dropout** | Tidak ada | Drop 10–30% titik secara acak, lalu duplikasi kembali ke N=128 | Simulasi oklusi parsial & pantulan lemah |
| **Random Translation** | Tidak ada | Geser ±0.1 unit (post-normalisasi) | Subjek tidak selalu di pusat FoV radar |
| **MixUp Point Cloud** | Tidak ada | 20% probabilitas blend 2 point cloud | Regularisasi geometris cross-sample |

Augmentasi juga harus diterapkan secara **konsisten** ke skeleton ground truth (rotasi, scale, translation harus identik untuk xyz dan skeleton):

```python
class PointCloudAugment:
    def __call__(self, pts, skeleton=None):
        # ... transformasi ...
        if skeleton is not None:
            skeleton = apply_same_transform(skeleton, rot_matrix, scale, translation)
        return pts, skeleton
```

> [!WARNING]
> **Perubahan Breaking pada Interface:** `PointCloudAugment.__call__()` sekarang menerima dan mengembalikan tuple `(pts, skeleton)` alih-alih hanya `pts`. Semua pemanggil harus diperbarui.

---

### Komponen 2: Multi-Channel Radar Input (Doppler + SNR)

#### [MODIFY] [mmfi_dataset.py](file:///c:/Users/Rio%20Aslab/Documents/pemrograman/Tugas_Akhir/eksperimen_model/datasets/mmfi_dataset.py)

Saat ini `__getitem__` memotong input menjadi `xyz = pc[:, :3]`, membuang channel Doppler (indeks 3) dan SNR (indeks 4). Perubahan:

```python
# Sebelum:
xyz = pc[:, :3]

# Sesudah:
if self.use_extra_features:
    features = pc[:, :5]  # x, y, z, doppler, snr
    # Normalize doppler dan snr ke range [-1, 1]
    features[:, 3] = (features[:, 3] - features[:, 3].mean()) / (features[:, 3].std() + 1e-8)
    features[:, 4] = (features[:, 4] - features[:, 4].mean()) / (features[:, 4].std() + 1e-8)
else:
    features = pc[:, :3]
```

- Menambahkan parameter `use_extra_features: bool = True` ke constructor.
- Augmentasi hanya diterapkan pada 3 channel xyz, channel Doppler & SNR tidak diputar/diskala.
- Normalisasi tetap hanya pada xyz (unit sphere).

---

### Komponen 3: Joint-Query Cross-Attention Pose Head

Ini adalah perubahan arsitektur paling signifikan. Alih-alih mean-pooling seluruh patch menjadi 1 vektor lalu regresi linear, kita membuat 17 **learnable joint queries** yang masing-masing melakukan cross-attention terhadap 16 patch tokens.

#### [MODIFY] [point_mae_encoder.py](file:///c:/Users/Rio%20Aslab/Documents/pemrograman/Tugas_Akhir/eksperimen_model/models/point_mae_encoder.py)

**Arsitektur baru Pose Head:**

```python
class JointQueryPoseHead(nn.Module):
    """
    Cross-Attention Pose Head: 17 learnable joint queries attend to G patch tokens.
    Each query independently attends to spatially relevant patches.
    """
    def __init__(self, embed_dim=384, num_joints=17, num_heads=6, depth=2, dropout=0.3):
        super().__init__()
        self.joint_queries = nn.Parameter(torch.randn(1, num_joints, embed_dim) * 0.02)
        
        self.cross_attn_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=embed_dim * 2,
                dropout=dropout, batch_first=True
            ) for _ in range(depth)
        ])
        
        self.joint_regressor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)  # Setiap query → (x, y, z) untuk 1 sendi
        )
    
    def forward(self, patch_tokens):
        # patch_tokens: (B, G=16, embed_dim=384)
        B = patch_tokens.shape[0]
        queries = self.joint_queries.expand(B, -1, -1)  # (B, 17, 384)
        
        for layer in self.cross_attn_layers:
            queries = layer(queries, patch_tokens)  # Cross-attend
        
        joints = self.joint_regressor(queries)  # (B, 17, 3)
        return joints
```

**Perubahan pada `PointMAEPoseEstimator`:**

```python
def forward(self, xyz):
    z_t, patch_tokens = self.extract_features(xyz)  # z_t: (B, 384), patch_tokens: (B, 16, 384)
    pred_joints = self.pose_head(patch_tokens)  # Bukan z_t lagi, tapi patch_tokens!
    return pred_joints, z_t
```

Keunggulan:
- **Wrist query** dapat attend ke patch di area perifer → mengurangi error ~349mm.
- **Pelvis query** dapat attend ke patch pusat → mempertahankan akurasi ~208mm.
- Setiap sendi memiliki **head regressor independen** (shared weights tapi input berbeda).

#### [MODIFY] [point_mae.py](file:///c:/Users/Rio%20Aslab/Documents/pemrograman/Tugas_Akhir/eksperimen_model/models/point_mae.py)

Modifikasi `MiniPointNetPatchEmbed` untuk mendukung `in_channel=5`:

```python
class MiniPointNetPatchEmbed(nn.Module):
    def __init__(self, in_channel: int = 3, out_dim: int = 384):
        # Conv1d(in_channel, 128, 1) — in_channel sekarang bisa 3 atau 5
```

Bobot pretrained Conv1d(3→128) akan di-expand secara cerdas: 3 channel pertama mempertahankan bobot ShapeNet, 2 channel tambahan diinisialisasi Xavier.

#### [MODIFY] [__init__.py (models)](file:///c:/Users/Rio%20Aslab/Documents/pemrograman/Tugas_Akhir/eksperimen_model/models/__init__.py)

Tambahkan export `JointQueryPoseHead`.

---

### Komponen 4: Composite Loss Function (MPJPE + Bone Length + Velocity Smoothness)

#### [MODIFY] [losses.py](file:///c:/Users/Rio%20Aslab/Documents/pemrograman/Tugas_Akhir/eksperimen_model/models/losses.py)

Menambahkan 2 loss tambahan yang menjaga **koherensi struktural anatomis**:

```python
class BoneLengthLoss(nn.Module):
    """
    Menjaga konsistensi panjang tulang antar prediksi dan ground truth.
    Bone AB: || ||pred_A - pred_B|| - ||gt_A - gt_B|| ||
    Prevents anatomically impossible skeleton deformations.
    """
    BONE_PAIRS = [
        (0, 7), (7, 8), (8, 9), (9, 10),  # Spine chain
        (0, 1), (1, 2), (2, 3),             # Right leg
        (0, 4), (4, 5), (5, 6),             # Left leg
        (8, 11), (11, 12), (12, 13),        # Left arm
        (8, 14), (14, 15), (15, 16)         # Right arm
    ]

class WeightedMPJPELoss(nn.Module):
    """
    Per-joint weighted MPJPE. Memberikan bobot lebih tinggi pada sendi 
    yang errornya tinggi (wrist, elbow) agar optimizer lebih fokus ke sana.
    """
    # Bobot adaptif berdasarkan inverse error dari eksperimen sebelumnya

class CompositePoseLoss(nn.Module):
    """
    Total Loss = α * WeightedMPJPE + β * BoneLengthLoss
    Default: α=1.0, β=0.5
    """
```

**Rasional:** Loss MPJPE murni tidak memberikan penalti jika skeleton terdistorsi secara struktural (misal: lengan memanjang 2x). `BoneLengthLoss` memaksa proporsionalitas anatomis tetap terjaga.

**Joint Weights (berdasarkan inverse normalized error dari run sebelumnya):**

| Kelompok | Sendi | Bobot |
|---|---|---|
| Upper Limbs (error tinggi) | Wrists, Elbows | **1.5× – 2.0×** |
| Central Body (error sedang) | Neck, Head, Thorax | **1.0×** |
| Lower Limbs (error rendah) | Hips, Knees, Ankles | **0.8×** |

---

### Komponen 5: Advanced Training Strategy (LR, Regularization, EMA)

#### [MODIFY] [train_pose.py](file:///c:/Users/Rio%20Aslab/Documents/pemrograman/Tugas_Akhir/eksperimen_model/train_pose.py)

**5a. Layer-wise Learning Rate Decay (LLRD):**
Menerapkan learning rate yang berbeda per kedalaman Transformer layer:

```python
def get_layer_wise_lr_groups(model, base_lr, lr_decay=0.75):
    """
    Layer 1 (terdalam/paling awal) → lr * decay^11
    Layer 12 (teratas)             → lr * decay^0 = lr
    Pose Head                       → lr * 2 (lebih agresif)
    """
```

Ini mencegah layer awal (yang menyimpan pengetahuan geometris ShapeNet paling fundamental) dari perubahan berlebihan.

**5b. Cosine Annealing with Warm Restarts:**

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-6
)
```

Restart pada epoch 20 dan 60 memungkinkan model "melarikan diri" dari minimum lokal.

**5c. Exponential Moving Average (EMA):**

```python
class EMAModel:
    """Maintains shadow weights for smoother evaluation."""
    def __init__(self, model, decay=0.999):
        self.shadow = {k: v.clone() for k, v in model.state_dict().items()}
        self.decay = decay
    
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v
```

**5d. Stochastic Depth (DropPath) pada Transformer Blocks:**

Menambahkan `drop_path_rate=0.1` secara bertingkat pada 12 Transformer Blocks di encoder, meningkat linear dari 0.0 (Block 1) ke 0.1 (Block 12).

**5e. Gradient Accumulation:**

Untuk efektif meningkatkan batch size dari 32 → 64 tanpa menambah VRAM:

```python
accumulation_steps = 2
loss = loss / accumulation_steps
loss.backward()
if (batch_idx + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

---

### Komponen 6: Konfigurasi Hyperparameter Baru

#### [NEW] [mmfi_pose_v2.yaml](file:///c:/Users/Rio%20Aslab/Documents/pemrograman/Tugas_Akhir/eksperimen_model/configs/mmfi_pose_v2.yaml)

```yaml
experiment_name: "point_mae_mmfi_pose_v2_optimized"

dataset:
  root_dir: "datasets/MM-Fi Dataset/filtered_mmwave"
  segments_csv: "datasets/MM-Fi Dataset/MMFi_action_segments.csv"
  num_points: 128
  environments: ["E01", "E02", "E03", "E04"]
  protocol: "protocol3"
  use_extra_features: true   # <<< BARU: Doppler + SNR
  split:
    type: "stratified_random_cross_subject"
    split_seed: 42
    train_subjects: [S01,S02,S03,S06,S08,S09, S11,S12,S14,S15,S16,S19, S21,S23,S26,S27,S29,S30, S31,S32,S33,S37,S38,S39]
    val_subjects: [S05,S10, S18,S20, S24,S28, S34,S35]
    test_subjects: [S04,S07, S13,S17, S22,S25, S36,S40]

model:
  pretrained_weights: "models/Point-MAE/pretrain.pth"
  num_points: 128
  num_groups: 16
  group_size: 16
  embed_dim: 384
  depth: 12
  num_heads: 6
  num_joints: 17
  in_channels: 5             # <<< BARU: xyz + doppler + snr
  freeze_encoder: false
  pose_head_type: "cross_attention"  # <<< BARU: "mlp" atau "cross_attention"
  pose_head_depth: 2         # <<< BARU: jumlah cross-attention layers
  pose_head_dropout: 0.3     # <<< BARU

training:
  batch_size: 32
  effective_batch_size: 64   # <<< BARU: via gradient accumulation (steps=2)
  num_workers: 4
  epochs: 100                # <<< BARU: dari 30 → 100
  lr: 0.0003                 # <<< BARU: sedikit lebih tinggi
  min_lr: 0.000001           # <<< BARU: lebih rendah
  weight_decay: 0.05
  warmup_epochs: 5           # <<< BARU: dari 3 → 5
  lr_layer_decay: 0.75       # <<< BARU: LLRD
  lr_schedule: "cosine_warm_restarts"  # <<< BARU
  lr_restart_period: 20      # <<< BARU: T_0
  loss_type: "composite"     # <<< BARU: weighted_mpjpe + bone_length
  loss_bone_weight: 0.5      # <<< BARU: β coefficient
  joint_loss_weights: "adaptive"  # <<< BARU: bobot per sendi
  gradient_clip_val: 1.0
  gradient_accumulation_steps: 2  # <<< BARU
  drop_path_rate: 0.1        # <<< BARU: stochastic depth
  ema_decay: 0.999           # <<< BARU: EMA
  seed: 42

augmentation:                # <<< BARU: seluruh section
  rotate_z_range: 45         # derajat
  rotate_x_range: 5          # derajat
  jitter_sigma: 0.02
  jitter_clip: 0.05
  scale_range: [0.90, 1.10]
  anisotropic_scale: true
  point_dropout_range: [0.1, 0.3]
  random_translation: 0.1
  mixup_prob: 0.2
```

---

### Komponen 7: Script Training v2 & Integrasi

#### [NEW] [train_pose_v2.py](file:///c:/Users/Rio%20Aslab/Documents/pemrograman/Tugas_Akhir/eksperimen_model/train_pose_v2.py)

Script training baru yang mengintegrasikan semua perubahan di atas. Script ini **tidak menggantikan** `train_pose.py` yang sudah ada, melainkan merupakan versi terpisah agar eksperimen lama tetap reproducible.

Fitur utama `train_pose_v2.py`:
1. Layer-wise LR groups via `get_layer_wise_lr_groups()`
2. `CosineAnnealingWarmRestarts` scheduler
3. Gradient accumulation loop
4. EMA model update setiap batch
5. Evaluasi menggunakan EMA weights
6. Composite loss (WeightedMPJPE + BoneLengthLoss)
7. Augmentasi konsisten pts↔skeleton
8. Early stopping patience = 25 epoch (mencegah training sia-sia jika plateau lama)
9. Tetap kompatibel dengan `ExperimentReporter` untuk otomatisasi laporan
10. Multi-channel input support (5D)

#### [MODIFY] [checkpoint.py](file:///c:/Users/Rio%20Aslab/Documents/pemrograman/Tugas_Akhir/eksperimen_model/utils/checkpoint.py)

Modifikasi `load_pretrained_point_mae()` untuk menangani ekspansi channel dari 3 → 5 pada `MAE_encoder.encoder.first_conv.0.weight` (Conv1d kernel shape berubah dari `[128, 3, 1]` → `[128, 5, 1]`):

```python
# Jika shape mismatch karena channel expansion:
if 'first_conv.0.weight' in name and ckpt_val.shape[1] != param.shape[1]:
    # Pertahankan bobot 3 channel, inisialisasi channel baru dengan Xavier
    new_weight = torch.zeros_like(param)
    new_weight[:, :ckpt_val.shape[1], :] = ckpt_val
    nn.init.xavier_uniform_(new_weight[:, ckpt_val.shape[1]:, :])
    matched_dict[name] = new_weight
```

---

## Ringkasan Perubahan File

| Tipe | File | Deskripsi Singkat |
|---|---|---|
| **MODIFY** | [`transforms.py`](file:///c:/Users/Rio%20Aslab/Documents/pemrograman/Tugas_Akhir/eksperimen_model/datasets/transforms.py) | Augmentasi agresif: rotasi ±45°, dropout titik, anisotropic scale, MixUp, co-transform skeleton |
| **MODIFY** | [`mmfi_dataset.py`](file:///c:/Users/Rio%20Aslab/Documents/pemrograman/Tugas_Akhir/eksperimen_model/datasets/mmfi_dataset.py) | Dukungan input 5-channel (xyz+doppler+snr), interface augmentasi baru |
| **MODIFY** | [`point_mae.py`](file:///c:/Users/Rio%20Aslab/Documents/pemrograman/Tugas_Akhir/eksperimen_model/models/point_mae.py) | `MiniPointNetPatchEmbed(in_channel=5)`, DropPath pada Block |
| **MODIFY** | [`point_mae_encoder.py`](file:///c:/Users/Rio%20Aslab/Documents/pemrograman/Tugas_Akhir/eksperimen_model/models/point_mae_encoder.py) | `JointQueryPoseHead`, forward menggunakan patch tokens bukan mean-pooled |
| **MODIFY** | [`losses.py`](file:///c:/Users/Rio%20Aslab/Documents/pemrograman/Tugas_Akhir/eksperimen_model/models/losses.py) | `WeightedMPJPELoss`, `BoneLengthLoss`, `CompositePoseLoss` |
| **MODIFY** | [`checkpoint.py`](file:///c:/Users/Rio%20Aslab/Documents/pemrograman/Tugas_Akhir/eksperimen_model/utils/checkpoint.py) | Channel expansion handling pada load pretrained |
| **MODIFY** | [`__init__.py` (models)](file:///c:/Users/Rio%20Aslab/Documents/pemrograman/Tugas_Akhir/eksperimen_model/models/__init__.py) | Export class baru |
| **MODIFY** | [`__init__.py` (datasets)](file:///c:/Users/Rio%20Aslab/Documents/pemrograman/Tugas_Akhir/eksperimen_model/datasets/__init__.py) | Export augmentasi baru |
| **NEW** | [`mmfi_pose_v2.yaml`](file:///c:/Users/Rio%20Aslab/Documents/pemrograman/Tugas_Akhir/eksperimen_model/configs/mmfi_pose_v2.yaml) | Konfigurasi hyperparameter baru |
| **NEW** | [`train_pose_v2.py`](file:///c:/Users/Rio%20Aslab/Documents/pemrograman/Tugas_Akhir/eksperimen_model/train_pose_v2.py) | Script training dengan semua optimasi terintegrasi |

---

## Prediksi Dampak Kuantitatif per Teknik

| Teknik | Target Penurunan MPJPE | Mekanisme |
|---|---|---|
| Joint-Query Cross-Attention Head | -30 ~ -50 mm | Sendi perifer (Wrist) bisa attend ke patch lokal yang relevan |
| Enhanced Augmentation + Dropout | -15 ~ -25 mm | Mengurangi overfitting, menutup generalization gap |
| Composite Loss (Bone Length) | -10 ~ -15 mm | Menjaga koherensi struktural, mengurangi outlier |
| Layer-wise LR Decay | -5 ~ -10 mm | Melindungi bobot fondasi ShapeNet di layer awal |
| Doppler + SNR Input | -5 ~ -15 mm | Informasi velocity membantu disambiguasi sendi yang bergerak cepat |
| EMA + Longer Training (100 ep) | -5 ~ -10 mm | Rata-rata bobot yang lebih stabil, eksplorasi lebih lama |
| **Total Estimasi** | **-70 ~ -125 mm** | **Target: 120–175 mm (12–17.5 cm)** |

---

## Verification Plan

### Automated Tests

1. **Sanity Check Pipeline** (sebelum training skala penuh):
   ```powershell
   & ".venv\Scripts\python.exe" eksperimen_model/test_pipeline.py
   ```

2. **Quick Debug Run** (2 epoch, 2 subjek, validasi shapes & gradient flow):
   ```powershell
   & ".venv\Scripts\python.exe" eksperimen_model/train_pose_v2.py --config eksperimen_model/configs/mmfi_pose_v2.yaml --epochs 2 --train_sub S01 S02 --val_sub S03 --env E01
   ```

3. **Full Training Run** (100 epoch, semua data):
   ```powershell
   & ".venv\Scripts\python.exe" eksperimen_model/train_pose_v2.py --config eksperimen_model/configs/mmfi_pose_v2.yaml
   ```

4. **Post-Training Evaluation**:
   ```powershell
   & ".venv\Scripts\python.exe" eksperimen_model/evaluate_pose.py --checkpoint eksperimen_model/checkpoints/pose_estimation_v2/best_model.pth
   ```

### Manual Verification

- Bandingkan laporan otomatis di `docs/report_training/INDEX.md` antara Run v1 (243 mm) dan Run v2 (target <175 mm).
- Periksa apakah E04 gap mengecil (target: deviasi <15% dari rerata, bukan +31%).
- Verifikasi visual pada `skeleton_3d_comparison.png` bahwa prediksi Wrist/Elbow lebih presisi.
- Periksa `loss_curve.png` bahwa generalization gap stabil di <30 mm.
