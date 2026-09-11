# Filtered mmWave Radar Dataset (MM-Fi Adapter & Pipeline Integration)

Repositori lokal ini memuat data mmWave radar terfilter (*filtered point cloud*) dari **MM-Fi Dataset** (*NeurIPS 2023*). Data disiapkan secara khusus untuk memfasilitasi dua tahap awal pada proyek riset **Pemodelan Representasi Fisik Spasial-Temporal dan Penyelarasan LLM**:

1. **Fase 2:** *Fine-tuning* Sensor Encoder (Point-MAE) untuk estimasi 3D Human Skeleton Pose.
2. **Fase 3:** Pelatihan Dynamics Model (Temporal Transformer/ST-GCN) untuk peramalan lintasan fisik ($Z_{t+1:t+h}$).

```text
C:\Users\Rio Aslab\documents\pemrograman\tugas_akhir\datasets\filtered_mmwave

```

---

## 📋 Daftar Isi

* [1. Ikhtisar Dataset](https://www.google.com/search?q=%231-ikhtisar-dataset)
* [2. Hierarki dan Struktur Direktori](https://www.google.com/search?q=%232-hierarki-dan-struktur-direktori)
* [3. Rincian Lingkungan Eksperimen (E01 - E04)](https://www.google.com/search?q=%233-rincian-lingkungan-eksperimen-e01---e04)
* [4. Rincian Aksi dan Aktivitas Subjek (A01 - A27)](https://www.google.com/search?q=%234-rincian-aksi-dan-aktivitas-subjek-a01---a27)
* [5. Format Data & Spesifikasi Tensor](https://www.google.com/search?q=%235-format-data--spesifikasi-tensor)
* [6. Integrasi Kode & Pipeline PyTorch](https://www.google.com/search?q=%236-integrasi-kode--pipeline-pytorch)
* [7. Referensi Sitasi](https://www.google.com/search?q=%237-referensi-sitasi)

---

## 1. Ikhtisar Dataset

MM-Fi adalah dataset *wireless human sensing* multi-modalitas 4D non-intrusif berskala besar. Varian `filtered_mmwave` di dalam direktori ini merepresentasikan titik pantulan radar mmWave (60 GHz / 77 GHz) yang telah dibersihkan dari *static clutter*, ditapis berdasarkan threshold SNR, dan disatukan via agregasi temporal.

* **Total Environment:** 4 variasi ruangan fisik (`E01`–`E04`)
* **Total Subjek:** 40 partisipan unik (`S01`–`S40`)
* **Total Kategori Aksi:** 27 aksi terannotasi (`A01`–`A27`)
* **Tujuan Penggunaan:** Ekstraksi *Physical State Representation* ($Z_t$) dan estimasi *3D Human Skeleton Pose* (17 keypoints).

---

## 2. Hierarki dan Struktur Direktori

Data tersusun secara sistematis mengikuti konvensi penamaan MM-Fi:

```text
filtered_mmwave/
├── E01/                                # Environment 01 (Laboratorium / Kantor)
│   ├── S01/                            # Subject 01
│   │   ├── A01/                        # Action 01 (Stretching and relaxing)
│   │   │   ├── mmwave/                 # Sub-folder titik radar terfilter
│   │   │   │   ├── frame_0001.npy      # Matrix Point Cloud (N x 5)
│   │   │   │   ├── frame_0002.npy
│   │   │   │   └── ...
│   │   │   └── ground_truth/           # Label 3D Skeleton (17 Keypoints)
│   │   │       ├── frame_0001.json
│   │   │       └── ...
│   │   ├── A02/
│   │   └── ... (hingga A27)
│   └── S02 ... S10
├── E02/                                # Environment 02 (Living Room Setup)
├── E03/                                # Environment 03 (High-Obstacle Room)
└── E04/                                # Environment 04 (Open Space)

```

---

## 3. Rincian Lingkungan Eksperimen (E01 - E04)

| Kode | Lingkungan | Karakteristik Spasial & Pemantulan |
| --- | --- | --- |
| **E01** | *Standard Lab / Office* | Dinding beton, meja laboratorium, komputer, dan papan tulis. Pantulan multipath level sedang. |
| **E02** | *Living Room Setup* | Interior rumah dengan bahan penyerap energi (sofa, karpet) serta batasan dinding dekat. |
| **E03** | *High-Obstacle Room* | Ruangan padat rintangan fisik di mana subjek sering mengalami oklusi spasial. |
| **E04** | *Open / Low-Clutter Space* | Area terbuka luas dengan interferensi pantulan minimal untuk *baseline evaluation*. |

---

## 4. Rincian Aksi dan Aktivitas Subjek (A01 - A27)

Dataset MM-Fi membagi 27 aksi ke dalam dua kelompok fungsional:

| Kode | Nama Aksi | Kategori |
| --- | --- | --- |
| **A01** | Stretching and relaxing | Rehabilitation |
| **A02** | Chest expansion (horizontal) | Daily |
| **A03** | Chest expansion (vertical) | Daily |
| **A04** | Twist (left) | Daily |
| **A05** | Twist (right) | Daily |
| **A06** | Mark time | Rehabilitation |
| **A07** | Limb extension (left) | Rehabilitation |
| **A08** | Limb extension (right) | Rehabilitation |
| **A09** | Lunge (toward left-front) | Rehabilitation |
| **A10** | Lunge (toward right-front) | Rehabilitation |
| **A11** | Limb extension (both) | Rehabilitation |
| **A12** | Squat | Rehabilitation |
| **A13** | Raising hand (left) | Daily |
| **A14** | Raising hand (right) | Daily |
| **A15** | Lunge (toward left side) | Rehabilitation |
| **A16** | Lunge (toward right side) | Rehabilitation |
| **A17** | Waving hand (left) | Daily |
| **A18** | Waving hand (right) | Daily |
| **A19** | Picking up things | Daily |
| **A20** | Throwing (toward left side) | Daily |
| **A21** | Throwing (toward right side) | Daily |
| **A22** | Kicking (toward left side) | Daily |
| **A23** | Kicking (toward right side) | Daily |
| **A24** | Body extension (left) | Rehabilitation |
| **A25** | Body extension (right) | Rehabilitation |
| **A26** | Jumping up | Rehabilitation |
| **A27** | Bowing | Daily |

---

## 5. Format Data & Spesifikasi Tensor

Setiap file `.npy` / `.bin` di dalam folder `mmwave/` merepresentasikan satu *frame* temporal point cloud teragregasi.

* **Tipe Data:** Array NumPy 2D floating point (`float32`).
* **Shape Tensor:** $(N \times 5)$, di mana $N$ adalah jumlah titik pantulan ($100 \le N \le 300$).

### Matriks Kolom:

$$\mathbf{P}_i = \begin{bmatrix} x_i & y_i & z_i & v_{d,i} & \text{SNR}_i \end{bmatrix}$$

1. **Kolom 0 ($X$):** Posisi lateral spasial dalam meter ($m$).
2. **Kolom 1 ($Y$):** Jarak kedalaman (*range*) dalam meter ($m$) dari sensor.
3. **Kolom 2 ($Z$):** Ketinggian vertikal dalam meter ($m$) relatif terhadap sensor.
4. **Kolom 3 ($v_d$):** Kecepatan Doppler radial dalam $m/s$.
5. **Kolom 4 ($\text{SNR}$):** *Signal-to-Noise Ratio* dalam Decibel ($\text{dB}$).

---

## 6. Integrasi Kode & Pipeline PyTorch

Berikut adalah kumpulan contoh kode PyTorch untuk memuat dataset, mengkonfigurasi file `.yaml`, serta mengintegrasikan output **Point-MAE** ke **Dynamics Model** dan **MLP Alignment Module LLM**.

### A. Konfigurasi `config.yaml` (MM-Fi Protocol Setup)

```yaml
modality: mmwave
data_unit: frame # 'frame' untuk Point-MAE / 'sequence' untuk Dynamics Model
protocol: protocol 3 # protocol 1: daily, protocol 2: rehab, protocol 3: all
split:
  split_to_use: manual_split
  manual_split:
    train_subjects: [S01, S02, S03, S04, S05, S06, S07, S08]
    val_subjects: [S09, S10]
train_loader:
  batch_size: 32
  shuffle: true
  num_workers: 4
validation_loader:
  batch_size: 32
  shuffle: false
  num_workers: 4
init_rand_seed: 42

```

---

### B. PyTorch Custom Dataset Loader untuk MM-Fi Filtered

```python
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MMFiFilteredDataset(Dataset):
    def __init__(self, root_dir, env_list=['E01'], num_points=128):
        self.root_dir = root_dir
        self.num_points = num_points
        self.samples = []
        
        # Traversing file system
        for env in env_list:
            env_path = os.path.join(root_dir, env)
            if not os.path.isdir(env_path): continue
            for sub in os.listdir(env_path):
                sub_path = os.path.join(env_path, sub)
                for act in os.listdir(sub_path):
                    act_path = os.path.join(sub_path, act)
                    mmwave_dir = os.path.join(act_path, 'mmwave')
                    gt_dir = os.path.join(act_path, 'ground_truth')
                    
                    if os.path.exists(mmwave_dir) and os.path.exists(gt_dir):
                        for f in os.listdir(mmwave_dir):
                            if f.endswith('.npy'):
                                frame_id = f.split('.')[0]
                                self.samples.append({
                                    'mmwave': os.path.join(mmwave_dir, f),
                                    'gt': os.path.join(gt_dir, f"{frame_id}.json")
                                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 1. Load Filtered mmWave Point Cloud (N x 5)
        pc_data = np.load(self.samples[idx]['mmwave']).astype(np.float32)
        xyz = pc_data[:, :3] # Take X, Y, Z coordinates
        
        # Sampling / Padding to fixed num_points
        if len(xyz) >= self.num_points:
            indices = np.random.choice(len(xyz), self.num_points, replace=False)
            xyz = xyz[indices]
        else:
            indices = np.random.choice(len(xyz), self.num_points, replace=True)
            xyz = xyz[indices]
            
        # 2. Load Ground Truth 3D Skeleton (17 keypoints x 3)
        with open(self.samples[idx]['gt'], 'r') as f:
            gt_data = json.load(f)
        skeleton_3d = np.array(gt_data['keypoints_3d'], dtype=np.float32)
        
        return torch.from_numpy(xyz), torch.from_numpy(skeleton_3d)

# Example Usage
if __name__ == '__main__':
    dataset_path = r"C:\Users\Rio Aslab\documents\pemrograman\tugas_akhir\datasets\filtered_mmwave"
    dataset = MMFiFilteredDataset(dataset_path, env_list=['E01'])
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    for mmwave_pts, target_skeleton in dataloader:
        print(f"mmWave Batch Tensor Shape: {mmwave_pts.shape}") # [16, 128, 3]
        print(f"3D Skeleton Target Shape: {target_skeleton.shape}") # [16, 17, 3]
        break

```

---

### C. Fase 2 & 3: Integrasi Point-MAE Encoder ke Dynamics Model

```python
import torch
import torch.nn as nn

class PointMAE_Dynamics_Bridge(nn.Module):
    def __init__(self, pretrained_point_mae_encoder, latent_dim=384, pred_horizon=5):
        super().__init__()
        # 1. Point-MAE Encoder (Frozen after Phase 2)
        self.encoder = pretrained_point_mae_encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # 2. Dynamics Model (Temporal Transformer - Trainable in Phase 3)
        decoder_layer = nn.TransformerDecoderLayer(d_model=latent_dim, nhead=8, batch_first=True)
        self.dynamics_transformer = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.pred_horizon = pred_horizon
        self.query_embed = nn.Parameter(torch.randn(1, pred_horizon, latent_dim))

    def forward(self, x_seq):
        # x_seq shape: [Batch, Seq_Len, Num_Points, 3]
        B, S, N, C = x_seq.shape
        x_flat = x_seq.view(B * S, N, C)
        
        # Pass through frozen Point-MAE Encoder
        with torch.no_grad():
            z_tokens = self.encoder(x_flat) # Extract spatial features
            z_t = z_tokens.mean(dim=1).view(B, S, -1) # Physical State Sequence [B, S, 384]
            
        # Predict Future Physical States Z_{t+1:t+h}
        query = self.query_embed.repeat(B, 1, 1)
        z_future = self.dynamics_transformer(tgt=query, memory=z_t) # [B, pred_horizon, 384]
        
        return z_t, z_future

```

---

### D. Fase 4: MLP Alignment Module (Proyeksi Vektor Fisik ke LLM)

```python
class PhysicalToLLMAlignmentMLP(nn.Module):
    def __init__(self, physical_dim=384, llm_embed_dim=4096):
        super().__init__()
        # Cross-Modal Projector: Physical Vector -> Pseudo Tokens
        self.projector = nn.Sequential(
            nn.Linear(physical_dim, physical_dim * 2),
            nn.GELU(),
            nn.Linear(physical_dim * 2, llm_embed_dim)
        )

    def forward(self, z_current, z_future):
        # z_current: [B, 384], z_future: [B, Horizon, 384]
        z_concat = torch.cat([z_current.unsqueeze(1), z_future], dim=1) # [B, 1+Horizon, 384]
        
        # Project to LLM Embedding Space (Pseudo Tokens)
        pseudo_tokens = self.projector(z_concat) # [B, 1+Horizon, 4096]
        return pseudo_tokens

```

---

## 7. Referensi Sitasi

Sebutkan publikasi resmi berikut saat melaporkan hasil eksperimen:

```bibtex
@inproceedings{yang2023mm,
    title={MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for Versatile Wireless Sensing},
    author={Yang, Jianfei and Huang, He and Zhou, Yunjiao and Chen, Xinyan and Xu, Yuecong and Yuan, Shenghai and Zou, Han and Lu, Chris Xiaoxuan and Xie, Lihua},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
    year={2023},
    url={https://openreview.net/forum?id=1uAsASS1th}
}

@inproceedings{pang2022masked,
    title={Masked autoencoders for point cloud self-supervised learning},
    author={Pang, Yatian and Wang, Wenxiao and Tay, Francis EH and Liu, Wei and Tian, Yonghong and Yuan, Li},
    booktitle={Computer Vision--ECCV 2022},
    pages={604--621},
    year={2022}
}

```