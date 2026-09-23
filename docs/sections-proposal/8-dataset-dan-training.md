8. Dataset dan kurikulum pelatihan

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
