26. Scope penelitian

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
