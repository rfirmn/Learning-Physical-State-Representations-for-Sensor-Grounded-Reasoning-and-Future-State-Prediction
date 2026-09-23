30. Resource requirement

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
