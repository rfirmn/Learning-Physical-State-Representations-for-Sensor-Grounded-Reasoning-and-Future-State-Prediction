21. Baseline

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
