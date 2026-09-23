18. Evaluasi

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
