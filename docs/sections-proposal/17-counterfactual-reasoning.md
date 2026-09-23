17. Counterfactual reasoning & sensor controls

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
