DRAFT DOKUMENTASI EKSPERIMEN
Judul: Pemodelan Representasi Fisik Spasial-Temporal dan Penyelarasan LLM Berbasis Sinyal Radar mmWave

Dokumen ini merangkum secara komprehensif arsitektur, peran masing-masing modul, dan strategi pelatihan (training pipeline) untuk sistem kecerdasan buatan terpadu yang memetakan sinyal fisik mentah ke dalam penalaran bahasa alami. Sistem ini secara tegas memisahkan kemampuan persepsi fisik dari kemampuan kognitif bahasa guna memastikan efisiensi komputasi dan akurasi penalaran.

1. Ikhtisar Arsitektur Sistem (The Pipeline)
Arsitektur sistem dibangun berdasarkan prinsip modularitas (decoupling), yang terbagi menjadi dua ranah utama: Ranah Fisik (Pemrosesan Sensor & Dinamika) dan Ranah Kognitif (Penyelarasan & Penalaran Bahasa).

Alur data saat sistem beroperasi penuh (inference) adalah sebagai berikut:
Gelombang mmWave Mentah $\rightarrow$ Sensor Encoder $\rightarrow$ Representasi Fisik ($Z_t$) $\rightarrow$ Dynamics Model $\rightarrow$ Prediksi Masa Depan ($Z_{t+1:t+h}$) $\rightarrow$ MLP Proyektor $\rightarrow$ LLM (Frozen) $\rightarrow$ Penalaran Teks Alami

2. Ranah Fisik: Modul Machine Learning Sensor
Kelompok ini bertanggung jawab untuk "melihat" dan "meramalkan" kejadian fisik secara matematis. Walaupun nantinya bekerja sebagai satu kesatuan pipeline, secara arsitektur ia terdiri dari dua modul fungsional yang dilatih dengan fokus berbeda:

A. Sensor Encoder (Modul Persepsi Spasial)
Modul ini bertugas mengubah data titik koordinat yang acak (sparse point clouds: $x, y, z$, Doppler, SNR) menjadi Physical State Representation ($Z_t$)—sebuah vektor laten berdimensi tetap yang kaya akan informasi posisi, postur, dan relasi spasial pada waktu saat ini ($t$).

Karakteristik: Fokus pada ekstraksi fitur spasial.


Arsitektur Ideal: PointNet, Point Transformer, atau 1D Convolutional Neural Networks.


Target Pembelajaran: Memastikan representasi laten ($Z_t$) mampu merekonstruksi status fisik nyata (State Estimation), seperti koordinat sendi manusia (3D skeleton).


B. Dynamics Model (Modul Peramalan Temporal / Future-State Predictor)
Modul ini diletakkan tepat setelah Sensor Encoder. Tugas utamanya adalah menerima urutan historis dari status fisik ($Z_{t-k} \dots Z_t$) untuk mempelajari hukum pergerakan fisika (momentum, arah, kecepatan), lalu memproyeksikan status tersebut ke masa depan ($Z_{t+1 \dots t+h}$).

Karakteristik: Fokus pada peramalan deret waktu (temporal forecasting).


Arsitektur Ideal: Recurrent Neural Networks (GRU/LSTM), Neural ODE, atau Temporal Transformer berskala kecil.


Target Pembelajaran: Meminimalkan selisih (MSE) antara prediksi lintasan model dengan ground truth pergerakan subjek di masa depan.


3. Ranah Kognitif: Modul Penyelarasan dan Bahasa
Ranah ini sama sekali tidak memproses sinyal radar mentah, melainkan hanya bekerja menggunakan angka-angka matematis (vektor) yang dihasilkan oleh Ranah Fisik, lalu menerjemahkannya ke dalam logika bahasa.

A. MLP Alignment Module (Proyektor Lintas-Modalitas)
Ini adalah modul jembatan (cross-modal adapter). Data representasi fisik ($Z_t$) dan prediksi masa depan ($Z_{t+1:t+h}$) berada pada dimensi numerik yang berbeda dengan otak LLM.

Karakteristik: Lapisan Multi-Layer Perceptron dangkal (kombinasi Linear Layer dan aktivasi GELU).


Fungsi: Memetakan vektor fisik ke dalam ruang representasi (embedding space) milik LLM, mengubahnya menjadi "token fisik semu" (pseudo-tokens) yang bisa digabungkan dengan token teks biasa.


B. Large Language Model (Reasoning Engine)
Ini adalah inti kognitif dari sistem. LLM menerima prompt teks (misal: "Apakah orang ini akan menabrak rintangan?") beserta "token fisik" dari MLP.

Karakteristik: Model fondasi berskala masif (seperti seri Llama atau Qwen).


Fungsi Utama: Melakukan penalaran (spatial, temporal, counterfactual reasoning) berdasarkan data fisik yang telah diterjemahkan, lalu menghasilkan jawaban naratif.


4. Rencana Eksekusi Pelatihan (Training Strategy & Curriculum)

Untuk menghasilkan arsitektur di atas tanpa membebani komputasi, proses training dibagi ke dalam 4 tahapan kronologis. Pendekatan ini memanfaatkan *transfer learning* dari *foundation model* 3D untuk melompati insialisasi dari nol, sehingga proses pembelajaran tetap efisien dan berfokus pada kebaharuan metode penyelarasan.

Tahap 1: Inisialisasi Persepsi via Transfer Learning (Inisialisasi Sensor Encoder - Fase 1)

Tindakan: Memuat *pre-trained weights* dari Point-MAE (diadaptasi dari *pre-training self-supervised* pada dataset ShapeNet) sebagai *backbone* Sensor Encoder. Pada tahap ini, komponen Decoder bawaan Point-MAE dibuang, sehingga hanya modul Transformer Encoder yang dipertahankan.

Tujuan: Mengintegrasikan bobot persepsi spasial 3D yang sudah matang dari *foundation model* tanpa perlu melatih dari nol pada dataset sinyal sederhana.

Status Model: Menghasilkan "Model A" (Pre-trained Point-MAE Encoder Backbone).

Tahap 2: Pematangan Persepsi & Adaptasi Domain Radar (Fine-Tuning Sensor Encoder - Fase 2)

Tindakan: Memuat bobot "Model A", lalu melakukan *fine-tuning* langsung menggunakan dataset utama yang kompleks (misal: MM-Fi) yang memiliki label 3D skeleton. Proses ini menyesuaikan *patch embedding* dan lapisan Transformer agar toleran terhadap karakteristik *sparse point cloud* serta *noise* spesifik mmWave.

Tujuan: Mengadaptasi pengetahuan geometris 3D ke domain sinyal radar dan mempertajam akurasi Physical State Representation ($Z_t$) secara anatomis (3D skeleton estimation).

Status Model: Menghasilkan "Model Av2" (Encoder Utama). Setelah tahap ini selesai, seluruh bobot Encoder dibekukan (frozen).
*Catatan Hasil Eksperimen Lapangan:* Dokumentasi evaluasi kuantitatif lengkap, metrik MPJPE per sendi, dan status checkpoint dapat dilihat di [docs/laporan_eksperimen_1.md](file:///c:/Users/Rio%20Aslab/Documents/pemrograman/Tugas_Akhir/docs/laporan_eksperimen_1.md).

Tahap 3: Pemahaman Waktu (Training Dynamics Model - Fase 3)

Tindakan: Memasukkan sekuens spasial deret waktu ($Z_{t-k} \dots Z_t$) yang diekstrak oleh Encoder yang sudah beku ke dalam Dynamics Model (misal: Temporal Transformer atau ST-GCN) untuk dilatih dari nol.

Tujuan: Memampukan sistem memprediksi vektor status fisik masa depan ($Z_{t+1 \dots t+h}$) berdasarkan sejarah pergerakan subjek.

Status Model: Menghasilkan Dynamics Model yang matang. Setelah tahap ini selesai, bobot Dynamics Model juga dibekukan (frozen).

Tahap 4: Penyelarasan Kognitif Lintas-Modalitas (Training MLP Alignment - Fase 4)

Tindakan: Sensor Encoder, Dynamics Model, dan LLM dibekukan sepenuhnya (*frozen*). Hanya lapisan proyektor MLP (Linear + GELU) yang dilatih menggunakan dataset pasangan representasi fisik dan instruksi teks (QA Pairs).

Tujuan: Memetakan gabungan vektor fisik saat ini ($Z_t$) dan prediksi masa depan ($Z_{t+1:t+h}$) ke dalam ruang *embedding* LLM menjadi *pseudo-tokens*, sehingga LLM dapat melakukan penalaran berbasis fisik (*physically-grounded reasoning*).

Status Model: Seluruh sistem menyatu secara utuh. Modul persepsi fisik dan otak LLM terhubung secara presisi melalui adaptor lintas-modalitas.

Penguraian beban kognitif ini membuktikan bahwa kita tidak perlu membangun LLM baru khusus radar maupun melatih *encoder* 3D dari awal. Dengan memanfaatkan Point-MAE sebagai fondasi persepsi dasar, eksperimen dapat langsung berfokus pada evaluasi empiris metodologi penyelarasan sinyal fisik ke penalaran bahasa alami.


