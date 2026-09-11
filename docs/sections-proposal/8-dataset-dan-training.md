8. Dataset dan training

Berikut adalah gambaran bagaimana ketiga fase tersebut melebur menjadi 1 model utuh:
1. Peran Muler / Dataset Prototype (Fase 1)
Fase: Menggunakan dataset sederhana (seperti MVRHAR).
Hasil: Menghasilkan modul Sensor Encoder yang dasar. Modul ini menjadi fondasi pertama untuk memastikan sinyal mentah radar berhasil diterjemahkan menjadi angka koordinat/vektor yang bersih.
2. Peran Dataset Utama / MM-Fi (Fase 2)
Fase: Menggunakan dataset yang lebih kaya (MM-Fi dengan 3D skeleton dan dinamika spasial).
Hasil: Model Encoder dari Fase 1 diperkaya kemampuannya, lalu disambungkan dengan Dynamics Model untuk memprediksi masa depan ($S_t \rightarrow S_{t+k}$). Pada titik ini, Anda sudah memiliki satu modul persepsi dan prediksi fisik yang utuh.
3. Peran Dataset Lanjutan / M4Human (Fase 3 - Opsional/Extension)
Fase: Menggunakan dataset paling kompleks untuk stress test.
Hasil: Menyempurnakan ketahanan (robustness) dari model fisik dan dinamika yang sudah dibangun di Fase 2 agar tidak mudah patah saat menghadapi gerakan manusia yang liar dan cepat.
Wujud Akhir Saat Digunakan (Inference)
Ketika model ini sudah matang melalui ketiga fase tersebut, kode program Anda hanya akan mengeksekusi 1 pipeline tunggal yang menerima input dan mengeluarkan output akhir:
$$\text{Input: mmWave Radar Point Cloud} \quad \longrightarrow \quad \mathbf{[ 1 \text{ Model Utuh}]*} \quad \longrightarrow \quad \text{Output: Jawaban Reasoning LLM}$$
*Catatan di dalam kotak [1 Model Utuh]: Berisi rangkaian fungsi yang sudah menyatu—mulai dari Encoder yang membaca radar, Dynamics Model yang menghitung arah/kecepatan, hingga LLM yang membaca hasil vektor tersebut untuk menjawab pertanyaan relasional atau kontrafaktual.
