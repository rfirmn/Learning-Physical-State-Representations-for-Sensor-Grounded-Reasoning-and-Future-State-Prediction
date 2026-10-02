# Arah Riset Terbaru

**Judul kerja:** *Sensor-Grounded Kinematic Dynamics Reasoning with a Frozen Language Model*

**Status:** Dokumen arah riset dan bahan diskusi  
**Tanggal:** 2 Oktober 2026  
**Repositori:** *Learning Physical State Representations for Sensor-Grounded Reasoning and Future-State Prediction*

> Dokumen ini menggantikan framing lama yang menjadikan “LLM membaca masa depan” atau forecasting sebagai tujuan utama. Arsitektur teknis masih terbuka untuk diubah. Yang dikunci terlebih dahulu adalah pertanyaan ilmiah, batas klaim, dataset, dan cara validasinya.

---

## 1. Ringkasan inti

Penelitian ini mengkaji apakah representasi dinamika gerak manusia yang dipelajari dari sensor radar mmWave dapat digunakan oleh frozen Large Language Model (LLM) untuk melakukan penalaran temporal dan kinematik yang ter-grounding. Fokus penelitian bukan sekadar mengenali label aktivitas seperti squat, punch, atau walking, dan bukan pula membuat LLM “membaca masa depan”. Fokusnya adalah memodelkan bagaimana keadaan tubuh berubah dari waktu ke waktu—misalnya perpindahan pose, arah, velocity, acceleration, fase gerakan, dan bila tersedia, kontak atau gaya—kemudian menguji apakah LLM dapat menggunakan representasi tersebut untuk menjawab pertanyaan tentang perubahan keadaan fisik secara konsisten.

Pipeline tingkat tinggi tetap dipertahankan sebagai pembagian fungsi:

```text
Sensor observation → Physical state representation → Dynamics representation
                  → Physical validation + Language reasoning
```

Detail internal encoder, dynamics model, decoder, tokenizer, dan projector tidak dianggap final. Komponen tersebut dipilih setelah definisi state, target pembelajaran, dan evaluasi ditetapkan.

---

## 2. Masalah penelitian

Model sensor–bahasa yang ada telah menunjukkan bahwa sinyal sensor dapat diselaraskan dengan bahasa untuk activity recognition, trend description, dan QA. Namun, sebagian besar pendekatan tersebut masih mengevaluasi apakah model dapat mengenali aktivitas atau menghasilkan deskripsi yang sesuai. Hal itu belum menjawab pertanyaan yang lebih kuat:

> Apakah LLM dapat melakukan penalaran atas perubahan keadaan fisik yang terstruktur dari waktu ke waktu, bukan hanya mencocokkan pola sensor dengan label atau caption?

Masalah ini penting karena gerakan manusia bukan sekumpulan pose statis. Dua frame dapat memiliki pose yang mirip tetapi memiliki makna dinamika yang berbeda: tubuh dapat sedang mempercepat, melambat, berbalik, kehilangan keseimbangan, atau berada pada fase berbeda dari suatu gerakan. Karena itu, informasi pose tunggal tidak cukup untuk mewakili dinamika.

Penelitian ini berusaha menjembatani tiga ranah yang biasanya dipisahkan:

1. persepsi keadaan fisik dari sensor non-visual;
2. pemodelan perubahan keadaan atau dinamika gerak;
3. penalaran bahasa atas dinamika tersebut.

---

## 3. Pertanyaan penelitian

### Pertanyaan utama

> **Sejauh mana frozen LLM dapat melakukan penalaran temporal dan kinematik yang ter-grounding pada representasi dinamika gerak manusia yang dipelajari dari sensor radar mmWave?**

### Pertanyaan turunan

1. Apakah sekuens radar dapat diubah menjadi representasi keadaan yang memuat lebih dari pose sesaat?
2. Apakah model dinamika dapat menangkap perubahan pose, velocity, acceleration, arah gerak, dan fase gerakan secara konsisten?
3. Apakah representasi tersebut lebih berguna untuk penalaran dibandingkan state statis, label aktivitas, atau persistence baseline?
4. Apakah LLM benar-benar menggunakan informasi temporal, atau hanya mengandalkan prior bahasa dan label aktivitas?
5. Apakah kemampuan tersebut tetap berlaku pada subjek, lingkungan, kecepatan, dan jenis gerakan yang tidak dilihat saat training?

---

## 4. Tujuan penelitian

Penelitian ini bertujuan untuk:

1. membangun representasi dinamika gerak manusia dari data sensor mmWave;
2. memisahkan informasi keadaan, perubahan gerak, dan penalaran bahasa;
3. menguji penggunaan frozen LLM sebagai reasoning engine atas representasi sensor-derived dynamics;
4. merancang evaluasi yang membedakan physical grounding dari sekadar klasifikasi atau caption generation;
5. mengukur generalisasi terhadap subjek, lingkungan, dan pola gerakan yang tidak dikenal.

Forecasting dapat digunakan sebagai auxiliary objective atau diagnostic metric, tetapi bukan tujuan akhir penelitian.

---

## 5. Batas istilah: kinematic dynamics versus physical dynamics

Istilah “physical dynamics” harus digunakan dengan hati-hati.

### 5.1 Kinematic dynamics

Kinematic dynamics membahas perubahan gerakan yang dapat diamati dari trajectory tubuh, antara lain:

- posisi joint;
- displacement;
- velocity;
- acceleration;
- arah gerakan;
- fase atau siklus gerakan;
- global trajectory;
- perubahan relatif antarbagian tubuh.

Radar dan motion capture umumnya cukup untuk mendukung level ini.

### 5.2 Physical dynamics penuh

Physical dynamics dalam arti biomekanik yang lebih kuat memerlukan informasi seperti:

- ground-reaction force;
- contact force dan contact location;
- joint torque;
- massa dan inertia;
- center of mass;
- energi, momentum, atau kerja mekanis;
- interaksi dengan objek atau lingkungan.

Jika label force, torque, dan contact tidak tersedia, klaim yang aman adalah **human motion kinematics** atau **kinematic dynamics**, bukan pembelajaran hukum Newton secara penuh.

---

## 6. Posisi terhadap penelitian terkait

Cross-check literatur menunjukkan bahwa penelitian ini tidak boleh lagi mengklaim sebagai “radar-to-LLM pertama”.

### RadarLLM

**RadarLLM** menggunakan sekuens point cloud mmWave, motion-guided radar tokenizer, masked trajectory modeling, physics-aware radar-text synthesis, dan radar-aware language model untuk menghasilkan deskripsi gerakan. Ini adalah baseline terdekat dan wajib dibahas.

Perbedaan arah penelitian ini:

- RadarLLM berfokus pada radar-to-text motion understanding dan kualitas caption;
- penelitian ini berfokus pada representasi dinamika kinematik yang terpisah dan evaluasi temporal/physical reasoning;
- penelitian ini mengutamakan pengujian controlled grounding, bukan hanya kualitas bahasa;
- frozen LLM dapat dipertahankan sebagai kondisi utama untuk menguji kemampuan alignment tanpa melatih ulang pengetahuan bahasa.

Sumber: [RadarLLM—AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/download/37500/41462), [RadarLLM—arXiv](https://arxiv.org/abs/2504.09862).

### SensorLLM

SensorLLM menyelaraskan sensor wearable multivariat dengan deskripsi trend dan menggunakannya untuk human activity recognition. Kontribusinya penting sebagai pembanding sensor-language alignment, tetapi fokusnya masih pada sensor 1D wearable, trend description, dan HAR.

Sumber: [SensorLLM—ACL Anthology](https://aclanthology.org/2025.emnlp-main.19/).

### LLaSA

LLaSA menggunakan data IMU/smartphone dan LLM untuk human activity analysis serta sensor-based QA. Penelitian ini relevan untuk sensor-aware language reasoning, tetapi tidak secara eksplisit memisahkan model dinamika kinematik dari language model dan bukan berbasis radar spatial point cloud.

Sumber: [LLaSA—arXiv](https://arxiv.org/abs/2406.14498).

### World models

Literatur world model memberikan konsep penting berupa latent state, transition dynamics, predictive representation, dan reasoning/planning. Namun, contoh Wireless Dreamer berfokus pada optimasi jaringan nirkabel dan UAV, bukan dinamika tubuh manusia dari radar.

Sumber: [World Models for Cognitive Agents](https://arxiv.org/abs/2506.00417).

### Motion representation dan kinematic language

MotionBERT dan KinMo menunjukkan bahwa representasi temporal, struktur joint, dan bahasa kinematik merupakan area riset yang mapan. Keduanya menjadi rujukan untuk motion representation dan motion-language modeling, bukan pengganti langsung untuk radar-based dynamics grounding.

Sumber: [MotionBERT—ICCV](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_MotionBERT_A_Unified_Perspective_on_Learning_Human_Motion_Representations_ICCV_2023_paper.pdf), [KinMo—ICCV](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Zhang_KinMo_Kinematic-aware_Human_Motion_Understanding_and_Generation_ICCV2025_paper.pdf).

### Posisi kebaruan yang aman

Kebaruan penelitian ini sebaiknya dirumuskan sebagai berikut:

> Penelitian ini menyelidiki penggunaan representasi dinamika kinematik manusia yang dipelajari dari sekuens radar mmWave nyata sebagai dasar bagi frozen LLM untuk melakukan penalaran temporal yang ter-grounding, dengan evaluasi berbasis fakta kinematik dan kontrol perturbasi temporal.

Klaim “first radar LLM”, “LLM memahami hukum fisika”, dan “LLM membaca masa depan” tidak digunakan.

---

## 7. Strategi dataset

Dataset dipilih berdasarkan label dinamika yang tersedia, bukan berdasarkan jumlah kelas aktivitas.

### 7.1 Dataset utama: M4Human

M4Human dipilih sebagai kandidat dataset utama karena menyediakan radar mmWave, motion capture, global trajectory, serta 50 aktivitas dengan gerakan in-place, sit-in-place, walking, dan olahraga. M4Human memiliki sekitar 661 ribu frame dan 20 subjek.

Yang digunakan:

- radar point cloud;
- skeleton atau mesh-derived joints;
- global trajectory;
- metadata waktu dan subjek.

Yang tidak digunakan untuk menjaga beban komputasi:

- RGB;
- depth;
- raw radar tensor, kecuali ada eksperimen khusus;
- seluruh dataset penuh sekaligus.

Repository resmi mencantumkan processed radar modality sekitar 50 GB dan konfigurasi benchmark multi-GPU. Karena itu, penelitian ini menggunakan subset subjek dan membuat cache ringkas point cloud plus motion labels, bukan menjalankan benchmark resminya secara penuh.

Sumber: [M4Human project page](https://fanjunqiao.github.io/M4Human-site/), [M4Human repository](https://github.com/FanJunqiao/M4Human).

### 7.2 Dataset supervisi fisik: AddBiomechanics

AddBiomechanics digunakan jika penelitian ingin melampaui kinematic dynamics dan menguji dinamika biomekanik. Dataset ini menyediakan posisi, velocity, acceleration, ground-contact force, joint torque, massa, inertia, dan center-of-mass kinematics pada lebih dari 24 juta frame dan 273 partisipan.

AddBiomechanics tidak digunakan sebagai input radar langsung. Perannya adalah:

- supervisi atau pretraining dynamics representation;
- validasi bahwa representation memuat informasi velocity/acceleration/force;
- physical probe terhadap state yang dipelajari.

Tidak perlu memproses seluruh dataset. Subset terstratifikasi sekitar 2–3 juta frame cukup untuk eksperimen awal.

Sumber: [AddBiomechanics Dataset](https://www.addbiomechanics.org/download_data.html).

### 7.3 Validasi eksternal: PhysioNet Radar–Force Plate

Dataset PhysioNet yang berisi radar 24 GHz, motion capture, force plate, dan fase foot-lift/stability/foot-touchdown digunakan sebagai external validation. Dataset ini kecil, yaitu 32 partisipan dan 1.241 trial, tetapi hanya mencakup one-legged stand.

Dataset ini tidak digunakan sebagai sumber utama training seluruh jenis gerakan. Perannya adalah menguji apakah representasi sensor dapat berkorelasi dengan keadaan stabilitas dan ground interaction pada domain yang berbeda.

Sumber: [PhysioNet Radar–Force Plate Dataset](https://physionet.org/content/?topic=force+plate).

### 7.4 MM-Fi

MM-Fi tetap digunakan dalam peran terbatas:

- sanity check pipeline;
- baseline dari eksperimen yang sudah ada;
- validasi awal encoder;
- perbandingan dengan latent forecasting sebelumnya.

MM-Fi tidak dijadikan bukti utama physical dynamics karena label utamanya adalah 3D skeleton dan action segments, tanpa ground-reaction force, joint torque, atau contact dynamics. Selain itu, action set MM-Fi yang tersedia tidak mencakup walking sebagai aktivitas utama.

### 7.5 Keputusan dataset

Konfigurasi utama yang direkomendasikan:

```text
M4Human        → radar-based motion/kinematic dynamics
AddBiomechanics → physical/biomechanical supervision
PhysioNet      → external radar + force/contact validation
MM-Fi          → development baseline dan sanity check
```

Jika resource atau waktu tidak cukup:

```text
M4Human + PhysioNet
```

menjadi konfigurasi minimum. AddBiomechanics hanya wajib jika tesis ingin mempertahankan klaim force, torque, atau contact reasoning.

---

## 8. Bentuk penalaran yang ingin diuji

LLM tidak diuji hanya dengan pertanyaan “aktivitas apa yang dilakukan?”. Pertanyaan utama harus berkaitan dengan perubahan fisik.

### Kinematic reasoning

- Bagian tubuh mana yang bergerak paling cepat?
- Apakah joint sedang mempercepat atau memperlambat?
- Ke arah mana pusat massa atau torso berpindah?
- Apakah subjek berada pada fase awal, puncak, atau akhir gerakan?
- Apakah gerakan bersifat stabil, repetitif, atau transisional?

### Relational temporal reasoning

- Apakah tangan bergerak sebelum torso?
- Apakah lutut menekuk sebelum pusat massa turun?
- Apakah kecepatan kaki meningkat ketika torso mulai berpindah?
- Apakah dua trajectory memiliki urutan temporal yang sama?

### Physical/biomechanical reasoning

Pertanyaan ini hanya digunakan bila label yang diperlukan tersedia:

- Kaki mana yang menerima ground force lebih besar?
- Apakah keadaan menunjukkan kehilangan stabilitas?
- Apakah acceleration dan contact pattern konsisten dengan foot touchdown?
- Apakah perubahan gerak memerlukan respons gaya yang lebih besar?

Pertanyaan harus dibuat dari ground truth terukur atau aturan deterministik, bukan caption subjektif yang dibuat bebas oleh LLM.

---

## 9. Prinsip evaluasi

### 9.1 Validasi representasi dinamika

Evaluasi tidak berhenti pada action accuracy. Minimal mencakup:

- position/joint error;
- velocity error;
- acceleration error;
- trajectory error;
- arah perpindahan;
- phase consistency;
- bone-length consistency;
- contact/stability metrics bila tersedia.

### 9.2 Baseline wajib

- static state only;
- last-state persistence;
- constant-velocity baseline;
- temporal model tanpa Doppler;
- temporal model dengan urutan frame asli;
- frame yang diacak;
- frame yang dibalik;
- representation yang diganti dengan state dari subjek atau aksi lain.

### 9.3 Validasi LLM

LLM dianggap menggunakan dinamika sensor hanya jika:

1. jawaban berubah secara tepat ketika velocity atau acceleration berubah;
2. jawaban sensitif terhadap urutan waktu;
3. informasi statis tetap stabil ketika hanya perubahan temporal yang diubah;
4. jawaban memburuk pada token/sequence yang diacak;
5. hasil tetap berlaku pada subjek dan sesi yang tidak dilihat saat training.

Kemampuan menghasilkan kalimat yang lancar bukan bukti physical grounding.

### 9.4 Counterfactual dan perturbation

Perturbasi harus tetap berada dalam ruang fisik yang valid. Contoh yang aman:

- membalik arah velocity;
- memperbesar atau memperkecil displacement dalam batas realistis;
- menghapus satu fase gerakan;
- menukar dua urutan waktu;
- mengganti trajectory dengan trajectory dari aksi lain.

Jangan langsung menyimpulkan kausalitas hanya dari manipulasi arbitrary latent vector. Istilah yang lebih aman adalah **controlled perturbation test**.

---

## 10. Batas hardware RTX 3060 12 GB

Desain eksperimen harus menggunakan pre-extraction dan staged training.

### Rekomendasi operasional

- Radar encoder: mixed precision, batch sekitar 8–16.
- Dynamics model: batch sekitar 32–64 karena inputnya sudah berupa fitur.
- Language alignment: batch 1–2 dengan gradient accumulation.
- Tidak menggunakan RGB/depth dalam training utama.
- Tidak menggunakan raw M4Human 2 TB.
- Tidak menjalankan konfigurasi resmi M4Human yang membutuhkan empat GPU.
- Tidak melakukan hyperparameter sweep besar; satu baseline dan satu ablation utama cukup.

Alur komputasi:

```text
Raw radar → extract/cache compact features → train dynamics
                                          → build QA/physical probes
                                          → train language alignment
```

Dengan cara ini, GPU tidak perlu terus-menerus membaca dan memproses seluruh raw dataset selama eksperimen language alignment.

---

## 11. Hal yang tidak diklaim

Penelitian ini tidak mengklaim bahwa:

- frozen LLM memperoleh pemahaman fisika umum seperti manusia;
- LLM mempelajari hukum Newton secara lengkap;
- radar dapat mengukur semua gaya dan torsi tanpa supervisi biomekanik;
- forecasting akurat berarti LLM memahami dinamika;
- caption yang baik membuktikan physical reasoning;
- penelitian ini adalah radar-to-LLM pertama.

Klaim yang aman:

> Sistem mempelajari representasi dinamika gerak manusia dari data sensor dan menguji apakah frozen LLM dapat menggunakannya untuk melakukan penalaran temporal/kinematik yang ter-grounding.

---

## 12. Risiko riset dan mitigasi

| Risiko | Dampak | Mitigasi |
|---|---|---|
| Dataset radar tidak memiliki force/torque | Klaim physical dynamics terlalu kuat | Gunakan istilah kinematic dynamics atau tambahkan AddBiomechanics/PhysioNet |
| M4Human terlalu besar | Tidak selesai diproses di RTX 3060 | Gunakan processed radar-only, subset subjek, dan cache compact |
| LLM menjawab dari prior bahasa | Grounding palsu | Gunakan shuffle, reversal, counterfactual, dan output terstruktur |
| Latent dynamics tidak interpretable | Sulit membuktikan representasi | Tambahkan physical probes untuk pose, velocity, acceleration, dan phase |
| Domain gap radar–biomekanik | Transfer supervisi tidak stabil | Gunakan AddBiomechanics untuk pretraining/probe, bukan campur langsung sebagai raw input |
| Dataset terlalu sempit | Generalisasi rendah | Pisahkan training, validation, dan external validation berdasarkan subjek/sesi |
| Klaim novelty bertumpuk dengan RadarLLM | Kontribusi dianggap incremental | Fokus pada explicit dynamics representation, frozen LLM, dan controlled grounded evaluation |

---

## 13. Tahapan kerja yang disepakati

1. Tetapkan definisi final antara kinematic dynamics dan physical dynamics.
2. Audit akses dan format M4Human, AddBiomechanics, dan PhysioNet.
3. Buat subset dataset yang sesuai dengan RTX 3060.
4. Tentukan state target dan label dinamika yang benar-benar tersedia.
5. Bangun baseline kinematic dynamics tanpa LLM.
6. Validasi velocity, acceleration, trajectory, phase, dan stability.
7. Buat QA terstruktur dari ground truth.
8. Uji frozen LLM dengan physical tokens/representation.
9. Jalankan ablation temporal order, shuffle, persistence, dan static-only.
10. Bandingkan hasil dengan baseline internal dan RadarLLM secara konseptual.

Tidak ada training projector atau LLM alignment sebelum tahap 1–6 menghasilkan representation yang lulus physical/kinematic validation.

---

## 14. Kesimpulan proposal

Penelitian ini mengusulkan studi tentang bagaimana dinamika gerak manusia yang diperoleh dari sensor radar mmWave dapat direpresentasikan dan digunakan untuk mendukung penalaran temporal oleh frozen LLM. Fokusnya bukan lagi pada klasifikasi aktivitas atau prediksi masa depan sebagai tujuan akhir, melainkan pada kemampuan sistem untuk menangkap perubahan keadaan fisik yang terukur—terutama pose, trajectory, velocity, acceleration, fase gerak, dan bila tersedia gaya serta kontak—kemudian menguji apakah LLM dapat melakukan penalaran yang konsisten terhadap informasi tersebut. Penelitian ini diposisikan bukan sebagai radar-to-LLM pertama, melainkan sebagai kajian tentang explicit sensor-derived dynamics representation dan grounded temporal reasoning dengan evaluasi terkontrol. M4Human menjadi kandidat dataset radar utama, AddBiomechanics menyediakan supervisi biomekanik bila klaim physical dynamics dipertahankan, PhysioNet menjadi validasi eksternal radar–force, dan MM-Fi dipertahankan sebagai baseline pengembangan. Keberhasilan penelitian dinilai bukan dari kelancaran output bahasa, tetapi dari kesesuaian jawaban LLM terhadap ground truth dinamika, sensitivitas terhadap urutan waktu dan perturbasi, serta generalisasi pada subjek dan kondisi yang tidak terlihat saat training.

---

## 15. Referensi utama

1. [RadarLLM: Empowering Large Language Models to Understand Human Motion from Millimeter-Wave Point Cloud Sequence](https://ojs.aaai.org/index.php/AAAI/article/download/37500/41462)
2. [SensorLLM: Aligning Large Language Models with Motion Sensors for Human Activity Recognition](https://aclanthology.org/2025.emnlp-main.19/)
3. [LLaSA: A Multimodal LLM for Human Activity Analysis Through Wearable and Smartphone Sensors](https://arxiv.org/abs/2406.14498)
4. [World Models for Cognitive Agents: Transforming Edge Intelligence in Future Networks](https://arxiv.org/abs/2506.00417)
5. [MotionBERT: A Unified Perspective on Learning Human Motion Representations](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_MotionBERT_A_Unified_Perspective_on_Learning_Human_Motion_Representations_ICCV2023_paper.pdf)
6. [KinMo: Kinematic-aware Human Motion Understanding and Generation](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Zhang_KinMo_Kinematic-aware_Human_Motion_Understanding_and_Generation_ICCV2025_paper.pdf)
7. [M4Human project page](https://fanjunqiao.github.io/M4Human-site/)
8. [M4Human official repository](https://github.com/FanJunqiao/M4Human)
9. [AddBiomechanics Dataset](https://www.addbiomechanics.org/download_data.html)
10. [GroundLink Dataset](https://csr.bu.edu/groundlink/)
11. [PhysioNet Radar–Force Plate Dataset](https://physionet.org/content/?topic=force+plate)
