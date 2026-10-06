# Peta Rujukan untuk Menulis Bab I

**Arah riset:** pelestarian informasi kinematik pada token ringkas dari radar point cloud M4Human untuk tanya jawab temporal dan relasional dengan model bahasa beku. Dokumen ini berisi bahan bacaan dan batas sitasi. Bab I dianggap belum ditulis; belum ada hasil eksperimen M4Human yang dapat dilaporkan sebagai temuan penelitian.

**Cara memakai dokumen:** “klaim yang dapat disitasi” berarti isi paper yang boleh ditulis ulang dengan kata-kata sendiri, disertai nama penulis dan tahun. Bagian “batas” menunjukkan kesimpulan yang tidak mengikuti bukti paper itu. Tautan mengarah ke prosiding penerbit atau naskah penulis; arXiv ditandai sebagai preprint. Folder “jurnal” lokal berisi artikel prosiding, preprint, dan survei. Status terbit tiap sumber perlu diperiksa saat menyusun daftar pustaka.

## 1. Peta argumen dan urutan baca

| Pertanyaan bagi penulis | Bacaan awal | Apa yang dapat dibangun darinya |
|---|---|---|
| Data apa yang diamati dan acuan gerak apa yang tersedia? | Fan dkk., *M4Human*; Liang dkk., *Wave2Body* | Radar point cloud tidak sama dengan pose. M4Human menyediakan anotasi gerak untuk membentuk target offline; Wave2Body menunjukkan bahwa pemulihan pose dari pantulan radar tetap masalah tersendiri. |
| Apakah radar sudah pernah dihubungkan ke bahasa? | Lai dkk., *RadarLLM*; Zhang dkk., *mmMind* | Ya. Maka penelitian baru perlu menjelaskan perbedaan tugas dan bukti, bukan mengulang klaim “radar ke bahasa pertama”. |
| Apakah tanya jawab radar sudah ada? | Zhang dkk., *mmMind*; Shin dkk., *mmWave-QA* | Ya, termasuk pertanyaan waktu dan bagian tubuh. Perbedaan yang hendak diuji terletak pada token setelah kompresi, bukan keberadaan QA itu sendiri. |
| Mengapa posisi, perubahan gerak, dan waktu perlu diperiksa? | Fang dkk., *HuMoCon*; Markhorst dkk., *FiGMo*; Endo dkk., *HumanMotionQA* | Ketiganya memberi preseden tugas atau hasil ablasi pada rincian gerak. Angka mereka berasal dari gerak/pose atau video, bukan bukti bahwa token radar M4Human kehilangan informasi. |
| Apakah supervisi kinematik pada token sudah pernah dicoba? | Huang dkk., *SeMoCo*; Cao dkk., *MotionCtrl*; Gu dkk., *MoTok* | Sudah ada token gerak yang menjaga detail tubuh dan dipakai untuk bahasa atau generasi. Nama loss, token, maupun bagian tubuh tidak cukup sebagai klaim kebaruan. |
| Bagaimana menafsirkan probe dan jawaban berbasis sensor? | Hewitt dan Liang; Goyal dkk. | Probe yang kuat dapat mempelajari tugas sendiri. Pasangan pertanyaan sama dengan acuan sensor berbeda membantu memeriksa prior bahasa, tetapi perlu protokol radar yang sah. |

Urutan baca yang hemat: M4Human → RadarLLM → mmMind → mmWave-QA → HuMoCon → FiGMo → SeMoCo. Setelah posisi penelitian jelas, baca Wave2Body untuk persoalan persepsi radar dan rujukan metodologis untuk rancangan evaluasi. Urutan ini mengikuti masalah riset, bukan tahun terbit.

## 2. Rujukan inti: klaim, letak bukti, dan batasnya

### R1. Fan dkk. (2026), M4Human

**Sitasi:** Fan, J., Zhou, Y., Yang, Y., Cui, X., Zhang, J., Xie, L., Yang, J., Lu, C. X., & Ding, F. (2026). *M4Human: A Large-Scale Multimodal mmWave Radar Benchmark for Human Mesh Reconstruction*. CVPR, 42836–42846. [Naskah prosiding](https://openaccess.thecvf.com/content/CVPR2026/html/Fan_M4Human_A_Large-Scale_Multimodal_mmWave_Radar_Benchmark_for_Human_Mesh_CVPR_2026_paper.html); [suplemen](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Fan_M4Human_A_Large-Scale_CVPR_2026_supplemental.pdf).

**Klaim yang dapat disitasi.** Penulis melaporkan sekitar 661 ribu frame, 999 sekuens, 20 subjek, dan 50 aktivitas. Dataset menyediakan radar point cloud terproses (RPC), radar tensor (RT), RGB, depth, serta anotasi MoCap yang memuat bentuk tubuh dan lintasan global. Suplemen menjelaskan bahwa titik RPC membawa koordinat x, y, z dan intensitas. Lihat abstrak dan §3 pada naskah utama, serta bagian format modalitas pada suplemen.

**Mengapa dipakai.** Rujukan ini menetapkan data apa yang mungkin diamati oleh model dan target apa yang dapat dibentuk untuk menguji posisi, lintasan pelvis, atau kecepatan hasil turunan lintasan. Kehadiran label gerak memberi alasan memilih pertanyaan yang jawabannya dapat diaudit, bukan berhenti pada kelas aktivitas.

**Batas.** Angka dataset adalah laporan paper, bukan jumlah window sah dalam paket processed milik peneliti. RPC M4Human tidak boleh disebut memiliki Doppler atau SNR sebelum skema paket diperiksa. MoCap, parameter tubuh, dan nama aktivitas tidak boleh muncul sebagai masukan saat sistem menjawab dari radar.

### R2. Lai dkk. (2026), RadarLLM

**Sitasi:** Lai, Z., Yang, J., Xia, S., Lin, L., Sun, L., Wang, R., Liu, J., Wu, Q., & Pei, L. (2026). *RadarLLM: Empowering Large Language Models to Understand Human Motion from Millimeter-wave Point Cloud Sequence*. AAAI, 40(7), 5791–5799. [DOI dan artikel prosiding](https://doi.org/10.1609/aaai.v40i7.37500); [versi panjang](https://arxiv.org/html/2504.09862v2).

**Klaim yang dapat disitasi.** RadarLLM membentuk token dari sekuens point cloud mmWave dengan Aggregate VQ-VAE, lalu menyelaraskannya dengan model bahasa untuk deskripsi gerak. Paper melaporkan hasil radar-ke-teks pada data sintetis dan nyata. Versi panjang, Appendix H, memperlihatkan bahwa VQ menaikkan ROUGE-L pada salah satu pengaturan radar-ke-teks dari 29,8 menjadi 31,2, sementara METEOR turun. Rincian arsitektur ada pada bagian metode; hasil dan ablasi ada pada bagian eksperimen serta Appendix H.

**Mengapa dipakai.** RadarLLM membuktikan kelayakan jalur radar menuju keluaran bahasa dan menjadi pembanding konseptual terdekat untuk tokenisasi radar. Penelitian baru perlu menguji besaran fisik pada token akhir dan jawaban beracuan, karena metrik teks tidak langsung mengukur keterbacaan posisi atau kecepatan.

**Batas.** Jangan tulis bahwa kompresi RadarLLM sudah terbukti merusak kinematika, bahwa semua metriknya naik karena VQ, atau bahwa penelitian ini baru pertama kali menghubungkan radar dan bahasa. Model bahasanya ikut dilatih; hasilnya tidak otomatis berlaku pada model bahasa tanpa adapter maupun fine-tuning.

### R3. Zhang dkk. (2026), mmMind

**Sitasi:** Zhang, D., Yin, Z., Yao, Z., Qin, H., Zhang, X., Yang, H., Sun, J., Wang, J., Fan, Z., Magno, M., & Zhang, D. (2026). *Teaching Foundation Models to Read mmWave: Pose-Guided Kinematic Representation for Human Behavior Understanding* [preprint]. [arXiv:2608.04127v2](https://arxiv.org/html/2608.04127v2).

**Klaim yang dapat disitasi.** mmMind menerima titik radar dengan kanal x, y, z, dan kecepatan radial; encoder menghasilkan satu token kontinu per frame. Kepala pose membaca token tersebut dan dilepas saat inferensi. Pada tahap QA, proyektor dan adapter LoRA dilatih, sedangkan bobot dasar LLM tetap beku. Table 2 melaporkan skor QA satu putaran pada mmMind-Bench: 89,2 untuk model lengkap, 68,6 tanpa pralatih pose, dan 85,0 tanpa loss kinematik tambahan. Lihat §3 untuk pose-guided pretraining dan jalur bahasa, Table 2 untuk ablasi, serta Appendix C untuk tahapan pelatihan.

**Mengapa dipakai.** Ini preseden radar yang paling dekat dengan rancangan kinematika dan QA. Penulis tidak boleh mengklaim bahwa supervisi pose pada token radar atau QA spasial-temporal dari radar merupakan hal pertama. Pertanyaan penelitian yang tersisa perlu dinyatakan lebih sempit: apakah target fisik tetap terbaca setelah representasi kaya dikompresi menjadi K token pada budget terkontrol, dan apakah model bahasa tanpa adapter terlatih memakai token tersebut.

**Batas.** mmMind sudah memberi supervisi fisik pada token per frame; jangan gambarkan loss-nya hanya bekerja pada fitur yang sepenuhnya terpisah dari token. Titiknya juga memakai kecepatan radial, sedangkan RPC M4Human yang dijelaskan pada suplemen memuat intensitas sebagai kanal keempat. Pembeda yang dapat diuji ialah tahap kompresi lanjutan dengan pembanding sepadan dan probe pada keluaran K token. Skor mmMind-Bench tidak menjadi baseline numerik M4Human.

### R4. Shin dkk. (2026), mmWave-QA

**Sitasi:** Shin, J., Kim, J., Ko, D., & Choi, J. (2026). *Can Language Models Understand mmWave Data? Benchmarking Large Language Models for mmWave Radar-Based Human Understanding* [CVPR 2026 Findings preprint]. [arXiv:2608.14179](https://arxiv.org/html/2608.14179v1).

**Klaim yang dapat disitasi.** mmWave-QA menyajikan point cloud radar sebagai teks untuk LLM siap pakai dan menguji lima jenis pertanyaan di beberapa skenario. Panelnya meliputi urutan aksi dan fokus gerak anggota tubuh; tugas lainnya meliputi pengenalan aktivitas, perpindahan lintasan, dan jumlah aksi. Lihat abstrak dan §3.1.

**Mengapa dipakai.** Paper ini memastikan masalah “QA dari radar” sudah mempunyai pendahulu yang nyata. Ia juga memberi contoh bahwa pertanyaan temporal atau tentang bagian tubuh tidak harus selalu dijawab melalui token laten terlatih. Perbandingan konsep dengan tekstualisasi dapat membantu penulis menjelaskan alasan memilih token.

**Batas.** mmWave-QA tidak melatih kompresor gerak yang kemudian diuji untuk fidelitas posisi dan kecepatan. Fitur radar yang dipakai oleh benchmark gabungan itu tidak boleh disamakan begitu saja dengan empat kanal RPC M4Human. Jangan sebut penelitian baru sebagai benchmark QA radar pertama.

### R5. Liang dkk. (2026), Wave2Body

**Sitasi:** Liang, B., Gong, C., Gao, W., & Xu, C. (2026). *Wave2Body: Rethinking mmWave Human Pose Estimation as Radar-to-Body Token Translation* [preprint]. [arXiv:2607.18875](https://arxiv.org/html/2607.18875).

**Klaim yang dapat disitasi.** Wave2Body melatih tokenizer radar secara swasupervisi, melatih tokenizer tubuh dari pose, lalu mempelajari penerjemah antara kedua ruang token untuk estimasi pose. Paper menguji M4Human dan mmBody. Lihat §3.1–3.4 dan §4; penjelasan masalah observasi radar yang ambigu ada pada §1.

**Mengapa dipakai.** Radar mengukur pantulan dari tubuh, bukan koordinat sendi. Wave2Body menunjukkan bahwa kualitas keadaan fisik sebelum kompresi layak diuji tersendiri; kegagalan membaca posisi dari token akhir bisa berasal dari persepsi radar, bukan dari kompresor.

**Batas.** Target utama Wave2Body ialah estimasi pose, bukan tanya jawab dengan LLM. Token tubuhnya merupakan ruang keluaran untuk merekonstruksi pose; jangan samakan langsung dengan token ringkas yang masuk ke proyektor bahasa. Hasil M4Human mereka tidak menjadi angka pembanding bagi QA tanpa data dan protokol yang sama.

### R6. Fang dkk. (2025), HuMoCon

**Sitasi:** Fang, Q., Tang, C., Tekin, B., Ma, S., & Yang, Y. (2025). *HuMoCon: Concept Discovery for Human Motion Understanding*. CVPR, 7179–7190. [Naskah prosiding](https://openaccess.thecvf.com/content/CVPR2025/html/Fang_HuMoCon_Concept_Discovery_for_Human_Motion_Understanding_CVPR_2025_paper.html); [versi HTML](https://arxiv.org/html/2505.20920v1).

**Klaim yang dapat disitasi.** HuMoCon memakai objektif *velocity reconstruction* untuk menjaga perubahan gerak pada representasi motion/video. Pada Table 3, skor BABEL-QA model lengkap ialah 0,711 dan turun menjadi 0,637 ketika dua objektif pendukung velocity reconstruction ditiadakan. Lihat §3.2 dan Table 3.

**Mengapa dipakai.** Ablasi ini mendukung alasan mengukur perubahan gerak, bukan hanya label tindakan atau kemiripan rekonstruksi. Ia juga menunjukkan bahwa objektif berbasis velocity sudah mempunyai preseden.

**Batas.** “Velocity” di HuMoCon merujuk pada perubahan state/fitur dan optical flow; jangan ubah angka tersebut menjadi bukti kelajuan pelvis dalam meter per detik. HuMoCon memakai masukan motion/video dan melakukan instruction tuning pada LLM. Penurunan BABEL-QA bukan akibat kompresi token radar yang diisolasi.

### R7. Markhorst dkk. (2026), FiGMo

**Sitasi:** Markhorst, T., Lin, Z.-Y., Chew, J. Y., van Gemert, J., & Zhang, X. (2026). *Fine-grained Human Motion Understanding with Language Models* [preprint]. [arXiv:2606.20888](https://arxiv.org/html/2606.20888v1).

**Klaim yang dapat disitasi.** FiGMo memberi timestamp eksplisit pada rangkaian pose. Pada ablasi BABEL-QA, skor model lengkap 0,758 dan varian tanpa timestamp 0,711 menurut protokol penulis. Lihat §3 untuk representasi pose-waktu dan §4.4/Table 4 untuk ablasi.

**Mengapa dipakai.** Penulis skripsi dapat memakai FiGMo untuk menjelaskan mengapa asal waktu sebuah token perlu dijaga ketika pertanyaan menyangkut urutan atau perubahan gerak. Ketepatan model bahasa pada satu klip tidak cukup jika waktu asal geraknya hilang.

**Batas.** Masukan FiGMo berupa pose dua atau tiga dimensi yang sudah tersedia, bukan RPC. Ablasi timestamp bukan percobaan yang mengubah jumlah token radar. Skornya tidak boleh diperlakukan sebagai baseline numerik M4Human.

Skor BABEL-QA dari Endo dkk., HuMoCon, dan FiGMo juga tidak layak dibandingkan sebagai satu papan peringkat tanpa memeriksa evaluatornya. Endo dkk. melaporkan akurasi jawaban terstruktur; HuMoCon dan FiGMo memakai penilaian keluaran generatif berbantuan model bahasa pada sebagian hasil. Angka ablasi di atas berguna untuk membaca efek komponen **di dalam paper masing-masing**.

### R8. Huang dkk. (2026), SeMoCo

**Sitasi:** Huang, T., Guo, H., Cai, Z., Wang, S., Zhang, Y., Fan, Z., Song, X., Wu, G., & Zheng, X. (2026). *SeMoCo: A Semantic-First Motion Codec for Motion Language Modeling* [preprint]. [arXiv:2608.24334v2](https://arxiv.org/html/2608.24334v2).

**Klaim yang dapat disitasi.** SeMoCo memisahkan token semantik dari residu kinematik dan melatih decoder dari token terkuantisasi menggunakan galat posisi sendi, perubahan kecepatan, serta percepatan. Bukti berada pada §3.2 dan rincian objektif di §8.3.

**Mengapa dipakai.** Paper ini langsung membatasi klaim kebaruan: supervisi kinematik setelah tokenisasi telah dicoba. Penelitian radar perlu menerangkan apa yang berbeda pada sumber sensor, ukuran token yang dikendalikan, pengukuran sebelum dan sesudah kompresi, serta penggunaan token oleh LLM beku.

**Batas.** SeMoCo menargetkan generasi gerak bersyarat bahasa; hasil rekonstruksi dan generasinya tidak membuktikan kualitas QA dari radar. Jangan memakai satu jenis loss sebagai satu-satunya dasar klaim metode pertama.

## 3. Rujukan pendukung: pakai saat argumennya diperlukan

| Sumber dan status | Klaim yang layak diambil; alasan membaca | Batas dan tempat yang cocok |
|---|---|---|
| [Li dkk. (2025), SensorLLM](https://aclanthology.org/2025.emnlp-main.19/), EMNLP, DOI [10.18653/v1/2025.emnlp-main.19](https://doi.org/10.18653/v1/2025.emnlp-main.19) | Deret waktu sensor wearable dapat diselaraskan dengan deskripsi tren untuk pengenalan aktivitas; berguna sebagai latar sensor-bahasa. | IMU/wearable dan HAR, bukan RPC atau QA kinematik. Cukup satu kutipan ringkas dalam latar, rincian tahap pelatihan dapat masuk Bab II. |
| [Endo dkk. (2023), Motion Question Answering via Modular Motion Programs](https://proceedings.mlr.press/v202/endo23a.html), ICML, 9312–9328 | HumanMotionQA ialah tugas; BABEL-QA ialah dataset yang mereka susun dari BABEL/AMASS untuk tugas tersebut. Pertanyaannya memeriksa petunjuk motorik lokal, atribut, dan hubungan temporal. | Bukan data radar. Bedakan nama tugas dari nama dataset, serta evaluatornya dari studi generatif berikutnya. |
| [Cao dkk. (2025), MotionCtrl](https://openaccess.thecvf.com/content/ICCV2025/html/Cao_MotionCtrl_A_Real-time_Controllable_Vision-Language-Motion_Model_ICCV_2025_paper.html), ICCV, 12253–12262 | Part-aware residual quantization sudah dipakai untuk kontrol bagian tubuh pada generasi gerak. | Bukan QA radar; dipakai untuk membatasi klaim “token berbasis bagian tubuh pertama”. |
| [Gu dkk. (2026), MoTok](https://arxiv.org/html/2603.19227v1), preprint arXiv:2603.19227 | Token gerak ringkas juga diuji pada motion-to-text captioning dengan captioner yang dilatih; lihat Appendix A.2 dan D.3. | Bukan pembuktian QA kinematik radar dengan LLM tanpa adapter. Cocok untuk pembahasan karya terkait di Bab II. |
| [Hewitt dan Liang (2019), Designing and Interpreting Probes with Control Tasks](https://aclanthology.org/D19-1275/), EMNLP, DOI [10.18653/v1/D19-1275](https://doi.org/10.18653/v1/D19-1275) | Akurasi probe dapat mencerminkan kemampuan probe mempelajari target, bukan semata isi representasi; alasan menyamakan kapasitas dan memakai kontrol. | Contoh aslinya bahasa, sehingga ini dasar metode evaluasi, bukan hasil radar. |
| [Goyal dkk. (2017), Making the V in VQA Matter](https://openaccess.thecvf.com/content_cvpr_2017/html/Goyal_Making_the_v_CVPR_2017_paper.html), CVPR, 6904–6913 | Pertanyaan sama dengan input pengamatan yang mirip tetapi jawaban benar berbeda membantu menguji ketergantungan pada pengamatan. | Aslinya VQA gambar. Pasangan donor radar tetap perlu label, dukungan waktu, dan kelas aktivitas yang sah. |
| [Li dkk. (2023), BLIP-2](https://proceedings.mlr.press/v202/li23q.html), ICML, 19730–19742 | Modul penghubung yang dilatih dapat memasukkan fitur nonteks ke LLM beku; preseden rancangan projector/query. | Modalitasnya gambar. Jangan jadikan bukti bahwa proyektor radar pasti berhasil. |

## 4. Isi folder jurnal lokal

| Berkas | Keputusan baca | Alasannya |
|---|---|---|
| [RadarLLM.pdf](<../jurnal/RadarLLM.pdf>) | Inti | Sumber terdekat di folder untuk radar point cloud, tokenisasi, dan bahasa. Gunakan metadata AAAI 2026, bukan tahun unggah preprint 2025. |
| [EMNLP 2025 Paper 19.pdf](<../jurnal/EMNLP 2025 Paper 19.pdf>) | Pendukung | Ini paper SensorLLM, berguna untuk sensor-bahasa; tugas utamanya HAR wearable. |
| [LLaSA Sensor-Aware LLM.pdf](<../jurnal/LLaSA Sensor-Aware LLM.pdf>) | Pendukung Bab II | [LLaSA](https://arxiv.org/abs/2406.14498) menguji tanya jawab dari sensor wearable, tetapi domain dan pelatihannya berbeda. PDF lokal ialah v3; arXiv telah memuat v4, sehingga jangan mencampur angka dataset dari dua versi. |
| [SensorLM- Learning the Language of Wearable Sensors.pdf](<../jurnal/SensorLM- Learning the Language of Wearable Sensors.pdf>) | Pendukung Bab II | [SensorLM](https://proceedings.neurips.cc/paper_files/paper/2025/hash/42cd98f0e7520d4a63c34891ac1c972f-Abstract-Conference.html) memperluas riset sensor-bahasa pada wearable; tidak menjawab pelestarian kinematik token radar. |
| [Reasoning in Signals Survey.pdf](<../jurnal/Reasoning in Signals Survey.pdf>) | Bacaan orientasi | Survei dapat membantu mencari istilah dan jalur sitasi, tetapi klaim teknis utama sebaiknya merujuk paper asli yang diuji. |
| [World Models for Cognitive Agents.pdf](<../jurnal/World Models for Cognitive Agents.pdf>) | Tidak perlu untuk Bab I | Fokus proposal sekarang bukan forecasting atau world model umum. Memakainya sebagai pijakan utama akan mengaburkan masalah kompresi token radar dan QA terukur. |

Folder lokal belum memuat M4Human, mmMind, mmWave-QA, Wave2Body, HuMoCon, FiGMo, dan SeMoCo. Tautan primer untuk semua sumber itu tercantum di atas; pembaca tidak perlu mengasumsikan bahwa hanya PDF yang sudah diunduh yang boleh menjadi rujukan.

## 5. Klaim yang perlu dijaga saat kelak menulis Bab I

1. **Bukan “radar dan LLM pertama”.** RadarLLM, mmMind, dan mmWave-QA sudah lebih dahulu menghubungkan radar dengan bahasa atau QA.
2. **Bukan “loss kecepatan pertama”.** HuMoCon dan SeMoCo sudah memakai supervisi perubahan gerak; target, posisi loss, dan tugasnya perlu dibedakan.
3. **Bukan “kompresi terdahulu terbukti gagal”.** RadarLLM melaporkan perubahan metrik yang bercampur setelah VQ. Ada atau tidaknya kehilangan target kinematik pada budget M4Human harus diukur.
4. **Bukan “LLM beku pasti memahami radar”.** mmMind membekukan bobot dasar, tetapi melatih adapter LoRA saat QA; rancangan tanpa adapter tetap perlu pengujian. Jawaban benar juga bisa muncul dari prior bahasa.
5. **Bukan “661 ribu sampel independen tersedia untuk eksperimen”.** Jumlah frame paper berbeda dari jumlah sekuens, subjek, window sah, dan isi paket processed yang benar-benar diterima.
6. **Bukan “kelajuan pelvis sama dengan gaya atau pusat massa”.** Kinematika yang diamati tidak mengukur seluruh parameter fisika tubuh. Tren pada satu interval juga tidak berarti setiap frame bergerak monoton.
7. **Bukan “hasil metode sudah terbukti”.** Paper lain memberi alasan dan pembanding konsep. Keunggulan token usulan memerlukan audit data, probe pada token akhir, pembanding kompresor sepadan, QA beracuan, serta uji subjek terpisah.

## 6. Batas bukti yang harus datang dari penelitian sendiri

Literatur dapat menetapkan bahwa data dan pendekatan serupa ada; literatur tidak dapat menggantikan audit paket M4Human yang akan dipakai. Sebelum menyatakan target posisi atau kecepatan sah, peneliti perlu memeriksa skema RPC, satuan, waktu, koordinat, joint map, dan kesesuaian rekaman dengan anotasi. Split subjek, aturan jawaban, serta ambang ketidakpastian harus dikunci pada data latih dan validasi.

Untuk pertanyaan pelestarian token, sumber representasi sebelum kompresi dan token akhir perlu diuji pada target yang sama dengan probe independen. Kondisi dasar dan kondisi bersupervisi kinematik harus memakai kompresor, jumlah token, sumber data, serta objektif rekonstruksi yang sepadan; perbedaannya terletak pada tempat gradien fisik mencapai kompresor. Keberhasilan probe tidak otomatis berarti LLM memakai informasi itu. Uji QA memerlukan pembanding tanpa sensor dan pasangan input radar dengan pertanyaan sama tetapi jawaban acuan berbeda. Laporan biaya harus memakai pengukuran aktual, bukan perkiraan dari ukuran dataset.
