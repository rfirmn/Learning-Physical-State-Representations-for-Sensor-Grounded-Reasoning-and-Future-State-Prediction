# Proposal Riset Terbaru — Pelestarian Kinematik pada Token Radar Ringkas

**Judul kerja:** *Kinematic-Preserving Radar Tokenization for Sensor-Grounded Temporal Question Answering with a Frozen Language Model*

**Revisi menyeluruh:** 4 Oktober 2026. **Kontrak:** `m4human_kinetok_v3`.

**Status:** usulan metode dan protokol proof of concept; bukan hasil eksperimen. Dataset processed M4Human berada pada perangkat lain. Root `/dataset`, kualitas label, runtime, dan kelayakan GPU belum diverifikasi dari checkout ini.

> Posisi bersama: penelitian ini mengembangkan metode sekaligus menguji hipotesis tentang pelestarian informasi kinematik pada token akhir. Keberhasilan radar-to-text sebelumnya menjadi pijakan kelayakan, bukan bukti pemahaman sensor universal atau bukti bahwa kompresi RadarLLM gagal. Forecasting, HAR semata, dan biomekanika penuh bukan tujuan utama.

Nama repositori, instruksi arsitektur MM-Fi, checkpoint, dan laporan lama merupakan konteks historis. Aturan operasional proyek tetap berlaku, tetapi arah ilmiah M4Human mengikuti proposal ini. Artefak lama tidak dihapus dan tidak dipresentasikan sebagai hasil penelitian baru.

## 1. Ringkasan penelitian dalam satu paragraf

Penelitian ini mengembangkan dan mengevaluasi metode pembentukan token gerak dari radar processed M4Human yang mempertahankan informasi kinematik terpilih pada representasi ringkas untuk digunakan oleh frozen LLM. Masalah yang diteliti bukan apakah radar dapat dihubungkan dengan bahasa untuk pertama kalinya, melainkan sejauh mana informasi bagian tubuh, perubahan kecepatan, dan relasi temporal tetap dapat digunakan setelah kompresi, serta apakah pembelajaran yang secara langsung menjaga informasi tersebut pada token akhir memberi manfaat dibandingkan tokenisasi dasar yang sepadan. Metode usulan menggabungkan representasi gerak spasial-temporal dengan kompresor terlatih dan supervisi kinematik pada keluaran token, kemudian memisahkan pembuktian keterbacaan informasi dari pembuktian pemanfaatannya oleh LLM. Kontribusi yang ditargetkan adalah metode yang terdefinisi, karakterisasi trade-off budget token dengan kualitas representasi, dan bukti empiris sensor-grounded QA pada tugas yang labelnya dapat diverifikasi. Kebaruan dan keunggulan merupakan klaim yang harus diuji, bukan asumsi; hasil negatif yang valid membatasi hipotesis, sedangkan kegagalan implementasi atau data menghasilkan kesimpulan yang belum dapat ditentukan. Scope dibatasi pada kinematika manusia yang diamati, bukan seluruh hukum fisika, gaya, torsi, niat, atau prediksi masa depan sebagai tujuan utama.

## 2. Apa yang menjadi masalah, dan apa yang belum diketahui?

### 2.1 Masalah praktis

Sekuen radar dan gerak mengandung informasi menurut bagian tubuh dan waktu. Antarmuka bahasa membutuhkan representasi yang cukup ringkas agar biaya komputasinya dapat dikelola. Pengurangan panjang sekuens, pooling, atau kuantisasi dapat membuang variasi yang tidak relevan maupun informasi yang diperlukan suatu tugas. Karena itu, kompresi tidak dianggap selalu buruk atau selalu baik.

Contoh kebutuhan informasi: membedakan peningkatan dan penurunan speed, membandingkan gerak tangan dengan kaki relatif terhadap tubuh, atau menentukan urutan onset dua bagian tubuh. Label aktivitas yang sama tidak otomatis menentukan jawaban pertanyaan tersebut.

### 2.2 Ketidakpastian ilmiah

1. Apakah informasi target sudah tersedia pada representasi sebelum kompresi? Jika belum, masalahnya dapat berasal dari persepsi atau pembelajaran gerak.
2. Apakah keterbacaan informasi menurun setelah kompresi pada budget tertentu? Ini perlu pengukuran, bukan disimpulkan dari caption yang kurang baik.
3. Apakah supervisi pada token akhir memperbaiki pelestarian informasi, dibandingkan kompresor yang sama tanpa keterikatan supervisi fisik pada keluaran token?
4. Apakah perbaikan representasi juga berguna bagi frozen LLM, atau masih terhalang alignment dan keterbatasan model bahasa?

**Masalah penelitian tidak mensyaratkan bahwa studi sebelumnya telah terbukti gagal.** Ia dapat berangkat dari bukti kelayakan sensor-ke-bahasa dan pertanyaan tentang karakter serta batas representasi yang digunakan.

### 2.3 Dasar teori dan batas inferensi

**T — Identitas:** mean pooling biasa tidak membedakan urutan `[A,B]` dan `[B,A]`. Namun, bila fitur A/B sudah mengodekan riwayat dan waktu, identitas itu tidak membuktikan seluruh sistem kehilangan urutan. Kehilangan aktual harus diperiksa pada representasi terlatih.

**R — Alasan desain:** rekonstruksi latent menjaga kesetiaan pada representasi sumber, tetapi belum menjamin fidelitas besaran yang penting untuk suatu QA. Sebaliknya, decoder kinematik yang baik sebelum kompresi belum menjamin token sesudah kompresi cukup informatif. Karena itu, objective dan pengujian ditempatkan pada keluaran token itu sendiri.

**H — Hipotesis:** supervisi kinematik pada token akhir dapat mengarahkan kompresi agar mempertahankan informasi terpilih pada kapasitas terbatas. Penambahan loss, attention, atau nama modul tidak otomatis membuktikan manfaat tersebut.

Istilah “informasi dipertahankan” berarti **keterbacaan dan kegunaan pada target yang diuji**, bukan pengukuran seluruh informasi Shannon atau jaminan bahwa latent sepenuhnya invertibel.

## 3. Landasan literatur: bukti, batas, dan implikasi

| Penelitian | Bukti yang relevan | Batas bagi rumusan kita |
|---|---|---|
| [RadarLLM](https://arxiv.org/html/2504.09862v2), §4–5 dan Appendix H | Tokenizer radar dan model bahasa terlatih mendukung radar-to-text pada data sintetis/nyata. VQ meningkatkan ROUGE-L 29,8→31,2 pada setting radar-to-text-only, dengan METEOR yang menurun. | Tidak membuktikan kompresi selalu merusak. Urutan token tetap tersedia; evaluasi bahasa bukan pengukuran langsung pelestarian joint/velocity pada token. Model bahasa dilatih, bukan bukti kondisi frozen LLM kita. |
| [HuMoCon, CVPR 2025](https://arxiv.org/html/2505.20920v1), §3–4 | Mengangkat temporal over-smoothing dan kehilangan detail perubahan gerak. Penghilangan dua objective pendukung velocity reconstruction menurunkan skor BABEL-QA 0,711→0,637. | Bukti manfaat objective representasi, bukan isolasi pengurangan jumlah token. “Velocity” mencakup delta motion/state, bukan otomatis m/s. Supervisi perubahan gerak bukan ide pertama kita. |
| [FiGMo, preprint Juni 2026](https://arxiv.org/html/2606.20888v1), §4.4 | Timestamp grounding mendukung motion QA; penghapusannya menurunkan skor 0,758→0,711, yaitu 4,7 poin persentase. | Kehilangan waktu asal pose berbeda dari hilangnya seluruh urutan. Ablation timestamp bukan bukti kausal kompresi RadarLLM. |
| [MotionCtrl, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Cao_MotionCtrl_A_Real-time_Controllable_Vision-Language-Motion_Model_ICCV_2025_paper.html) | Mengusulkan part-aware residual quantization untuk kontrol bagian tubuh pada motion generation. | Tokenisasi berbasis bagian tubuh sudah mempunyai pendahulu; tugas generasi bukan sensor-grounded QA. |
| [SeMoCo, preprint Agustus 2026](https://arxiv.org/html/2608.24334v2), §3.2 | Memisahkan semantic token dan residual kinematic tokens; decoder dari token terkuantisasi disupervisi posisi, velocity, dan acceleration. | Bahkan supervisi kinematik sesudah tokenisasi sudah mempunyai pendahulu. Pembeda yang diusulkan harus mencakup konteks radar, budget terkontrol, serta validasi fisik dan frozen-LLM QA, bukan output-side loss saja. |
| [MoTok, preprint Maret 2026](https://arxiv.org/html/2603.19227v1), Appendix A.2/D.3 | Compact motion tokenizer dan diffusion decoding; juga menguji motion-to-text captioning dari token dengan captioner terlatih. | Perbaikan token→bahasa juga bukan ide pertama. Captioning motion bukan pengujian kinematik dan temporal QA dari radar dengan frozen LLM, tetapi tetap preseden yang dekat. |

Sumber di atas mendukung motivasi sekaligus membatasi klaim. Penelusuran ini bukan systematic review yang membuktikan ketiadaan seluruh metode serupa. Perbandingan empiris dengan RadarLLM/codec generatif hanya dilakukan jika data, tugas, kode, dan resource kompatibel; tanpa itu tidak ada klaim mengungguli paper tersebut.

[MotionBERT](https://arxiv.org/abs/2210.06551) menjadi preseden representasi joint–waktu, bukan checkpoint radar siap pakai. [SMD, preprint 2026](https://arxiv.org/html/2604.21668v2) memotivasi pembanding fakta kinematik deterministik; ia tidak membuktikan AI selalu diperlukan atau tidak diperlukan.

## 4. Rumusan masalah dan pertanyaan penelitian

**Pertanyaan utama:**

> Bagaimana membentuk token ringkas dari representasi gerak radar yang mempertahankan informasi kinematik terpilih, dan sejauh mana pelestarian tersebut meningkatkan kemampuan frozen LLM menjawab pertanyaan temporal dan relasional yang dapat diverifikasi?

| ID | Pertanyaan | Bukti yang diperlukan |
|---|---|---|
| RQ1 — Diagnostik | Sejauh mana informasi kinematik/temporal dapat dibaca sebelum dan setelah kompresi? | Probe kapasitas sepadan pada memori aktual sebelum kompresi dan token akhir; target, waktu, split, serta coverage sama. |
| RQ2 — Metode | Apakah supervisi kinematik pada token akhir memberi manfaat dibandingkan tokenisasi dasar yang sepadan? | `C_kin` vs `C_base` pada budget sama, sumber representasi tetap, dan protokol matched. |
| RQ3 — Pemanfaatan | Apakah frozen LLM menggunakan token untuk menjawab berdasarkan sensor, dan apakah manfaat RQ2 mencapai QA? | Projector independen yang dilatih sepadan, QA berlabel sah, text-only/direct probe, serta kontrol sensor berlabel benar. |
| RQ4 — Batas | Bagaimana kualitas dan biaya berubah menurut budget token dan pada subjek held-out? | Budget yang ditetapkan sebelum test, ukuran token/precision/latency, hasil per task/subjek, dan ketidakpastian berkelompok. |

RQ1 boleh menghasilkan “tidak ditemukan penurunan yang relevan pada tugas ini”. RQ2 boleh tidak mendukung keunggulan. RQ3 boleh menunjukkan bahwa perbaikan fisik belum dimanfaatkan LLM. Tidak ada tuntutan bahwa semua jawaban harus positif.

## 5. Hipotesis yang dapat ditolak

- **H1 — Pelestarian:** pada budget yang sama, `C_kin` meningkatkan keterbacaan besaran kinematik atau task-probe terpilih dibandingkan `C_base`, tanpa degradasi tak dapat diterima pada target utama lainnya.
- **H2 — Kegunaan:** token `C_kin` membantu frozen LLM menjawab panel QA lebih baik daripada `C_base` dengan alignment yang sepadan. H1 tidak otomatis membuktikan H2.
- **H3 — Grounding terbatas:** jawaban mengikuti perbedaan sensor yang reference-nya diketahui, tidak hanya prior bahasa, label aktivitas, atau distribusi kelas.

Toleransi non-inferiority/degradasi yang diperbolehkan, efek praktis minimum, agregasi metrik, dan panel task ditetapkan pada train/validation sebelum test. Proposal tidak menetapkan angka kemenangan arbitrer atau menjanjikan statistical significance.

## 6. Tujuan, kontribusi, dan kebaruan

### 6.1 Tujuan

1. Mengembangkan metode tokenisasi gerak yang ter-grounding secara kinematik dari observasi radar.
2. Mengukur pelestarian target pada **token akhir yang benar-benar masuk projector**, bukan hanya latent penuh.
3. Menguji manfaat pelestarian bagi QA temporal/relasional dengan frozen LLM.
4. Mengkarakterisasi batas kualitas–budget–biaya dan melaporkan hasil positif, parsial, atau negatif secara terpisah.

### 6.2 Kontribusi yang ditargetkan

| Bentuk kontribusi | Isi | Syarat klaim |
|---|---|---|
| Metode | Kompresor token dengan supervisi posisi/velocity pada keluaran ringkas dan support waktu/bagian tubuh yang jelas | Spesifikasi dapat direproduksi dan pembanding yang tidak sengaja dilemahkan. Penamaan modul bukan bukti kebaruan. |
| Empiris | Pelestarian kinematik sebelum/sesudah kompresi menurut budget | Probe matched, sumber target valid, serta pemisahan error sensor dan kompresi. |
| Pemanfaatan | Hubungan kualitas token dengan jawaban frozen LLM | QA benar secara terstruktur dan mengikuti bukti sensor; bukan kelancaran caption saja. |
| Protokol terbatas | Split, recipe label, kontrol grounding, serta laporan biaya yang dapat diulang | Bukan klaim benchmark publik besar atau dataset QA baru sebelum artefak dan lisensinya tersedia. |

### 6.3 Letak kebaruan yang diusulkan

Kandidat kebaruan adalah **pengembangan dan validasi output-side kinematic preservation pada token ringkas yang berasal dari radar, di bawah budget terkontrol, dengan pemeriksaan keterbacaan fisik dan pemanfaatan oleh frozen LLM**. Ini kontribusi metodologis/empiris yang mungkin bersifat inkremental, bukan klaim bahwa seluruh komponennya baru. SeMoCo menunjukkan bahwa supervisi kinematik setelah tokenisasi sendiri sudah mempunyai preseden; MoTok juga mengevaluasi token→caption. Karena itu, output-side loss atau token→bahasa secara terpisah tidak diposisikan sebagai pembeda baru.

Bukan klaim kebaruan: radar-to-language pertama; tokenizer gerak pertama; velocity loss pertama; attention pertama; memakai M4Human; membekukan LLM; atau merangkai encoder–dynamics–projector. Modifikasi internal bernilai ilmiah bila alasan, treatment yang diuji, dan kontribusinya terukur.

Jika metode tidak unggul, klaim keunggulan ditarik; spesifikasi metode tetap ada dan hasil yang valid dapat mengungkap batasnya. Nilai hasil negatif tidak otomatis menjamin publikasi atau novelty tinggi.

## 7. Scope dan definisi operasional

**Kinematika:** posisi joint relatif pelvis, trajectory pelvis global, velocity global/relatif, speed, perubahan speed, dan relasi gerak antarbagian tubuh. Target velocity merupakan estimasi turunan trajectory berwaktu, bukan channel Doppler yang diasumsikan tersedia.

**Pemahaman operasional:** kemampuan sistem menjawab task terpilih secara benar dan mengikuti bukti sensor pada held-out data. QA kategori juga dapat dipelajari sebagai pemilihan kelas; hasil ini tidak membuktikan reasoning umum atau pemahaman fisika seperti manusia.

**Tokenisasi:** pembentukan tensor ringkas dari observasi/representasi gerak. Token tidak wajib diskret atau bagian dari vocabulary teks.

**Motion/dynamics representation:** pembelajaran keadaan dan perubahan dalam interval yang telah diamati. Tanpa model transisi dan rollout yang diuji, modul ini bukan world model atau autonomous dynamics simulator.

**Tidak termasuk scope utama:** gaya, torsi, kontak terukur, COM, energi mekanis, diagnosis keseimbangan, niat, causal counterfactual, dan forecasting. Pelvis bukan otomatis COM; causal attention bukan pemahaman sebab-akibat.

## 8. Dataset dan target yang dapat dipertanggungjawabkan

M4Human dipilih karena menyediakan radar dan anotasi gerak/trajectory, bukan hanya label aktivitas. Paper melaporkan sekitar 661K frame, 999 sekuens, 20 subjek, dan 50 aktivitas; sensor sekitar 12 Hz disejajarkan dengan MoCap 100 Hz. Angka tersebut fakta sumber, bukan jumlah contoh valid pada paket pengguna atau bukti generalisasi universal. [M4Human, §3](https://arxiv.org/html/2512.12378v3).

Paket processed sekitar 50 GB menurut konteks pengguna; ukuran aktual dan isi di perangkat eksperimen tetap diaudit. Pilihan utama RPC, dengan input XYZ/intensity setelah verifikasi schema; RGB, depth, dan RT tidak menjadi requirement. Helper resmi memeriksa empat kanal tersebut. [Kode preprocessing resmi](https://github.com/FanJunqiao/M4Human/blob/main/dataset/m4human_utils.py).

| Jalur | Data | Penggunaan |
|---|---|---|
| Inference utama | RPC, timestamp/frame order, kalibrasi yang sah, validity sensor | Membentuk predicted state, representasi gerak, dan token. |
| Target offline | Joints/body parameters yang diaudit serta waktu | Target posisi, velocity, dan label task. Tidak masuk token/prompt. |
| Metadata | Subject/action/recording/index | Split, sampling, provenance, laporan; bukan jawaban tersembunyi. |

Joint map, transformasi koordinat, satuan, pelvis, dan hubungan `trans` dengan joints belum boleh ditebak. GT velocity dihitung dengan recipe terverifikasi; default local polynomial endpoint tujuh titik derajat dua, menggunakan timestamp aktual dan warm-up validity. Interpolasi label processed tidak menghasilkan pengukuran MoCap baru.

Pemisahan subjek dan seluruh support recording/context/derivative wajib bebas kebocoran. Banyak sliding windows tidak berarti banyak sampel independen. MM-Fi dan checkpoint lama hanya sejarah atau sumber transfer setelah audit, bukan hasil M4Human.

## 9. Tugas pembuktian minimum

| Task | Contoh | Target |
|---|---|---|
| `root_speed_trend` — wajib | Selama interval ini, apakah speed pelvis secara keseluruhan cenderung meningkat, menurun, atau tidak menunjukkan tren naik/turun yang cukup kuat? | Slope robust speed pelvis global terhadap waktu pada support yang dikunci. |
| `relative_limb_motion` — wajib | Apakah bagian tubuh yang ditanyakan bergerak lebih cepat daripada bagian pembanding relatif terhadap pelvis? | Agregat norm velocity relatif joint/segmen yang namanya diaudit. |
| `limb_onset_order` — bersyarat | Bagian tubuh mana mulai bergerak lebih dahulu? | Onset, threshold, durasi minimum, tie/undefined, dan reliabilitas yang diverifikasi. |

Dua task pertama adalah P0. Klaim kemampuan urutan antarperistiwa hanya dibuat jika task onset lolos quality gate dan benar-benar diuji. Jika tidak, scope dinyatakan sebagai perubahan speed dan relasi tubuh selama interval, bukan event-order reasoning.

Untuk `root_speed_trend`, `speeding_up`/`slowing_down` menyatakan tren keseluruhan, bukan perubahan monoton setiap frame. `no_overall_trend` menyatakan slope dalam deadband pada support/fit yang sah; speed tetap dapat naik lalu turun dalam interval. `unknown` digunakan jika support atau kualitas fit tidak memadai. Definisi dan threshold mengikuti recipe dokumen 03.

Label dibentuk dari recipe deterministik yang diaudit, bukan tebakan LLM atau label HAR. Label invalid tidak disamakan dengan “tidak bergerak”; coverage, ambiguity, tie, dan unknown dilaporkan. Dukungan task mungkin hanya sebagian aktivitas/window. Sudut, acceleration, fase, dan repetisi merupakan perluasan bersyarat.

## 10. Prinsip metode dan kontrak pembanding

Alur konseptual:

```text
RPC → encoder sensor → predicted states + features
    → learned motion representation H
    → learned compact tokenizer C → U
                                 ├─ readout/probes → bukti pelestarian kinematik
                                 └─ projector → frozen LLM → QA terverifikasi
```

Decoder/readout fisik adalah jalur pembuktian dan supervisi, **bukan tahap yang harus mengubah GT menjadi jawaban untuk LLM**. Input utama LLM adalah `Projector(U)`, bukan GT, keluaran readout, atau raw-feature bypass.

### 10.1 Dua kondisi utama yang sepadan

Satu encoder dan satu motion backbone dipilih pada train/validation lalu dibekukan untuk kedua kondisi. Ini mengendalikan kualitas persepsi dan representasi sumber pada eksperimen kompresi.

| Unsur | `C_base` | `C_kin` |
|---|---|---|
| Sumber | Memori joint–waktu M yang sama, sensor-derived | Identik |
| Kompresor | Learned bin-query attention, K token × 256 dimensi | Arsitektur, initialization protocol, dan budget sama |
| Fidelity objective | Rekonstruksi seluruh H yang dibekukan dari U | Identik; bukan hanya summary joint-pooled |
| Auxiliary kinematic readout | Membaca memori sebelum kompresi M | Membaca token akhir U |
| Target/readout | Posisi/velocity, support dan kapasitas readout sama | Identik |
| Gradien auxiliary | Mengubah auxiliary head, tidak mencapai kompresor melalui M yang frozen | Mencapai kompresor melalui U |
| Language alignment | Projector sendiri, frozen LLM | Projector sendiri, protokol matched |

**Treatment utama adalah tempat keterikatan supervisi kinematik**, bukan sekadar penambahan raw data, model lebih besar, atau LLM yang dituning. Kompresor baseline tetap terlatih melalui fidelity objective yang mencakup **semua joint dan waktu pada H**; baseline tidak dibiarkan random atau hanya merekonstruksi summary yang sudah mengabaikan bagian tubuh.

Semua akses label upstream sama; perbedaan jalur gradien auxiliary dinyatakan terbuka. Jumlah parameter training/inference, objective weights, update budget, serta sumber label dicatat. Tidak mengklaim semua distribusi gradien identik atau bahwa kontrast ini membuktikan keunggulan arsitektur terhadap seluruh alternatif.

`C_history` vs `C_dynamics` dari v2 bukan primary comparison baru: akses predicted state, kapasitas, dan supervisinya berubah sekaligus. Perbandingan tersebut hanya diagnostik historis jika diperlukan.

### 10.2 Pemisahan tahap

1. **E:** latih/validasi encoder sensor, lalu freeze.
2. **M:** latih motion backbone dan physical readout, lalu freeze dan buat cache H/M/Z yang konsisten.
3. **C:** latih `C_base` dan `C_kin` beserta fidelity/auxiliary readouts dari cache tetap.
4. **L:** freeze seluruh upstream dan readout; latih projector saja dengan frozen LLM.
5. **Test:** evaluasi protokol yang telah dikunci, termasuk probes independen dan QA.

Modul gabungan motion+compressor dapat disebut tokenizer gerak, tetapi komponen di dalamnya mempunyai tahap training berbeda. Tidak mengalirkan QA gradient ke tokenizer pada kondisi utama.

### 10.3 Budget dan batas kompresi

Referensi H=32 waktu × 23 joint/global × 128 dimensi; M menggabungkan joint dan global menjadi 256 dimensi. Z=32 × 256 merupakan summary antara yang **sudah** melakukan pooling joint. U=K × 256 adalah token akhir.

K=16 menjadi primary budget; K=8 dan 32 adalah secondary budgets yang ditetapkan sebelum test. M→U adalah tahap kompresi aktual. Probe Z tambahan dapat mendiagnosis pooling sebelumnya; **selisih Z–U saja tidak mengisolasi kompresi M→U**.

Jumlah token, width, precision, dan biaya dibandingkan eksplisit; codebook size tidak sama dengan token count. Jika hanya satu budget feasible, laporkan single-budget proof of concept dan jangan mengklaim kurva trade-off menyeluruh.

## 11. Evaluasi dan arti hasil

Tiga lapis bukti dipisahkan: kualitas sumber fisik, pelestarian pada U, dan penggunaan oleh LLM.

- **Fisik:** MPJPE relatif pelvis, root/global position error, serta joint/root velocity error; satuan dan validity dinyatakan.
- **Representasi:** probe fisik independen yang dilatih pada source frozen M/U dengan arsitektur query/readout, label, split, dan training budget sepadan. Skor task-probe utama diperoleh dari prediksi posisi/velocity probe melalui recipe deterministik yang sama; classifier task langsung bukan bagian P0. Readout training yang berhasil bukan satu-satunya bukti.
- **Bahasa:** macro-F1/accuracy jawaban terstruktur pada panel identik; parse invalid dihitung salah, coverage dan unknown dilaporkan. BLEU/caption quality bukan metrik utama.
- **Grounding:** text-only/majority, direct rule/probe tanpa LLM, dan perubahan input dengan reference yang benar. Panel donor wajib mencakup pasangan recording dalam aktivitas sama dengan **reference jawaban berbeda**, pertanyaan identik, dan support waktu yang kompatibel; laporkan kedua jawaban benar sekaligus perubahan jawaban sesuai reference, coverage pasangan, dan hasil per task. Donor dengan reference sama dapat menguji invariance, bukan sensitivitas terhadap perubahan. Token shuffle adalah diagnostik gangguan/OOD, bukan bukti kausal fisik.
- **Biaya:** parameter trainable/total, panjang token, cache, peak VRAM/RAM, throughput, dan latency aktual. Tidak menjanjikan real-time sebelum profiling.

Komparasi pooled/explicit-state atau estimator matematis membantu diagnosis, bukan pengganti primary comparison. Oracle GT hanya batas atas berlabel terpisah. Raw M→LLM tidak wajib jika resource tidak cukup; pembanding precompression melalui probe tetap tersedia.

| Hasil | Kesimpulan yang diperbolehkan |
|---|---|
| C_kin lebih baik pada U dan QA grounded | Mendukung metode dan kegunaannya pada kondisi/task yang diuji. |
| U lebih baik, QA tidak | Manfaat representasi didukung; manfaat untuk frozen LLM belum didukung. |
| QA lebih baik, metrik fisik tidak | Manfaat task ditemukan; belum bukti perbaikan pelestarian kinematik terpilih. |
| Baseline tidak menunjukkan penurunan relevan | Bottleneck tidak terbukti pada target/budget ini; bukan bukti seluruh kompresi selalu aman. |
| Tidak ada keunggulan yang jelas | Laporkan efek, ketidakpastian, dan sensitivitas; interval lebar berarti belum konklusif, bukan bukti kesetaraan. |
| Training/data/gradien gagal | Kegagalan engineering atau validitas, bukan hasil negatif hipotesis ilmiah. |

Satu seed cukup untuk PoC eksploratif tetapi tidak untuk klaim stabil lintas-seed. Ketidakpastian berkelompok menurut subjek/recording, bukan frame independen. Model/threshold/task/budget dipilih pada validation; test tidak digunakan untuk mengganti objective.

## 12. Batas implementasi dan status keputusan

Target resource tetap RTX 3060 12 GB, RAM 16 GB, Pure-PyTorch, dan `.venv`/`uv`. Cache/streaming, microbatch, dan gradient accumulation adalah strategi biaya, bukan hasil pengukuran. Dataset 50 GB tidak dimuat sekaligus ke VRAM dan bukan dasar memperkirakan durasi training.

| Dikunci pada arah ilmiah | Kandidat implementasi yang masih harus diuji |
|---|---|
| Kinematic preservation pada token akhir, frozen LLM, M4Human utama | Encoder set sederhana dan motion backbone joint–waktu berkapasitas kecil |
| Primary C_base vs C_kin dari sumber tetap | Bin-query learned compressor, width dan hyperparameter |
| GT hanya target/evaluasi, split bebas kebocoran | Transfer encoder jika kompatibel; scratch sebagai jalur referensi |
| Posisi/velocity, dua QA terukur, event order bersyarat | Filter/threshold reliabilitas, jumlah update, precision dan microbatch |

Arsitektur internal adalah kendaraan pengujian hipotesis. Bila pilot mengharuskan perubahan, dokumentasikan pada train/validation dan terapkan simetris pada kondisi matched, sebelum membuka test.

Development kode dan fixture sintetis dapat dilakukan sebelum dataset tersedia, kemudian audit serta smoke/tiny-overfit nyata dijalankan pada perangkat M4Human. [Planning evaluasi §12.1–12.3](planning_m4human/05_evaluasi_testing.md#121-kesiapan-eksperimen-pertama-dan-development-lintas-perangkat) menetapkan kesiapan P0, log minimum sejak training encoder, validasi artefak dan diagnosis/iterasi setelah run pertama. Log lengkap mendukung analisis dan pemilihan perbaikan melalui train/validation, tetapi tidak menjamin peningkatan performa. Scope P0 tetap; test yang telah dibuka tidak menjadi dasar tuning ulang.

## 13. Peta baca dan tanggung jawab dokumen

| Dokumen | Tanggung jawab |
|---|---|
| [01 — Encoder dan data](planning_m4human/01_encoder.md) | Audit dataset, koordinat/waktu/joint map, input sensor, transfer/scratch, persepsi bersama dan cache. |
| [02 — Motion representation dan tokenizer](planning_m4human/02_dynamic_model.md) | Backbone, kompresor terlatih, token/provenance, baseline vs proposed, tahap freeze. |
| [03 — Decoder dan supervisi](planning_m4human/03_decoder.md) | Full-H readout, fidelity reconstruction, auxiliary kinematic readout, recipe target dan probes. |
| [04 — LLM layer](planning_m4human/04_llm_layer.md) | QA, actual U→projector→frozen LLM, autograd, alignment matched dan format output. |
| [05 — Evaluasi dan testing](planning_m4human/05_evaluasi_testing.md) | RQ→eksperimen→metrik→klaim, fairness, test lock, kontrol grounding, hasil negatif dan reproduksi. |

Kontrak v3 menggantikan deterministik 8 bin × 2 slot v2 pada penelitian baru. API/cache/checkpoint v2 tidak dianggap kompatibel tanpa migrasi. Semua path implementasi yang belum ada harus disebut **rencana**, bukan executable yang telah diverifikasi.

## 14. Kesimpulan final

Penelitian ini bukan proyek untuk membuktikan ulang bahwa sensor dapat dihubungkan ke bahasa, dan bukan proyek yang wajib memenangkan semua baseline. Ia mengusulkan mekanisme pelestarian kinematik pada token radar ringkas, mengukur apa yang dipertahankan atau hilang, dan memeriksa apakah frozen LLM memanfaatkan informasi itu pada task terpilih. Literatur memberi pijakan sekaligus menunjukkan bahwa tokenizer, velocity supervision, dan motion understanding sudah mempunyai pendahulu. Kontribusi kita harus berdiri pada treatment yang jelas, perbandingan yang adil, dan bukti pada token akhir serta QA; kekuatan klaim mengikuti hasil, bukan sebaliknya.
