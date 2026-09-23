# Rencana Pemulihan Tahap 4 — Revisi 2

**Status:** kode pemulihan dan [runbook](stage4_recovery_runbook.md) tersedia; training ulang dan gerbang empiris belum dijalankan.  
**Dasar:** pembacaan kode Tahap 2–4, laporan Batch 1, dan metadata segmentasi MM-Fi pada 23 September 2026.  
**Batas bukti:** fitur lengkap, QA JSONL, checkpoint, dan keluaran mentah Batch 1 tidak ada di checkout ini. Jumlah data rusak dan skor model belum dapat dihitung ulang.

## 1. Hasil evaluasi ulang rencana sebelumnya

| Keputusan rencana awal | Temuan evaluasi | Revisi |
|---|---|---|
| Menolak semua jendela yang melintasi batas repetisi | Dari 16.448 repetisi dalam CSV, hanya 4.625 (28,1%) yang panjangnya minimal 24 frame; A01 tidak memiliki repetisi sepanjang itu. Berdasarkan panjang rekaman di CSV, sekitar 9,4% jendela 24 frame ber-stride 4 berada utuh dalam satu repetisi. | Untuk pemulihan dengan Dynamics 16+8 yang ada, definisikan unit utama sebagai **rekaman aksi kontinu**. Tandai apakah jendela melintasi batas repetisi dan laporkan kedua strata. Jika riset menghendaki prediksi dalam satu repetisi, pilih horizon lebih pendek dan latih ulang Dynamics sebagai eksperimen terpisah. |
| Menggunakan B3 dengan menghapus token masa depan dari projector B4 | Projector yang hanya dilatih dengan token masa depan akan menerima distribusi input berbeda saat B3. B4 lama juga memakai Dynamics dari 16 frame, sedangkan B3 hanya melihat lima latent; keunggulan B4 dapat berasal dari 11 frame riwayat tambahan. | Berikan 16 latent riwayat yang sama kepada semua kondisi. Latih B3, B3P (delapan token persistensi), dan B4 dengan protokol sepadan; perbandingan primer B4–B3P menyamakan panjang input dan sumber riwayat. |
| Mengandalkan MPJPE Av2 sebagai bukti kualitas latent global | Pose head cross-attention menerima 16 patch token, sedangkan Tahap 3/4 memakai rata-ratanya, yaitu z_t. | Ukur kemampuan z_t menyandi setiap target lewat probe dengan label benar dan mask per target sebelum training projector. |
| Menggunakan degradasi MPJPE Dynamics sebagai validasi fisik | Evaluasi mengulang satu z prediksi ke 16 slot pose head, lalu membandingkannya dengan pose oracle dari patch token aktual. | Bandingkan Dynamics dengan prediksi ulang state terakhir dan baseline temporal sederhana. Jika perlu metrik pose, latih decoder z_t-ke-pose pada data train dan pakai decoder yang sama untuk latent prediksi serta latent teramati. |
| Menganggap observed-future B5 sebagai batas atas teoretis | B5 memakai latent encoder dari radar masa depan, bukan state fisik GT; distribusi projector juga memengaruhi skor. | Sebut B5 **diagnostik observasi masa depan**, dan latih projector untuk input B5 bila kondisi ini dipakai. Jangan menyebut skornya batas atas matematis. |
| Memasukkan semua tugas QA lama setelah parser diperbaiki | Normalisasi per frame menghilangkan centroid XYZ, skala metrik, serta offset Doppler/SNR sebelum encoder; beberapa label dan pertanyaan juga tidak cocok. | Mulai dari tugas relatif yang labelnya sah dan terbukti dapat dibaca dari z_t. Tunda depth/lateral absolut, jarak dalam meter, dan kecepatan translasi hingga sinyal metrik dipertahankan serta diuji. |

Estimasi jendela di atas dihitung dari [CSV segmentasi](../datasets/MM-Fi%20Dataset/MMFi_action_segments.csv) dengan panjang rekaman sesuai batas frame terakhir; ini belum menghitung frame radar yang mungkin hilang dalam artefak fitur. CSV memuat satu rentang terbalik, E02/S12/A02 **217–148**, yang harus dikarantina dan dicek ke sumber asli, tanpa koreksi otomatis ([baris 300](../datasets/MM-Fi%20Dataset/MMFi_action_segments.csv#L300)).

### Klaim riset yang dicakup

Pemulihan ini menguji dua pertanyaan terbatas: **apakah z_t membawa atribut fisik terpilih pada subjek baru**, dan **apakah transformasi prediktif Dynamics membantu SLM menjawab target masa depan dari riwayat yang sama**. Forecast deterministik berasal dari riwayat; ia tidak menambah informasi sensor baru. Benchmark ini belum menguji H3 tentang tujuan representasi predictive vs classification-only, H4 tentang LLM dengan observasi sensor terstruktur, RQ4/H5 tentang counterfactual berlabel, atau robustness lingkungan baru. Hipotesis tersebut memerlukan eksperimen tersendiri ([hipotesis proposal](sections-proposal/6-hipotesis.md), [pertanyaan riset](sections-proposal/5-pertanyaan-penelitian.md)).

## 2. Kontrak data dan tugas pemulihan

1. **Unit waktu:** satu jendela menggunakan 16 latent teramati sampai t dan target t+1 sampai t+8 pada laju 10 Hz yang dinyatakan konfigurasi. Periksa nomor frame asli, frekuensi nyata, GT yang sesuai, dan keberlanjutan sinyal di sekitar batas repetisi. Simpan penanda batas repetisi sebagai metadata analisis; jangan berikan informasi batas itu ke model tanpa kondisi pembanding yang jelas.
2. **Target pilot:** pilih satu atribut keadaan saat ini dan satu atribut **perubahan fisik sampai t+8** yang berdefinisi geometris jelas. Contohnya kelas sudut sendi relatif atau perubahan konfigurasi anggota tubuh yang dinormalisasi terhadap ukuran tubuh. Rumus, sumbu koordinat, unit, kelas ambigu, serta distribusi label harus diaudit pada train/val. Pilihan target dan metrik dibekukan sebelum test dibuka. Bila tidak ada target masa depan dengan distribusi bermakna dan di atas baseline persistensi, hentikan klaim manfaat forecast untuk Tahap 4.
3. **Label:** hitung satu state terstruktur lengkap per jendela valid, kemudian bentuk soal dari field tersebut. Field jawaban harus sama persis dengan yang ditanyakan. Jangan mengisi field tak berlabel dengan kelas default. Tinjau contoh per kelas, subjek, aksi, dan dekat ambang; tolak GT nol/pengulangan frame pada artefak lama. Ekstraktor pemulihan kini menolak GT yang hilang atau indeks frame di luar rentang.
4. **Observabilitas:** input encoder saat ini dipusatkan dan diskalakan per frame, sedangkan ekstra kanal distandardisasi per frame ([normalisasi](../eksperimen_model/datasets/transforms.py#L52)). Besaran absolut hanya boleh dihidupkan kembali melalui dua jalur yang dilaporkan terpisah: side-channel fisik yang disimpan dan diberi baseline tersendiri, atau perubahan preprocessing/encoder yang diikuti ekstraksi dan training ulang Tahap 2/3. Side-channel tidak boleh dihitung sebagai bukti bahwa z_t sendiri menyimpan besaran tersebut.
5. **Bahasa:** gunakan satu bentuk soal kanonis dan keluaran JSON ringkas untuk benchmark fisik utama. Variasi parafrase test adalah eksperimen bahasa terpisah dengan template train, val, dan test yang benar-benar terpisah. Output naratif tidak menjadi target utama.

## 3. Kondisi pembanding pada panel sampel yang sama

| Kondisi | Input dan training | Fungsi |
|---|---|---|
| B0 prior/persistensi | Kelas mayoritas untuk atribut saat ini; ulang state terakhir atau baseline temporal sederhana untuk target masa depan. | Menentukan kesulitan tugas dan nilai tambah Dynamics. |
| B1 teks saja | SLM tanpa slot sensor dan tanpa projector fisik. | Mengukur prior jawaban dari pertanyaan. |
| B2 probe langsung | Probe z_t untuk target saat ini, dengan loss hanya pada target yang sah. | Memeriksa keterbacaan target dari latent global; bukan pembanding kapasitas LLM. |
| B3 riwayat | Seluruh 16 latent teramati sampai t, termasuk current sekali; projector dilatih khusus B3. | Kondisi tanpa token masa depan. |
| B3P persistensi | Input B3 ditambah delapan salinan latent terakhir sebagai prediksi sederhana; projector dilatih khusus B3P. | Pembanding utama yang menyamakan 24 token input dengan B4. |
| B4 forecast | Input B3 ditambah delapan keluaran Dynamics beku dari 16 latent yang sama; projector dilatih khusus B4. | Mengukur nilai transformasi prediktif di atas persistensi. |
| B5 observasi masa depan | Input B3 ditambah latent encoder dari radar t+1 sampai t+8; projector terpisah jika dijalankan. | Diagnostik nilai observasi masa depan, bukan pipeline deployment. |
| Diagnostik GT terstruktur | Field GT valid diberi sebagai teks, tanpa projector. | Memastikan soal, format jawaban, dan penilai konsisten. |

B3, B3P, dan B4 menggunakan split, panel, target, prompt, SLM, ukuran projector, seed yang dilaporkan, dan anggaran optimasi sepadan. **B4–B3P adalah perbandingan primer** karena kedua input memuat 16 latent riwayat dan delapan slot masa depan; B4–B3 menilai manfaat format dengan token prediksi dibanding riwayat saja. Semua selisih dihitung pada sampel test identik. Model B4 harus dilatih menggunakan forecast Dynamics yang benar, bukan latent observasi masa depan; pergantian input hanya saat evaluasi adalah diagnosis pergeseran distribusi.

Kontrol pertukaran sensor adalah analisis sekunder. Simpan pasangan donor dan target fisiknya, paksa aksi/subjek donor sesuai definisi kontrol, lalu periksa apakah jawaban bergerak menuju atribut donor yang diketahui. Pertukaran seluruh rekaman tidak membuktikan counterfactual kausal yang mengubah satu variabel saja.

## 4. Urutan implementasi dan gerbang keputusan

### Tahap 0 — Audit artefak dan integritas data

Pulihkan checkpoint Av2, Dynamics, statistik normalisasi train, projector, fitur, QA JSONL, config aktual, seed, dan raw output Batch 1. Buat manifest hash encoder/preprocessing untuk setiap fitur; ekstraksi parsial tidak boleh mencampur cache lama dan baru ([skip cache](../eksperimen_model/datasets/extract_physical_features.py#L314)). Audit frame ID, kesesuaian GT, frame kosong, sumber fallback, distribusi kelas, serta batas repetisi per split. Karantina rentang CSV terbalik dan window yang melintasinya.

**Lulus jika:** artefak yang akan dipakai memiliki provenance yang cocok, GT dan indeks frame valid, serta jumlah sampel per subjek/aksi/kelas dan jendela lintas repetisi tercatat. Jika artefak lama tidak tersedia, angka Batch 1 tetap berstatus tidak dapat direproduksi dari checkout ini. Loader dan evaluator harus berhenti bila checkpoint yang diwajibkan hilang.

### Tahap 1 — Uji kelayakan representasi dan Dynamics

Pada train/val, jalankan probe sederhana untuk target pilot dari z_t, bandingkan dengan B0, dan periksa kegagalan per subjek/aksi. Perbaiki B2 agar label yang tidak diminta tidak menjadi target palsu ([probe saat ini](../eksperimen_model/train_probe.py#L55)). Untuk Dynamics, gunakan statistik normalisasi Tahap 3 yang sama, bandingkan prediksi latent dengan persistensi z_t dan baseline temporal sederhana pada horizon yang sama. Metrik pose hanya memakai decoder global-latent yang sama untuk kedua kondisi.

**Lulus jika:** target pilot dapat didefinisikan dan diprediksi dari input yang sah pada val, serta target masa depan memiliki ruang perbaikan di atas baseline persistensi. Bila gagal, pilih target lain dari train/val atau perbaiki representasi/temporal lebih dulu; jangan menilai projector dari tugas yang inputnya tidak mendukung.

### Tahap 2 — Bangun manifest QA dan input temporal

Buat sampel dari file, frame ID, GT, dan penanda batas repetisi yang tervalidasi. Untuk jendela 16+8, catat apakah horizon melewati repetisi; laporkan performa dalam dan lintas batas secara terpisah. Simpan target terstruktur lengkap satu kali per jendela dan bentuk QA sesuai skema. Subjek train, val, test harus terpisah. Berikan 16 latent sampai t kepada B3/B3P/B4, tanpa duplikasi current. Bentuk B3P dari latent t yang diulang, dan B4 dari checkpoint Dynamics dengan statistik train. B5 saja yang boleh membaca radar setelah t. Panjang token fisik kini berbeda antar kondisi; dataset harus menghitung offset label dan attention mask dari panjang aktual, bukan konstanta 13.

**Lulus jika:** setiap field jawaban sesuai pertanyaan dan GT; seluruh jendela memiliki frame berurutan serta horizon fisik yang benar; mengubah radar setelah t tidak mengubah input B4; B3/B3P/B4 membaca 16 riwayat yang sama; panjang label cocok dengan jumlah token; dan manifest test dibekukan sebelum pemilihan model.

### Tahap 3 — Latih projector pembanding

Latih B3, B3P, dan B4 dengan parameter SLM tetap beku dan anggaran yang sepadan. Jawaban training berasal dari JSON target tunggal. Gunakan pembentukan prompt serta tokenisasi identik pada train dan evaluasi; saat ini dataset mematikan special tokens sementara evaluator memakai default ([dataset](../eksperimen_model/datasets/grounded_qa_dataset.py#L123), [evaluator](../eksperimen_model/evaluate_reasoning.py#L164)). Perbaiki ukuran grup terakhir gradient accumulation dan hitungan langkah scheduler. Pilih checkpoint dengan metrik jawaban terstruktur pada val serta cakupan parse; simpan CE sebagai diagnostik. Set seed dan simpan hash model/data. Dtype dan device dipilih eksplisit; CPU memakai float32 untuk pemeriksaan lokal. Bila hanya satu seed terjangkau, batasi inferensi pada checkpoint itu; replikasi seed diperlukan sebelum mengklaim kestabilan hasil training.

**Lulus jika:** checkpoint memuat provenance lengkap, hanya projector menerima gradien, input B4 benar-benar forecast, dan kondisi B3/B3P/B4 telah dilatih dengan protokol yang dapat dibandingkan.

### Tahap 4 — Evaluasi beku dan pelaporan

Jalankan B0, B1, B2, B3, B3P, dan B4 pada panel test yang sama; B5 boleh menyusul sebagai diagnostik. Parse JSON secara ketat dan simpan respons mentah, status parse, target, prediksi, sample ID, checkpoint, serta segmen. Laporkan jumlah total/valid/terparse per task, accuracy dan macro-F1 untuk kelas, MAE/RMSE untuk besaran kontinu yang memang dipertahankan, serta distribusi kelas. Gagal parse tidak menjadi MAE nol. Metrik primer ditetapkan sebelum test, misalnya **selisih macro-F1 target perubahan masa depan B4–B3P**. Ambang efek minimum yang bermakna dan cakupan parse minimum ditentukan dari train/val sebelum melihat test. Hitung selisih berpasangan dan interval dengan pengelompokan pada rekaman atau subjek; tampilkan hasil tiap subjek dan strata batas repetisi. Jangan menyimpulkan robustness lingkungan dari split yang hanya menahan subjek.

**Lulus jika:** evaluator memberi skor benar pada contoh jawaban sah dan gagal format, semua kondisi memakai sample ID identik, denominator terlihat, dan tabel/kesimpulan dihasilkan dari angka yang benar-benar dihitung.

### Tahap 5 — Putuskan arah Batch 2

- GT terstruktur gagal: revisi rumus target, soal, atau format evaluasi.
- Probe z_t gagal untuk target valid: perbaiki representasi/ekstraksi Tahap 2 atau sempitkan task. Menambah epoch projector tidak mengatasi informasi yang tidak tersedia.
- Dynamics tidak mengungguli persistensi pada target yang relevan: revisi Tahap 3 sebelum menguji manfaat forecast pada SLM.
- Probe dan Dynamics layak, tetapi B4 tetap gagal: iterasi projector, target training, atau prompt Tahap 4.
- B4 lebih baik daripada B3P pada target primer dengan cakupan parse memadai dan selisih berpasangan yang dilaporkan: klaim bahwa transformasi prediktif membantu task, split, dan horizon yang diuji. Ini tidak berarti Dynamics menciptakan informasi sensor baru.

## 5. Implementasi minimum dan artefak

Prioritas file: generator QA, dataset QA, train_projector, evaluate_reasoning, train_probe, evaluasi Dynamics, serta konfigurasi yang terkait. Ubah ekstraksi fitur atau encoder hanya bila jalur metrik absolut dipilih. Gunakan kode parser/loader yang ada dan pemeriksaan kontrak kecil yang langsung menguji sumber frame, validitas target, serta kesamaan prompt; hindari kerangka evaluasi baru yang tidak diperlukan.

Komponen antarmuka dapat dikembangkan di mesin ini memakai fixture sintetis kecil. Training dan keputusan empiris membutuhkan fitur, GT, checkpoint, serta GPU pada mesin eksperimen. Pertahankan seluruh laporan Batch 1; hasil pemulihan masuk folder run baru dengan manifest, raw output, metrik, config, dan checksum. Dokumen Batch 1 perlu diberi status **tidak konklusif untuk klaim kontribusi Dynamics dan sensor-grounded reasoning** sampai evaluasi yang sah tersedia.

Implementasi kode saat ini mencakup ekstraksi dengan provenance, QA dua tugas relatif, loader B3/B3P/B4/B5, probe B2, lineage checkpoint Dynamics, training projector per kondisi, evaluasi berpasangan, dan panduan eksekusi. Checkpoint Dynamics Batch 1 tidak memiliki hash fitur/statistik Tahap 3; jalur B4 yang sudah didukung memerlukan checkpoint baru dengan lineage setelah pelatihan ulang Dynamics pada fitur recovery. Audit migrasi checkpoint lama belum diimplementasikan. Pemeriksaan sintaks lokal tidak menggantikan sanity check serta pelatihan di mesin eksperimen.
