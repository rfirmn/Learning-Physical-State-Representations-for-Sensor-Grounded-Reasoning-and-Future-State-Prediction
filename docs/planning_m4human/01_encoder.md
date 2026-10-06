# 01 — Fondasi Sensor Bersama: Audit Data dan Encoder M4Human

Versi kontrak: `m4human_kinetok_v3`. Revisi: 6 Oktober 2026. Status: protokol dengan implementasi lokal dan audit pending pada data nyata; bukan hasil eksperimen. Acuan ilmiah: [proposal riset terbaru](../proposal_riset_terbaru.md). Processed M4Human berada pada perangkat eksperimen lain; `/dataset` adalah root rencana yang belum diperiksa dari checkout ini. Target resource ialah satu RTX 3060 12 GB dan RAM 16 GB; kelayakan, runtime, dan peak memory masih harus diukur.

Dokumen ini menetapkan fondasi persepsi yang sama bagi seluruh kondisi: data sensor, target posisi, audit, encoder, dan cache. [02 — Model gerak dan tokenizer](02_dynamic_model.md) menetapkan backbone gerak bersama dan kompresor terlatih; [03 — Decoder](03_decoder.md) menetapkan readout fisik, rekonstruksi latent, dan recipe target; [04 — LLM layer](04_llm_layer.md) menetapkan alignment projector-only; [05 — Evaluasi dan testing](05_evaluasi_testing.md) menetapkan pembuktian pelestarian dan pemanfaatan informasi. Nama model merujuk implementasi lokal; kesiapan ilmiahnya tetap bergantung pada audit dan gate nyata.

Inti riset bukan memilih encoder terbaik atau membuktikan forecasting. Intinya adalah mengembangkan dan menguji tokenizer yang menjaga informasi kinematik terpilih pada token akhir yang ringkas. Encoder menyediakan titik awal yang dapat dipercaya; kualitas encoder tidak otomatis membuktikan kualitas token akhir.

## 1. Scope, status bukti, dan keputusan minimum

### 1.1 Batas ilmiah

Encoder mengestimasi keadaan tubuh yang diamati dari radar, lalu menyediakan predicted state dan fitur bagi backbone gerak bersama. Encoder disupervisi posisi; backbone gerak disupervisi posisi dan velocity. Setelah keduanya dibekukan, penelitian membandingkan dua kompresor berarsitektur sama: baseline `C_base` dan metode usulan `C_kin`. Hipotesisnya adalah bahwa supervisi kinematik yang ditempatkan pada token akhir U dapat meningkatkan pelestarian informasi dibanding supervisi sebelum kompresi, bukan bahwa kompresi pasti merusak informasi.

Matematika menyediakan target turunan, baseline, dan pemeriksaan konsistensi. Kinematika di sini berarti keadaan serta perubahan gerak yang terukur, bukan seluruh dinamika Newton. Root/pelvis bukan center of mass; gaya, torsi, kontak terukur, diagnosis keseimbangan, dan mekanika penuh berada di luar proof of concept. Forecasting dan HAR bukan objective utama. Acceleration, sudut, orientasi, fase, dan generalisasi lintas dataset merupakan perluasan bersyarat, bukan syarat P0.

F/R: output-side kinematic reconstruction bukan gagasan pertama: SeMoCo sudah mendekode quantized tokens dengan supervisi posisi/velocity/acceleration. MoTok juga mempunyai motion-to-text captioner, sehingga token gerak ke bahasa bukan pembeda baru. Kandidat kontribusi penelitian ini adalah kombinasi konteks radar M4Human, continuous compact tokens, paired protocol pelestarian informasi, serta pemanfaatannya oleh frozen LLM; kebaruan tidak ditetapkan hanya oleh lokasi loss. [SeMoCo, §3.2](https://arxiv.org/html/2608.24334v2), [MoTok, Appendix D.3](https://arxiv.org/html/2603.19227v1).

### 1.2 Cara membaca keputusan

| Penanda | Arti |
|---|---|
| F — fakta | Dukungan langsung sumber primer; bukan bukti paket perangkat lain identik |
| T — teori/derivasi | Hubungan matematis dengan syarat yang dinyatakan |
| R — desain beralasan | Pilihan untuk scope dan batas operasional proyek ini |
| H — hipotesis/default | Angka atau mekanisme awal yang belum diuji pada M4Human |

Spesifikasi implementasi berikut berstatus R/H kecuali dinyatakan F/T. Nilai empat frame, 512 titik, 22 joint, dan 256 fitur adalah kontrak awal, bukan optimum empiris. Perubahan setelah pilot disimpan sebagai konfigurasi baru beserta alasan validation, bukan disebut hasil tuning yang sudah terjadi.

### 1.3 P0 yang diperlukan

P0 menggunakan satu encoder `M4HumanSetEncoderV2` scratch, satu backbone `CausalDSTformerLiteV1` scratch beserta `KinematicReadoutV2`, dan dua instans `KinematicTokenLearnerV1` scratch dengan initialization sama. Encoder memakai set per frame dan empat-frame context kronologis. Tidak perlu sweep beberapa encoder untuk menguji hipotesis kompresi.

Tahapan yang dikunci:

| Tahap | Yang dipelajari | Setelah validation |
|---|---|---|
| E — persepsi | Encoder dan head posisi/root dari RPC | Freeze satu encoder untuk seluruh kondisi |
| M — gerak bersama | Backbone gerak dan full-H physical readout | Freeze satu backbone; cache H/M/Z yang sama |
| C — kompresi | `C_base` dan `C_kin`, fidelity decoder serta auxiliary head masing-masing | Freeze kompresor dan seluruh head fisik |
| L — bahasa | Projector independen tiap kondisi | Encoder, backbone, kompresor, head, dan LLM tetap frozen |

Pada tahap C, kedua kondisi menerima memori joint–waktu yang sama, merekonstruksi full frozen H, serta mempunyai arsitektur/head, target GT, split dan budget update yang sama. Auxiliary kinematic head membaca M pada `C_base` dan U pada `C_kin`. Karena M tetap, auxiliary baseline memperbarui head saja; pada metode usulan gradien juga mencapai kompresor. Ini treatment lokasi supervisi yang disengaja, bukan klaim gradien kedua kondisi identik. Baseline tetap kompresor terlatih melalui fidelity full-H, bukan random tokenizer.

Token akhir ialah satu continuous token per bin kronologis: U[B,K,256], K16 primer; K8/K32 adalah budget sekunder yang ditetapkan sebelum test. Tidak ada slot duplikat atau bypass f_enc. P0 mencakup probe sebelum/sesudah kompresi, QA temporal dan relasi tubuh, kontrol text-only/majority/direct-rule, serta donor berbeda recording dengan aktivitas sama dan reference answer berbeda. Pertanyaan/body_part/support waktunya kompatibel; donor labels dihitung sah, bukan memakai label penerima. Laporkan both-correct, correct-change dan coverage menurut03/05. Donor dengan label sama hanya kontrol invariansi. Task order ditambahkan hanya setelah audit label; tanpa itu klaim urutan peristiwa tidak dibuat.

| ID | Keputusan dan alasan | Bukti yang dapat mengubahnya |
|---|---|---|
| ENC-DATA | R: satu processed M4Human, adapter RPC-only, GT terpisah | Input privileged, schema berbeda, atau label tidak dapat disejajarkan |
| ENC-SET | T/R/H: shared point MLP dan pooling simetris menghindari ketergantungan pada urutan titik | Error joint lokal menunjukkan bottleneck pooling di train/val |
| ENC-TIME | R/H: trailing context pendek; concatenation kronologis mempertahankan urutan | Konteks tidak cukup/tidak membantu pada validation dan profil resource |
| ENC-STATE | T/R: pose relatif dan root global mempertahankan artikulasi serta perpindahan | Audit pelvis/koordinat gagal; target dibetulkan sebelum tuning |
| ENC-FEATURE | R/H: fitur sensor dapat menyimpan informasi di luar pose readout | Probe fitur/task menunjukkan informasi hilang atau tidak berguna |
| ENC-FREEZE | R: checkpoint tetap menjaga kualitas persepsi dan provenance cache konsisten | Perubahan encoder menjadi run/cache baru untuk semua kondisi |

## 2. Fakta sumber dan gerbang audit paket

F: paper M4Human melaporkan sekitar 661 ribu frame, 999 sekuens, 20 subjek, dan 50 aktivitas. Sensor radar/RGB-D sekitar 12 Hz disejajarkan dengan MoCap 100 Hz melalui pencocokan frame. Ini statistik paper, bukan inventaris paket perangkat eksperimen; MoCap 100 Hz tidak membuktikan trajectory aslinya ikut tersedia dalam processed release. [M4Human, statistik dan sinkronisasi](https://arxiv.org/html/2512.12378v3).

F: repository resmi menyebut processed radar sekitar 50 GB berupa LMDB dan merujuk asset SMPL-X. Ukuran disk bukan jumlah sampel independen, kebutuhan RAM/VRAM, atau jaminan model berhasil. Frame/window berdekatan berkorelasi. [M4Human, distribusi processed](https://github.com/FanJunqiao/M4Human).

Sumber primer digunakan untuk memahami distribusi, bukan untuk mengesahkan paket pengguna. Loader yang ditautkan memakai snapshot `e01b8fed6132306c3553ed6680be912c21594dce`; beberapa utility memakai `main`. Sebelum implementasi, simpan commit penuh dan hash source yang benar-benar dijadikan acuan. Jangan menyatakan semua utility berasal dari snapshot sama atau paket cocok tanpa pemeriksaan.

| Hal belum diketahui | Pemeriksaan penutup gate | Jika belum lulus |
|---|---|---|
| Lokasi/kelengkapan LMDB | Root aktual, tipe file/direktori, entry count, provenance distribusi | Hentikan pembacaan/training; jangan membuat database kosong |
| RPC/channel/sentinel | Decode contoh dan scan distribusi secara streaming | Jangan mempertukarkan intensity, Doppler, SNR |
| Recording/frame/timestamp | Cocokkan key, indicator, metadata, gap/duplikasi dan batas take | Jangan mengklaim derivative/time support tervalidasi |
| Kalibrasi/unit/origin | Round-trip, trajectory, pemeriksaan joint terhadap RPC | Jangan melatih root pada label dengan frame tidak jelas |
| Body model/joint/pelvis | Asset/regressor/map, gender convention, pelvis rekonstruksi | Jangan menyamakan `trans` dengan pelvis |
| Asal preprocessing RPC | Audit `RawPoints`/crop terhadap GT box, RGB, atau target tubuh | Ungkapkan privilege dan batasi klaim deployment |
| Cakupan target/task | Count root/joint/task/subject setelah mask/support | Pilih panel kecil reliable atau laporkan hasil terbatas |
| Resource | Sanity dan profil I/O/CPU/RAM/GPU di perangkat tujuan | Batch/epoch/runtime tetap H |

Audit kelengkapan sumber dapat mencakup semua split tanpa menggunakan performa test. Pemilihan model, threshold, dan kriteria efek bermakna hanya pada train/validation; test dikunci untuk evaluasi final.

## 3. Layout sumber, reader praktis, dan serializer

### 3.1 Root dan artefak yang diperlukan

`lmdb_dir` menunjuk folder aktual yang berisi database, misalnya `/dataset` atau subfoldernya setelah inventaris. Ini bukan instruksi memindahkan dataset. Pada Windows, gunakan path absolut dengan drive yang benar; jangan menerjemahkan `/dataset` menjadi drive yang ditebak.

| Artefak kandidat | Pemakaian adapter penelitian |
|---|---|
| `radar_pc.lmdb` | Wajib untuk input RPC |
| `params.lmdb` | Ekspor GT offline; tidak dibuka oleh inference sensor |
| `calib.lmdb` | Audit/ekspor GT; runtime hanya jika transform sensor statis memerlukannya |
| `indicator.lmdb` | Membantu index/recording; inference memakai manifest sensor yang diaudit |
| `indeces.pkl.gz` | Index/split opsional dengan provenance tepercaya; alternatif manifest custom |
| `radar_comp.lmdb` | RT; adapter RPC-only tidak membukanya |
| `rgb_feat.lmdb`, `image.lmdb` | Tidak menjadi input radar-only |
| `bbox.lmdb` | Tidak diperlukan; asalnya diaudit sebelum eksperimen lain menggunakannya |
| `*.lmdb-lock` | Koordinasi akses, bukan label/jumlah frame |

F: loader stok menyusun environment RT/RPC/params/calib/indicator, memuat body model, dan tetap memanggil pembaca RT pada `__getitem__` RPC. Constructor memiliki jalur pembuatan cache/index. Pada snapshot ini bbox dihitung dari parameter; `bbox.lmdb`/`rgb_feat.lmdb` tidak termasuk daftar environment. Memilih modality RPC saja belum menghasilkan reader RPC-only. [Loader resmi pada snapshot](https://github.com/FanJunqiao/M4Human/blob/e01b8fed6132306c3553ed6680be912c21594dce/dataset/m4human_dataset.py).

R: adapter custom menerima processed LMDB yang sudah ada, membaca RPC/metadata sensor seperlunya, dan menolak sumber tidak lengkap tanpa `process_and_cache`. Trainer memasangkan hasilnya dengan target offline. Inference tidak membutuhkan params, GT bbox, RGB, gender, betas, action ID, atau SMPL-X. Asal crop yang sudah berada dalam RPC release tetap audit terpisah; adapter bebas GT tidak otomatis membuktikan preprocessing upstream bebas GT.

### 3.2 Akses read-only

Buka environment lazy per proses worker: `readonly=True`, `create=False`, `writemap=False`, dan `subdir` sesuai inventaris. Transaksi read-only. Jangan mewariskan handle hidup melalui Windows spawn; buka ulang berdasarkan PID, dan jangan menyimpan transaction/cursor lintas batch. Copy array sebelum transformasi mutabel atau masa berlaku buffer transaction berakhir.

F: `readonly=True` masih dapat mengubah lock file; `lock=False` mengalihkan tanggung jawab koordinasi ke pemanggil. [Dokumentasi py-lmdb, Environment](https://lmdb.readthedocs.io/en/release/).

R: sumber reference adalah paket immutable tanpa concurrent writer. Hanya setelah kondisi itu dibuktikan gunakan `lock=False` agar source termasuk lock tidak ditulis. Jika writer aktif, hentikan sesi ini dan jadwalkan snapshot immutable/locking yang benar; jangan menonaktifkan koordinasi untuk melewati izin. Audit tidak menulis, mengompaksi, merepack, atau menghapus database sumber.

Mulai `num_workers=0`; coba 2 setelah reader/spawn lulus. Readahead, pin memory, prefetch dan persistent workers ditentukan profil disk/RAM. Memory-map bukan preload 50 GB. Manifest dibuat streaming; jangan menyimpan seluruh radar dan jutaan dictionary sample di RAM.

### 3.3 Decode aman dan trust boundary

F: serializer RPC resmi menggunakan header `struct` jumlah dimensi/shape dengan prefix `=` dan payload float32 contiguous. Params/calib menggunakan MessagePack dengan array bertag `__nd__`, `dtype`, `shape`, `data`; indicator menggunakan MessagePack. Native byte order format `=` membuat endianness host pembuat paket perlu dicatat. [Serializer resmi](https://github.com/FanJunqiao/M4Human/blob/main/dataset/lmdb_utils.py).

R: validasi sebelum alokasi/reshape:

1. Batasi key/value/header menurut `schema.json` yang diaudit. Reject missing value, truncation, dimensi tidak sah, overflow produk shape, dan payload berlebih/kurang. Batas titik/byte aktual masih gate audit.
2. RPC kontrak utama wajib `ndim=2`, shape `n×4`, dan payload tepat `n×4×4` byte. Channel order harus terbukti. Schema lain menghentikan run atau membuat kontrak baru, bukan silent slicing.
3. MessagePack dibatasi kedalaman/ukuran container/bin dan dtype numerik. Tolak object dtype, arbitrary extension, field tak dikenal yang mengubah semantics, atau ukuran array tidak cocok. Key string diparse aman tanpa `eval`.
4. Indicator/key divalidasi sebagai tuple integer setelah parse bounded; cocokkan manifest. Jangan menganggap base frame 0/1 dari MM-Fi atau sort byte-key sebagai urutan waktu.
5. Struktur rusak menolak frame. Titik non-finite disaring dengan jumlah/sebab tercatat, bukan dijadikan nol valid. Sentinel hanya dihapus setelah konvensinya dibuktikan.

F: unpickle dapat mengeksekusi kode; gzip tidak menghilangkan sifat tersebut. [Dokumentasi Python pickle](https://docs.python.org/3/library/pickle.html).

R: index `indeces.pkl.gz`, body asset, dan checkpoint berbasis pickle hanya dibaca bila asal distribusi/integritas/trust sudah diverifikasi. Hash sendiri memberi identitas, bukan bukti keamanan. Batasi hasil dekompresi; bila provenance index tidak jelas, jangan unpickle: buat manifest key/metadata sah dan sebut split custom. Import index tepercaya diekspor sebagai JSON/JSONL tervalidasi di derived root. Target/cache NumPy memakai `allow_pickle=False`. Checkpoint inference berupa tensor state dict dan metadata sederhana, dimuat `weights_only=True` jika versi PyTorch mendukung lalu `strict=True`; tidak fallback otomatis ke unrestricted pickle. [Dokumentasi torch.load](https://docs.pytorch.org/docs/main/generated/torch.load.html).

## 4. Index, recording, split, dan waktu

### 4.1 Manifest tanpa mengubah sumber

F: preprocessing menyimpan key `str([subject, action, frame])`, RPC dari `RawPoints`, dan params `joints`, `betas`, `pose_body`, `trans`, `root_orient`, `gender`. Helper temporal dapat mengulang frame ketika frame yang diharapkan tidak tersedia. [Preprocessing dan helper temporal](https://github.com/FanJunqiao/M4Human/blob/main/dataset/m4human_utils.py).

R: satu baris manifest per source frame, bukan per window:

```text
frame_uid, source_key, subject_id, action_id, recording_id, segment_id
source_frame_id, source_time_s|null, time_s, time_source
rpc_present, schema_valid, finite_point_count, sensor_frame_valid
target_present, annotation_root_valid, annotation_joint_valid[22]
source_manifest_version, coordinate_version, joint_map_version
```

`frame_uid` unik menurut package/recording/frame; `source_key` mempertahankan encoding yang dapat dibalik. Jangan menyatukan take hanya karena subject/action sama. Jika package tidak mengungkap take, audit reset/gap dan catat keterbatasan recording ID. Subject/action/recording hanya untuk grouping/split/join/laporan, tidak masuk neural input, pseudo-token, prompt, atau kategori jawaban.

Scan contoh dahulu, lalu coverage penuh streaming. Catat missing key per database, duplicate ID/time, empty RPC, order, NaN/Inf, calibration mismatch, dan label hilang. Missing annotation tidak otomatis menghilangkan radar context; target mask hanya memengaruhi supervisi. `rejections.jsonl` menyimpan reason dan fase penolakan.

### 4.2 Held-out subjects

F: index upstream mempunyai schema scale/split/train–val–test; konfigurasi berasal YAML dan index disimpan melalui gzip/pickle. [Pengelolaan split resmi](https://github.com/FanJunqiao/M4Human/blob/main/dataset/dataset_config_clean.py).

R: membership subject ditetapkan sekali setelah ID aktual diketahui. Jika index resmi tepercaya digunakan, audit subject disjoint dan keutuhan recording/context, kemudian catat subset actual. Jika tidak, buat split custom tetap; contoh H bagi 20 subjek lengkap ialah 14/2/4, tetapi daftar ID belum ditetapkan. Jangan menamainya S2 resmi tanpa protocol/index yang sesuai.

Semua recording, empat-frame context, derivative support, dan motion window satu subjek berada dalam satu split. Split sebelum window/statistik. Pilot memakai recording contiguous train/val; test tertutup. Sampling frame lintas dataset bukan bukti unseen-subject. Scale upstream tidak sama dengan kelompok aktivitas paper.

Normalizer hanya train; checkpoint, label recipe, QA threshold, gate akurasi, dan kriteria efek praktis hanya train/val dan dikunci sebelum test. Cross-action perluasan mensyaratkan encoder juga belum pernah dilatih pada action test. Laporkan subject, recording, segmen, frame eligible, window aktual, bukan menyebut window sebagai sampel independen.

### 4.3 Timestamp, gap, dan causal support

Utamakan timestamp sensor asli dalam detik. Simpan float64 untuk manifest/subtraction; cast relative time ke fp32 dalam model. `time_source` ialah `measured` atau `nominal_frame_index`. Tanpa timestamp, H memakai `(frame_id−frame_id_start)/12` hanya setelah audit membuktikan ID mewakili urutan/grid sensor. Jangan merapatkan gap dengan penomoran ulang. Jika hubungan ID/waktu tidak terbukti, gate derivative gagal.

Duplicate/non-monotonic timestamp, sensor frame hilang, reset recording/coordinate policy, atau interval di luar gap gate memutus segmen. Pada grid nominal wajib selisih ID satu. Batas interval/jitter measured ditentukan audit metadata/hardware, bukan plot test. Tidak ada interpolasi, repeat frame, atau future sample untuk menutupi gap pada kondisi utama.

Anchor t memakai empat observasi asli `[t−3,t−2,t−1,t]` dari segmen sama. Tiga frame pertama segmen belum menghasilkan encoder state P0. Reference mewajibkan context lengkap dengan cloud nonempty; partial-context memerlukan config baru. Kasus invalid tetap ditangani numerik API, tetapi tidak dihitung sebagai observasi scientific sah.

T: context empat frame 12 Hz memiliki endpoint span `3/12=0,25 s`. Motion window 32 encoder states memiliki span `31/12≈2,58 s`; union RPC support dapat mencakup 35 frame karena tiga pendahulu encoder. Untuk segmen `F_seg` frame sensor sah dan stride s:

```text
F_anchor = max(0, F_seg - 3)
W_seg = 0                              jika F_anchor < 32
W_seg = 1 + floor((F_anchor - 32) / s)   selain itu
```

R: sesuai 02, derivative GT dan mathematical comparator memakai support window-local yang sama: enam state pertama setiap L-window invalid meskipun cache recording mempunyai history sebelumnya. Support endpoint i adalah `[i−6,...,i]`, tidak sesudah i. Stride motion H 8/16/16 train/val/test mengikuti 02. Dukungan pre-window tambahan memerlukan perubahan bersama semua kondisi dan versi recipe baru.

## 5. Koordinat, ekspor GT offline, dan body map

### 5.1 Frame fisik/kalibrasi

R: target/output fisik memakai satu frame radar tetap dalam meter, tanpa normalisasi per-frame. `coordinate_audit.json` menyimpan frame, sumbu/handedness, origin radar, unit setiap field, arah transform, offset source, dan bukti pemeriksaan.

T: setelah translasi masing-masing dikonversi ke meter, transform column-vector adalah:

```text
q_cam_m   = R_cam_from_vicon @ q_vicon_m + t_cam_from_vicon_m
q_radar_m = inverse(R_cam_from_radar) @ (q_cam_m - t_cam_from_radar_m)
```

Bentuk transform didukung [kalibrasi M4Human](https://arxiv.org/html/2512.12378v3); unit numerik field paket tetap audit. Jangan membagi kedua translation vector dengan 1000 karena salah satunya millimeter. Uji finite rotation, orthogonality/determinant +1 sesuai tolerance, inverse round-trip, dan frame yang sama untuk RPC/label.

F: jalur mesh resmi menghasilkan vertices millimeter, mentransformasinya, lalu mengambil 22 row joint regressor dan membagi 1000. Fungsi kalibrasi parameter mengubah `trans`/`root_orient`; transform field `joints` di dalamnya dikomentari. `parameter.joints` tidak otomatis GT radar. [Kalibrasi dan joint readout resmi](https://github.com/FanJunqiao/M4Human/blob/main/dataset/data_loader_camera_calibration.py).

R: exporter reference merekonstruksi mesh dari params asli di frame yang diaudit, mengonversi unit eksplisit, mentransform ke radar tepat sekali, lalu mengambil regressor/map yang tervalidasi. Joint langsung alternatif hanya jika frame/unit/topology/equivalence terbukti. Jangan kalibrasi params lalu transform vertices kedua kali. Overlay RPC/joint dapat memeriksa alignment tanpa RGB input training.

F: `z−1.5` diterapkan di loader stok. [Lokasi offset](https://github.com/FanJunqiao/M4Human/blob/e01b8fed6132306c3553ed6680be912c21594dce/dataset/m4human_dataset.py).

R: adapter LMDB langsung tidak otomatis mengurangi/menambah 1.5. Audit apakah package pernah menyimpan offset. Reference mempertahankan physical radar frame sebelum offset neural; shifted config harus mencatat inverse transform dan origin radar ikut bergeser. Root radial direction terhadap origin konsisten, bukan nol yang salah di frame shifted. Sumbu vertikal harus lulus sebelum task naik/turun ditambahkan.

### 5.2 Ekspor target/pelvis

SMPL-X hanya offline untuk target bila diperlukan, dengan asset resmi yang sudah tersedia, versi/izin/provenance tercatat. Tidak ada download otomatis. Audit shape/unit `betas`, body pose, root orientation, translasi, gender convention, serta default tangan/wajah yang tidak disimpan. Input tidak lengkap menunda ekspor; jangan mengarang parameter atau menganggap seluruh subjek neutral. Trainer kemudian membaca array target; inference tidak mengimpor SMPL-X atau membuka asset body.

Exporter menghasilkan `joint_global_m[F,22,3]`, `root_m[F,3]`, `joint_relative_m[F,22,3]` fp32, serta `annotation_joint_valid[F,22]` dan `annotation_root_valid[F]`. Frame/key, provenance, recipe menyertai target. Raw params/labels immutable.

T: dengan pelvis index `j_root` hasil audit:

```text
r_gt[t]       = J_gt_global[t,j_root]
P_gt[t,j]     = J_gt_global[t,j] - r_gt[t]
J_global[t,j] = P_relative[t,j] + r[t]
P_relative[t,j_root] = 0
```

`trans` adalah parameter body-model translation, bukan otomatis pelvis. Audit selisih `trans−pelvis`. Root GT invalid membuat target relative yang memerlukan subtraction root invalid; target global joint lain masih dapat disimpan dengan mask masing-masing. Annotation mask tidak menentukan keberadaan joint sensor.

### 5.3 Unified body map

F: daftar nama SMPL-X diawali 22 body joint berikut. Ini referensi nama, bukan bukti asset/regressor package sudah cocok. [Nama joint SMPL-X resmi](https://github.com/vchoutas/smplx/blob/main/smplx/joint_names.py).

| Index referensi | Nama | Index referensi | Nama |
|---|---|---|---|
| 0 | pelvis | 11 | right_foot |
| 1 | left_hip | 12 | neck |
| 2 | right_hip | 13 | left_collar |
| 3 | spine1 | 14 | right_collar |
| 4 | left_knee | 15 | head |
| 5 | right_knee | 16 | left_shoulder |
| 6 | spine2 | 17 | right_shoulder |
| 7 | left_ankle | 18 | left_elbow |
| 8 | right_ankle | 19 | right_elbow |
| 9 | spine3 | 20 | left_wrist |
| 10 | left_foot | 21 | right_wrist |

`joint_map.json` berisi version/hash, ordered names, source row indices, pelvis index, parent/edge list dari asset, pasangan kiri/kanan, dan task groups. Jangan mengambil 22 row arbitrer dari output SMPL-X yang mencakup tambahan tangan/wajah. Map ini juga menentukan joint token dan physical head order.

R/H: kelompok awal kiri/kanan lengan memakai elbow+wrist, kaki memakai knee+ankle, dengan nama lengkap dari body map. Task `relative_limb_motion` memakai velocity root-relative. Task opsional `limb_onset_order` membandingkan onset kelompok kiri/kanan dengan recipe yang ditetapkan pada 03; ia tidak menggunakan label aktivitas sebagai proxy urutan. Perubahan kelompok memperbarui recipe bersama 03/04/05.

## 6. Input validity dan target kinematik terpisah

### 6.1 Input inference

Input hanya RPC yang lulus schema, waktu dalam detik, point mask, dan dukungan sensor. `point_mask` menunjukkan return nyata setelah validasi, bukan nol padding. `sensor_frame_valid` berasal dari cloud/time/context. Reference tanpa uncertainty head: `joint_sensor_valid` broadcast frame validity ke 22 joint, bukan confidence/visibility anatomis terkalibrasi.

Annotation validity hanya untuk loss/target QA, bukan forward encoder/motion/packer/projector/prompt. Sensor sample dibuat sebelum target join. Mengganti GT pose/root, body params, action ID, dan annotation mask pada RPC/time yang sama harus tidak mengubah inference. Eligibility QA dari GT boleh memilih baris evaluasi tetapi tidak dikirim sebagai token jawaban. Label hilang tidak mengubah sensor context. Alias mask target exporter ke `annotation_position_valid_joint/root` pada dokumen 03 dipetakan eksplisit oleh TargetSample; keduanya tidak menjadi input sensor.

### 6.2 Recipe derivative dan task

R/H: recipe bersama `aligned12hz_endpoint_sg_v2`, grid12Hz/window7/degree2, endpoint pos6/use dot/delta1/12. Fit koordinat terhadap `tau=time−time_endpoint`: posisi fit beta0, velocity beta1, optional acceleration 2*beta2. Gunakan koefisien chronological endpoint; bukan centered default, future edge interpolation, atau pembagian dt dua kali. [SciPy savgol_coeffs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_coeffs.html).

Raw calibrated GT position tetap target posisi encoder/motion; filtered position diagnostic terpisah. Velocity bernama `velocity_12hz_filtered`, bukan pengukuran langsung. Timestamp nonuniform memakai least squares aktual offline fp64 dengan rank/conditioning audit; koefisien uniform tidak ditempel begitu saja. Gap tidak dijembatani. Interpolasi processed labels tidak menghasilkan MoCap100Hz baru.

Mask derivative per root/joint memerlukan tujuh support sah; enam state pertama setiap L-window invalid, bukan gerak nol. GT validity dari annotation/time; comparator validity dari sensor/time. Causal support/operator/unit/recipe sama; sumber mask dan coverage dilaporkan. Acceleration tidak otomatis utama meski fit degree2 menghasilkannya.

Task IDs dan enum berikut adalah referensi kontrak bersama 03/04/05; implementasi mengimpor satu registry milik dokumen 03, bukan menyalin tabel menjadi registry kedua:

| Task | Kelas |
|---|---|
| `root_radial_direction` | `approaching`, `receding`, `stationary`, `unknown` |
| `root_speed_trend` | `speeding_up`, `slowing_down`, `no_overall_trend`, `unknown` |
| `relative_limb_motion` | `faster_left`, `faster_right`, `tie`, `unknown` |
| `limb_onset_order` (opsional, gate terpisah) | `left_before_right`, `right_before_left`, `approximately_simultaneous`, `unknown` |

Threshold/range guard/onset confirmation dipilih train/val pada shared recipe, bukan encoder config atau sesudah test. `root_speed_trend` mengikuti tren keseluruhan; `no_overall_trend` berarti slope dalam deadband dengan support/fit sah, bukan jaminan speed konstan atau label `unknown`. `stationary` radial deadband, bukan seluruh tubuh diam; tie/simultan memerlukan support reliable. Evidence/QA memakai `kinematic_recipe_v3`, yang menunjuk derivative subrecipe `aligned12hz_endpoint_sg_v2`; simpan kedua ID/hash karena operator endpoint tidak berubah. Registry memetakan `target_status=defined|unknown|undefined` ke alias manifest `validity=valid|unknown|excluded` secara satu-ke-satu. Evidence numerik undefined memiliki status/reason sendiri, tidak otomatis label unknown atau exclusion. Model abstention pada target valid tidak mengubah eligibility. Panel P0: `root_speed_trend` dan `relative_limb_motion`. Tanpa label order yang reliable dan probe order yang sah, penelitian tidak mengklaim pelestarian urutan peristiwa.

## 7. API dan kontrak tensor m4human_kinetok_v3

Nama ringkas state ialah `P_enc/r_enc/f_enc`. Field unit-bearing untuk API hilir mengikuti02: `P_enc_relative_m/r_enc_m/f_enc`; mapping eksplisit pada window adapter, bukan dua array dengan semantics berbeda.

```text
RPCOnlyReader.read(frame_uid) -> raw_rpc[n,4], sensor_metadata
M4HumanCausalSensorDataset[index] -> SensorSample, provenance
EncoderTrainingDataset[index] -> SensorSample, TargetSample, provenance

M4HumanSetEncoderV2.forward(rpc_m, point_mask, time_s)
  rpc_m:     [B,4,512,4] fp32; XYZ meter, intensity sesuai schema
  point_mask:[B,4,512] bool
  time_s:    [B,4] float64 chronological seconds
  -> P_enc[B,22,3], r_enc[B,3], f_enc[B,256]
     sensor_valid[B], root_sensor_valid[B]
     joint_sensor_valid[B,22], feature_valid[B]

EncodedStateWindowDataset[index] -> sensor_state, targets terpisah
  P_enc_relative_m[L,22,3], r_enc_m[L,3], f_enc[L,256], time_s[L]
  sensor masks, frame_uids[L], context_source_frames[L,4]
```

Encoder menghasilkan satu anchor, bukan empat pose sekaligus. Cache satu output per anchor; collate/window menghasilkan `P_enc[B,L,22,3]`, `r_enc[B,L,3]`, `f_enc[B,L,256]` dengan seconds/masks. Jangan memberi seluruh 32 RPC frame ke satu forward untuk setiap anchor.

Kontrak hilir02: L32, 23 spatial tokens (22 joint + global), D128, empat joint–time stage/empat head. Jangan memakai K23 untuk jumlah joint sekaligus budget kompresi K. Backbone menghasilkan `H[B,L,23,128]`; memori kaya sebelum kompresi adalah `M[t,j]=concat(H[t,j],H[t,global])`, berukuran `[B,L,23,256]`. Ringkasan `Z[B,L,256]` ialah concat masked mean H_joint128 + H_global128. H/M mempertahankan sumbu joint/waktu; Z adalah diagnostic sekunder, bukan satu-satunya fidelity target.

`KinematicReadoutV2` memberi delta posisi3+velocity3 per joint/global head, dilatih bersama trunk. Residual predicted-state bypass ini hanya untuk full-H readout bersama pada tahap M. Keberhasilannya tidak membuktikan kinematika dapat dibaca dari U. Auxiliary readout tahap C dan probe U dilarang memakai bypass P_enc/r_enc, full H/M, GT, atau kategori jawaban.

Objective motion/readout mengikuti 02/03, berbeda dari loss encoder: normalized masked component MSE `L_position_joint+L_position_root+0.5*(L_velocity_joint+L_velocity_root)/2`; lambda acceleration/bone/consistency reference 0. Posisi target raw calibrated GT, velocity endpoint recipe. Dokumen 03 memiliki normalizer: `s_delta_joint/root` adalah RMS komponen residual GT−predicted encoder di train, floor 0,01 m; `s_v_joint/root` adalah RMS komponen velocity train, floor 0,001 m/s. Skala residual mengatur inverse output delta posisi; skala error posisi loss tetap 1 m dan disimpan terpisah. Tidak menaksir ulang skala pada test/resume atau memakai body parameters GT per sample saat inference. Motion/readout joint checkpoint kemudian frozen, bukan decoder terpisah pada latent lain.

Tahap C menerima cache H/M/Z yang tetap. `KinematicTokenLearnerV1` memakai learned query cross-attention yang dibatasi bin kronologis pada M, satu continuous token per bin: `U[B,K,256]`. Default K16; budget8/32 sekunder dipatok sebelum test. Kedua kondisi memiliki compressor initialization/architecture, sensor support, time encoding, dan fidelity decoder yang sama. Common fidelity merekonstruksi full H melalui time+joint queries; auxiliary physical head membaca M pada `C_base` versus U pada `C_kin`, dengan target/kapasitas/budget yang sama. Lokasi gradien supervisi adalah treatment, bukan penambahan data GT pada metode usulan. H/M/Z dibekukan; f_enc tidak menjadi shortcut menuju projector.

H adalah target rekonstruksi frozen, bukan trajectory GT. M pada cache ialah concat mentah (`M_raw`); sumber attention compressor, auxiliary baseline, dan probe precompression memakai view `M_attn` dengan PE waktu sensor yang ditambahkan tepat sekali oleh helper bersama dokumen 02. View/mask sumber dan statistik fidelity memakai common frame support yang sama; probe tidak memperoleh partial frames tambahan. U adalah tensor persis yang diberikan ke projector dan wajib diperiksa secara mandiri. Rekonstruksi decoder tahap C mencakup seluruh observed window secara retrospektif: query pada waktu awal boleh membaca seluruh U. Ini bukan estimasi online causal pada tiap waktu, walaupun encoder dan motion trunk causal. C_pool/C_state hanya diagnostic tambahan yang dinyatakan, bukan pasangan utama; C_history/C_dynamics lama tidak menjadi evidence core.

Hilir memakai Qwen/Qwen2.5-1.5B-Instruct pretrained fully frozen, hidden size dibaca config (reference1536); projector scratch `LN256→Linear512→GELU→Linear d_llm` terpisah per kondisi. Hanya projector diupdate, tanpa LoRA; autograd melalui frozen LLM tetap ada, tidak `no_grad` saat training projector. Ini kontrak integrasi, bukan klaim model/hasil tersedia.

## 8. Encoder reference: set per frame + chronological aggregation

### 8.1 Alasan dan batas

F: PointNet memproses point set dengan permutation invariance titik. [PointNet, CVPR2017](https://arxiv.org/abs/1612.00593). T: shared point function disusul max/mean valid-only simetris terhadap permutasi set yang sama. Time axis tidak invariant karena urutan gerak bermakna.

R/H: M4HumanSetEncoderV2 adalah adaptasi lokal dari scratch tanpa T-Net/FPS/kNN/mesh head; bukan reproduksi exact PointNet. Pooling global mungkin melewatkan geometri lokal, diuji lewat per-joint error/probe. Concatenation kronologis memberi short history tanpa point tracking. Tidak ada teori bahwa fitur256 atau context4 pasti cukup.

### 8.2 Preprocessing persis

1. Decode/validasi n×4, XYZ physical radar frame. Intensity bukan Doppler/SNR atau calibrated power tanpa bukti. Filter non-finite/sentinel sesuai audit tanpa GT crop; simpan n_raw/n_clean/n_selected.
2. Jika n_clean>512, ambil512 tanpa replacement. Canonical sort XYZ/intensity dahulu; train RNG lokal run seed/epoch/frame UID, validation/cache stable frame-UID+preprocessing-version hash. Hindari Python hash() yang berubah antarproses; simpan algoritma/seed derivation.
3. Jika n_clean<512, zero-pad dengan mask false, tanpa replacement. Empty cloud invalid; scientific reference context ditolak.
4. Statistik XYZ train streaming dari titik sah tanpa padding. Input network `(xyz−mu_train)/max(std_train,0.001 m)`, bukan centering per frame. Intensity H signed_log1p lalu train standardization dengan std floor1e−5. Target/output fisik meter langsung, tidak inverse-normalize output dua kali.
5. Pilot reference tanpa rotasi/translasi/scaling/jitter. Augmentasi tambahan harus konsisten pada seluruh context/target/origin, dengan recipe baru. Evaluation/cache tanpa augmentasi. Sensor bounds ditetapkan lewat audit, bukan clamp batas ruangan MM-Fi.

Sampling bisa kehilangan detail; laporkan retention/count distribution. Uji invariance pada set sama dan sampling end-to-end terpisah. Seed global saja tidak menjamin cache deterministic.

### 8.3 Exact forward

Semua Linear biased, LN feature-only eps1e−5, GELU approximate none, dropout0. Point stem shared empat frame. Invalid input disanitasi ke nol sebelum arithmetic; non-finite valid input error, bukan NaN×mask.

| Langkah | Operasi | Shape |
|---|---|---|
| Point stem | Linear4→64/LN/GELU → Linear64→128/LN/GELU → Linear128→256/LN/GELU | [B,4,512,256] |
| Set pool | Concat masked max dan masked mean sepanjang point axis | [B,4,512] |
| Frame feature | Linear512→256/LN/GELU | g[B,4,256] |
| Chronological input | Flatten g oldest→anchor + relative times4 + valid fractions4 | [B,1032] |
| Temporal aggregate | Linear1032→512/LN/GELU → Linear512→256/LN/GELU | f_enc[B,256] |
| Pose head | Linear256→128/GELU → Linear128→63 | [B,21,3] |
| Pelvis constraint | Scatter21 outputs ke non-pelvis map22, pelvis exact0 | P_enc[B,22,3] m |
| Root head | Linear256→128/GELU → Linear128→3 | r_enc[B,3] m |

Relative time `(time_s−anchor_time)/(1/12 s)` dihitung setelah subtraction float64 lalu cast fp32. Valid fraction `point_mask.sum()/512` berasal sensor. Mean membagi jumlah titik sah; max mengecualikan padding. Empty frame memberi zero pooled feature/validity false. Zero invalid feature setelah biased affine block; bias tidak membuat missing observation nyata. Full-invalid context mengembalikan zero outputs/all masks false tanpa NaN; scientific trainer/cache menolak, bukan menyebut tubuh diam.

Reference `sensor_valid` memerlukan empat frame bertitik sah, time strictly increasing, dan reader context policy lulus. `root_sensor_valid=feature_valid=sensor_valid`; `joint_sensor_valid` broadcast ke 22 joint. Output anchor yang tidak memenuhi sensor validity di-zero dengan mask false; guard ini bukan pengganti penolakan scientific context. Key/segment/gap checks di reader; forward mengecek shape, finite-valid input, mask, dan monotonisitas waktu. f_enc dari aggregate mendapat gradient kedua head, bukan random feature adapter yang tidak terlatih.

### 8.4 Initialization dan trainable map

H: Linear Xavier-uniform/bias0, LN weight1/bias0. Setelah initializer umum, override final Linear pose/root dengan normal std0.01/bias0. Jangan zero-init seluruh encoder. Tidak memuat checkpoint PointNet/ShapeNet/MM-Fi/P4/MotionBERT; reuse gagasan bukan transfer bobot.

| Fase | Parameter diupdate | Tetap/frozen |
|---|---|---|
| Encoder | Stem, frame feature, chronological aggregate, pose/root heads | Raw GT/split/normalizer/exporter yang dikunci |
| Motion/readout | Trunk dan readout bersama | Encoder eval, semua requires_grad false |
| Kompresi | Tokenlearner + fidelity/auxiliary heads, per kondisi | Encoder, motion/full-H readout, H/M/Z reference |
| Projector | Satu projector per trained condition | Encoder, motion/readout, kompresor/physical heads, seluruh LLM |

Encoder extraction boleh inference mode/detach. Ini tidak berlaku pada frozen LLM di projector training. Tidak ada gradient language upstream.

## 9. Loss, training, dan checkpoint selection

### 9.1 Encoder position loss

R/H: SmoothL1 beta0.05m, reduction none lalu masked component mean; penalti quadratic residual kecil dan linear residual besar. [Dokumentasi SmoothL1Loss](https://docs.pytorch.org/docs/main/generated/torch.nn.SmoothL1Loss.html).

```text
m_relative = annotation_joint_valid AND annotation_root_valid
             AND sensor_valid AND joint != pelvis
m_root     = annotation_root_valid AND sensor_valid
L_pose = mean_selected_xyz(SmoothL1(P_enc, P_gt_relative, beta=.05), m_relative)
L_root = mean_selected_xyz(SmoothL1(r_enc, r_gt, beta=.05), m_root)
L_enc  = L_pose + L_root
```

Select valid targets sebelum arithmetic/reduction, bukan NaN×0. Term tanpa valid target finite zero tanpa denominator palsu; batch tanpa supervisi di-skip/dicatat. Pelvis-zero tidak ikut denominator relative. Tiap term memakai jumlah komponen validnya sendiri; coverage root/joint dilaporkan. Tidak ada action classification atau acceleration/forecast loss.

Bone-length diagnostic dari audited edge list. Optional regularizer lambda0.05 membutuhkan kedua endpoints valid dan target per sample; bukan skeleton universal. Default lambda_bone0. Tambahan bone/smoothness config terpisah, diuji fast-motion validation; jangan penalti semua perubahan.

### 9.2 Reference config belum dijalankan

Blok ini spesifikasi mendatang, bukan file YAML yang telah dibuat. Null adalah gate yang harus diisi/ditutup sebelum training.

```yaml
contract_version: m4human_kinetok_v3
paths:
  dataset_root: /dataset
  lmdb_dir: null
  smplx_dir: null                 # offline exporter saja bila diperlukan
  derived_root: derived_data/m4human/m4human_kinetok_v3
data:
  rpc_channels: [x, y, z, intensity]  # setelah schema gate
  schema_hash: null
  joint_map_hash: null
  coordinate_hash: null
  split_manifest_hash: null
  nominal_hz: 12.0
  time_source: null
  context_frames: 4
  context_policy: complete_contiguous_causal
  num_points: 512
  xyz_std_floor_m: 0.001
  intensity_transform: signed_log1p
  augmentation: none
encoder:
  model_id: m4human_set_v2
  initialization: scratch
  pretrained_checkpoint: null
  joint_count: 22
  feature_dim: 256
  point_widths: [64, 128, 256]
  pool: masked_max_mean
  temporal_input_dim: 1032
  temporal_widths: [512, 256]
  head_width: 128
  dropout: 0.0
training:
  seed: 42
  micro_batch: 4
  accumulation: 4
  effective_batch: 16
  optimizer: AdamW
  learning_rate: 0.0002
  betas: [0.9, 0.999]
  weight_decay: 0.01
  warmup_optimizer_step_fraction: 0.05
  scheduler: cosine
  learning_rate_min: 0.000002
  epochs_max: 50
  early_stop_patience: 8
  clip_grad_norm: 1.0
  smooth_l1_beta_m: 0.05
  lambda_root: 1.0
  lambda_bone: 0.0
  precision_reference: fp16_amp_with_grad_scaler
  physical_loss_precision: fp32
  workers: 0
  selection_metric: validation_mpjpe_global_m
  root_quality_gate_m: null       # train/val, dikunci sebelum test
```

Angka H: smoke satu forward/backward/step, overfit 8–16 sample train, pilot beberapa recording dengan cap 3–5 epoch, lalu final setelah gate. Final tidak wajib 50 epoch. Tidak ada training pada perangkat ini dalam pekerjaan dokumen.

Precision reference adalah fp16 AMP+GradScaler. bf16 hanya alternatif yang diuji dengan dukungan perangkat/PyTorch serta precision/forward/backward test, kemudian config aktual disimpan. Jangan mengklaim RTX 3060 pasti menjalankan precision tertentu. Neural layers AMP; normalizer/loss/metrik fp32, time subtraction fp64. Unscale sebelum clip; catat overflow/skipped step/NaN/Inf/grad norm. [PyTorch AMP](https://docs.pytorch.org/docs/main/amp.html), [bf16 support](https://docs.pytorch.org/docs/main/generated/torch.cuda.is_bf16_supported.html).

AdamW decay matriks weight, tanpa decay bias/LN. Warmup/cosine atas successful optimizer steps. Accumulation group pendek diskalakan jumlah microbatch aktual; ukuran microbatch konsisten dan last-batch policy dicatat. Jika ukuran bervariasi, gunakan weighting sample/valid-component yang ditetapkan, bukan selalu membagi4. Scheduler hanya maju saat update berhasil. Resume menyimpan optimizer/scheduler/scaler/RNG/sampler state dengan config cocok.

### 9.3 Validation sebelum freeze

Laporkan root-relative MPJPE, global MPJPE tanpa Procrustes/root alignment, root trajectory error, per-joint/fast-motion errors, dan valid counts. Pose global=P_enc+r_enc. MPJPE Euclidean norm lalu mean joint sah, bukan SmoothL1 atau mesh MVE. Meter dan konversi mm eksplisit; pelvis inclusion/exclusion denominator selaras05.

Pilih minimum validation MPJPE global pada cohort/mask sama, tie-break root error lalu epoch paling awal. Root harus lulus gate task-relevant yang ditetapkan train/val; relative pose baik saja tidak cukup root QA. Tidak ada universal improvement0.02. Gate gagal memicu perbaikan train/val atau laporan terbatas; checkpoint debug tidak menjadi bukti final. Tidak memilih checkpoint/cohort setelah QA test.

Sejak smoke/pilot dan training E, ikuti [kontrak log serta diagnosis pascarun dokumen 05 §12.1–12.3](05_evaluasi_testing.md#121-kesiapan-eksperimen-pertama-dan-development-lintas-perangkat): pose/root loss terpisah, same-metric train/val panel, error per-joint/root/subjek dengan counts, LR/gradient/skipped updates, selected-checkpoint records dan diagnostic trajectories. Tahap E dapat dianalisis sebelum seluruh P0 selesai; log lengkap memberi dasar diagnosis dan iterasi train/validation, tanpa menjamin performa naik.

## 10. Freeze, checkpoint, dan predicted-state cache

### 10.1 Freeze audit

Setelah selection, eval dan requires_grad false seluruh encoder. Simpan checkpoint/state hash sebelum/sesudah extraction, resolved config, software/precision/seed, validation metrics dan gate decision. Test memakai checkpoint terkunci sesuai05. Motion/projector tidak menimpa encoder.

Inference checkpoint: ID/config/state dict, contract, ordered joint map/hash, preprocessing/coordinate/split/source hashes, init/seed/epoch/validation metrics. Resume state terpisah tepercaya. Strict load menolak missing/mismatch keys, foreign hash atau changed normalizer. Perubahan ilmiah membuat run/cache baru.

### 10.2 Cache recording/segmen

```text
derived_data/m4human/m4human_kinetok_v3/   # layout rencana
  audit/{inventory,schema,coordinate_audit,temporal_audit}.json
  audit/rejections.jsonl
  manifests/{frames,recordings}.jsonl
  manifests/{joint_map,split_manifest}.json
  labels/<recipe_hash>/<recording_id>/...         # GT/masks terpisah
  states/<encoder_hash>/<split>/<recording_id>/
    P_enc.npy                 # [F_anchor,22,3] fp32 relative metres
    r_enc.npy                 # [F_anchor,3] fp32 global metres
    f_enc.npy                 # [F_anchor,256] fp16 setelah precision gate
    time_s.npy                # [F_anchor] float64
    sensor_valid.npy          # [F_anchor] bool
    context_source_frames.npy # [F_anchor,4] join manifest
    metadata.json             # frame UID/order/segment + lineage + complete
```

Window reader slicing/memmap bounded; tidak menyimpan context RPC ulang per anchor/window. Root/joint sensor masks reference diturunkan dari sensor_valid; annotation masks tidak masuk namespace state. Metadata mencatat keys/context/time_source/units/origin/joint order/checkpoint/preprocessing/split/source hashes/dtypes/file hash. QA/derivative recipe lineage terpisah menunjuk cache; regenerasi QA tidak mengubah raw GT.

Extraction tanpa augmentasi dengan sampling deterministik. Uji repeat lintas urutan/batch/proses/worker sesuai tolerance. Fitur fp16 hanya setelah precision probe versus fp32 lulus; jika gagal, fp32 dengan biaya tercatat. Physical coordinates fp32. Semua matched conditions memakai encoder/cache sama.

Implementasi kelak menulis temporary artifact pada derived filesystem, validates shape/count/hash, kemudian finalize atomic sejauh filesystem mendukung. Complete baru true setelah seluruh array selesai. Extraction terputus tidak dianggap sukses; jangan overwrite cache/checkpoint valid diam-diam. Folder historis docs/report_training dipertahankan; run report dan INDEX diperiksa setelah training actual sesuai SOP.

## 11. Resource arithmetic, bukan feasibility claim

T: aritmetika core cache untuk F anchor:

```text
bytes_per_anchor = 256*2 + (22*3 + 3)*4 = 788 byte
bytes_core       = F*788
```

Dengan 661.000 anchor sekadar ilustrasi sebelum rejection: fitur 338,432 MB desimal, pose/root 182,436 MB, total 520,868 MB (≈496,74 MiB). Bukan ukuran cache aktual: timestamp/masks/context/metadata/target/shards/temporary belum termasuk. Fitur fp32 menambah F×512 byte. Streaming seluruh dataset tidak mengharuskan seluruh array di VRAM.

T: terminal point-feature tensor microbatch 4 berisi 4×4×512×256 elemen, 4 MiB fp16. Ini satu activation, bukan peak training memory. Weight+gradient+dua momentum Adam fp32 sekitar 16×S byte bagi S parameters, ditambah activations/temporary/allocator/workspace/I/O/CPU objects. Jumlah parameter aktual dari module numel adalah acuan.

Cache motion bersama untuk tahap C harus menyediakan exact-window H yang tetap, bukan hanya Z. H fp16 ialah 32×23×128×2 = 188.416 byte/window (184 KiB), Z fp16 16.384 byte/window (16 KiB). Contoh aritmetika 30.000 unique windows memerlukan H+Z sekitar 6,14 GB desimal sebelum target/masks/container; angka tersebut bukan count paket. M diturunkan dari H on-the-fly, tidak perlu diduplikasi penuh di disk. Precision H/Z dikunci setelah fidelity/probe fp32-versus-fp16 lulus; jika tidak, pakai fp32 dan laporkan biayanya. Cache dibaca per shard/window secara streaming, tidak preload ke RAM16GB, tidak disalin per kondisi/paraphrase. Aritmetika decoder dimiliki03; payload tidak menjamin fit VRAM.

Profil warm-up dan minimal 100 steady-state step di perangkat tujuan mencatat peak allocated/reserved VRAM, process RSS, throughput, data wait, dan validation/extraction time. Untuk A_train anchor, microbatch b, accumulation a, dan tau_group yang terukur:

```text
microbatches_per_epoch = ceil(A_train/b)       # sesuaikan drop_last actual
optimizer_groups      = ceil(microbatches_per_epoch/a)
epoch_seconds         ≈ optimizer_groups*tau_group + validation_seconds
```

Bedakan microbatch/group optimizer; accumulation tidak membuat empat forward gratis. Runtime/GPU-hours diisi dari profil aktual, bukan ukuran 50 GB atau latency paper. Jika resource gagal, kurangi microbatch/worker/prefetch lalu profil. Perubahan N/context/model divalidasi sebelum final; subset contiguous adalah pilihan pilot/resource yang diungkapkan, bukan pengurangan subject otomatis.

## 12. Alternatif encoder bersyarat dan MM-Fi historis

P0 satu set encoder. Alternatif tidak wajib dilatih sekaligus dan tidak mengubah core study menjadi kompetisi encoder.

| Opsional | Trigger train/val | Dasar/batas |
|---|---|---|
| P4-lite lokal | Bottleneck short-context/geometri lokal nyata | Local space–time aggregation tanpa tracking; port ringan/mask-aware bukan reproduksi original. Radius convolution≠kNN. [P4Transformer resmi](https://github.com/hehefan/P4Transformer) |
| PointMLP-lite lokal | Pooling global kehilangan detail joint | Residual MLP/local grouping alternatif; stock memakai pointnet2_ops FPS, jadi tidak seluruhnya pure PyTorch. Port/tes mask diperlukan. [PointMLP paper](https://arxiv.org/abs/2202.07123), [model resmi](https://github.com/ma-xu/pointMLP-pytorch/blob/main/classification_ModelNet40/models/pointmlp.py) |
| Point-MAE scratch vs transfer | Pertanyaan pretraining/label efficiency memang diuji | MAE rekonstruksi patch memberi preseden, bukan bukti CAD→radar membantu. Pasangan same architecture/head/input/budget dan audited provenance. [Point-MAE paper](https://arxiv.org/abs/2203.06604) |

Sebelum challenger, tetapkan recipe grouping/masks/head/init/optimizer/budget konkret. Semua memakai channel/context causal/output22/256/target/split/metric sama. Satu selected encoder final untuk seluruh core comparisons; tanpa Cartesian sweep encoder×motion×LLM. Perbedaan architecture/init/optimization dilaporkan, bukan isolasi transfer.

Transfer opsional memetakan tensor explicit: jangan potong384→256 atau silent skip lalu menyebut pretrained. Audit channel/joint map/normalizer/source train subjects/actions/loaded numel/layer coverage/forward equivalence/checkpoint trust/hash. XYZ terverifikasi boleh adapter yang dinyatakan; Doppler/SNR MM-Fi tidak direlabel intensity. Head17 bukan22. Coverage policy sesuai intended backbone, bukan angka arbitrer yang menjamin kualitas.

Checkpoint/report MM-Fi/ShapeNet historis, bukan bukti keberhasilan M4Human. Schema lima kanal/17 joint, dt lama, Model A/Av2, dan forecasting lama tidak diwariskan otomatis. Utility MM-Fi `sample_or_pad_points` mengulang titik dan normalizer dapat memusatkan tiap frame; jangan reuse tanpa adaptasi mask/root global. Tidak mengubah kode/history lama dalam pekerjaan ini. Default Pure-PyTorch; compiled C++/CUDA baru memerlukan persetujuan proyek. Pilot compact tidak mendukung klaim SOTA.

## 13. Implementasi mendatang dan gate penerimaan

### 13.1 Path/API rencana, belum executable

| Path rencana | Tanggung jawab |
|---|---|
| eksperimen_model/datasets/m4human_dataset.py | RPC-only reader, serializer validation, sensor context, target join terpisah |
| eksperimen_model/datasets/m4human_state_dataset.py | Window read-only cache dan target recipe |
| eksperimen_model/models/m4human_encoder.py | M4HumanSetEncoderV2 dan output contract |
| eksperimen_model/configs/m4human_encoder.yaml | Config resolved berdasarkan reference |
| eksperimen_model/audit_m4human.py | Inventory/schema/split/coordinate/gap audit tanpa write source |
| eksperimen_model/export_m4human_targets.py | SMPL-X target offline/unified body map |
| eksperimen_model/train_m4human_encoder.py | Smoke/pilot/final, loss/mask/checkpoint/report |
| eksperimen_model/extract_m4human_states.py | Frozen extraction, determinism/lineage/finalization |
| eksperimen_model/test_m4human_encoder_contract.py | Satu self-check runnable fixture kecil/assertions |

Implementasi kelak memakai `.venv`/`uv`; sanity repo dan M4Human sebelum full training. Reuse NumPy/PyTorch/helper/report/checkpoint jika semantics sesuai. Reader dapat memerlukan lmdb/msgpack, exporter smplx; instalasi/versi actual diaudit. Jangan menyalin stock stack atau memasang native point ops otomatis. JSON/JSONL dan NumPy memmap cukup, tanpa mandatory Parquet/cache database baru. Tabel ini tidak memberi shell command yang berpura-pura modul sudah tersedia.

### 13.2 Minimum checks yang harus runnable saat implementasi

| Gate | Check | Evidence |
|---|---|---|
| G-DATA | Fixture valid/truncated/salah dtype-shape-endian; trusted index gate | Schema/provenance/rejection; source immutable |
| G-SPLIT | Subject disjoint; union context/derivative/window tidak lintas split/recording/gap | Split hash dan actual support/count |
| G-COORD | Transform inverse, mm↔m, pelvis decomposition, offset/origin/map22 | Unit/coordinate/joint audit dan examples |
| G-INPUT | GT/body/action/annotation-mask changes tidak mengubah inference | Dependency isolation assertion |
| G-MASK | Padding/permutation/<512/empty finite; no invalid mean/max | Pool/mask/sampling determinism |
| G-CAUSAL | Future RPC change tidak mengubah anchor; gap reset | Source support/causal fixture |
| G-TRAIN | Shape/finite loss/gradient; kedua heads memberi gradient ke stem/aggregate; step/overfit kecil | Trainable count/smoke/train-only overfit |
| G-VALIDATE | Relative/global/root/count; root/task gate dikunci | Selection/config/checkpoint tanpa test feedback |
| G-CACHE | Strict checkpoint/repeat extraction/frame join/precision | Hash/complete/tolerance |
| G-RESOURCE | Warm-up+steady-state actual; spawn jika dipakai | VRAM/RAM/I/O/throughput/runtime |

Satu self-check kecil dapat mencakup fixtures ini tanpa framework baru. Sanity lulus bukan jaminan hasil riset. Pekerjaan dokumentasi ini tidak menjalankan gate/data/model, mengakses `/dataset`, training, atau membuat checkpoint/cache.

### 13.3 Handoff

Handoff sah: audit source/schema/unit/time, locked subject split, joint22/pelvis map, target offline/masks terpisah, satu frozen encoder checkpoint, predicted P_enc/r_enc/f_enc cache dengan context lineage, dan actual resource profile. Tidak menyuntikkan GT ke motion. Gate gagal menghasilkan status debug/terbatas dan failure report, bukan hasil negatif ilmiah otomatis.

Integrasi kelima dokumen memeriksa `m4human_kinetok_v3`, field/units, sensor versus annotation masks, derivative warm-up first6, raw position/filtered velocity, body groups/enums, exact-window reset, H/M/Z/U, full-H fidelity, `C_base(aux M)` versus `C_kin(aux U)`, dan strict lineage. 02/03 melatih motion/readout bersama kemudian membekukannya; tahap C melatih kompresor dan readout training-only; 04 membekukan semuanya dan melatih projector saja; 05 menguji token U dan QA pada held-out subjects. Satu seed PoC bersifat eksploratif; uncertainty memakai kelompok subjek/recording, bukan frame independen.

Hasil negatif berarti hipotesis tidak didukung pada kondisi yang diuji hanya bila label reliable, training/optimasi memadai dan perbandingan valid. Bug, target salah, atau training tidak belajar berarti inconclusive. Jika baseline tidak menunjukkan kehilangan informasi yang relevan, jangan mengklaim telah menyelesaikan bottleneck; laporkan batas dan sempitkan klaim melalui protokol train/val sebelum test, bukan mengubah panel setelah melihat test.

## Pembaruan implementasi — 6 Oktober 2026

Audit, export target, trainer E, ekstraksi frozen E, profiler bounded dan gate checkpoint tersedia sebagai implementasi lokal. Validasi fixture bukan audit paket nyata; default kualitas, timestamp, calibration dan root threshold tetap membutuhkan bukti perangkat dataset. Input P0 ialah RPC XYZ/intensity; RT hanya kandidat ablation terpisah setelah schema/audit, bukan channel Doppler yang diasumsikan tersedia.

Loader mendukung Windows spawn, lazy LMDB/memmap per proses, persistent-worker epoch propagation, pinning, prefetch dan transfer nonblocking. Baseline E batch efektif 16 dipertahankan pada config referensi. `m4human_encoder_rtx3060_candidate.yaml` memilih 64×2=128 sebagai protokol baru yang belum diukur, tanpa automatic LR scaling; bandingkan melalui pilot train/val dan catat budget/exposure. Profil terikat config, bukan klaim batch128 muat/lebih cepat.

Root threshold dideklarasikan dari train/val sebelum selected run, tidak dari test. `real_smoke_passed` dan `resource_profile_passed` adalah mapping `artifact_path` + actual `artifact_sha256`, bukan boolean. Smoke bundle lengkap dan source/target/schema/joint/coordinate/split/recipe lineage diverifikasi. Profiler scientific readiness memerlukan m4human CUDA parity, no-overflow dan memory reserve; CPU/synthetic report tidak menutupnya. Gate spec/decision serta training mode disimpan pada selected checkpoint; smoke selalu non-scientific, pilot memerlukan predeclaration+readiness. Pengeditan config ekstraksi tidak mempromosikan model. Inferensi tidak membuka target atau readiness files lagi. Perintah dan schema aktual ada pada [runbook §7](../m4human_development_runbook.md#7-profiling-rtx-3060-12-gb-dan-gate-checkpoint).
