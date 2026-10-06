# Runbook development M4Human v3

Terakhir diperbarui: **6 Oktober 2026**, dengan optimasi RAM16GB/VRAM12GB, profiler E/M/C/probe/L dan dua belas suite verifikasi CPU. Mulai dari §0 jika dataset/metadata sudah tersedia dan smoke lama telah dijalankan. Bagian berikutnya menjadi referensi rinci setiap tahap.

Implementasi mengikuti lima dokumen `docs/planning_m4human/` dan proposal terbaru. Kontraknya `m4human_kinetok_v3`: E → M → compressor C → probe independen → projector ke Qwen beku. Ini implementasi P0 untuk membaca kinematika **jendela observasi**; bukan bukti forecasting atau reasoning counterfactual. Pipeline dan laporan MM-Fi tetap dipertahankan.

## 0. Urutan setelah menerima perbaikan kode

**Persiapan dataset tidak perlu dimulai dari nol. Pemeriksaan setelah perubahan kode tetap perlu diulang di komputer training.** Verifikasi terbaru **12/12 suite CPU passed**, dengan59 source/config hashes tidak berubah: [laporan hardware](m4human_verification_20261006_hardware.json). [9/9 suite sebelumnya](m4human_verification_20261006_hardening.json) tetap menjadi bukti historis. Environment smoke historis memakai PyTorch 2.6/CUDA; hasil lokal belum membuktikan kompatibilitas environment itu, kecepatan RTX 3060, atau kualitas data/model nyata.

| Artefak sebelumnya | Dapat dipakai kembali jika | Tindakan |
|---|---|---|
| LMDB dataset | Snapshot sumber sama dan immutable | Gunakan kembali; ulangi audit verify dengan kode baru |
| Schema, joint map, coordinate audit, manifest | Semantik serta bukti unit/joint/recording/split masih benar | Periksa bukti dan hash; tidak perlu menyusun ulang tanpa perubahan |
| Target offline | Paket sumber, export recipe/aset SMPL-X, frame manifest, joint map dan koordinat tetap cocok; integrity arrays/manifest valid | Gunakan kembali setelah pemeriksaan; ekspor ke direktori baru jika input/semantik berubah |
| Smoke/checkpoint/cache lama | Untuk referensi historis dan diagnosis | Simpan; jalankan smoke baru dan training E baru untuk kode yang diperbaiki |
| Profil hardware lama | Konfigurasi/environment lama | Ukur kembali; perubahan loader/AMP/batching memengaruhi hasil |

### 0.1 Sinkronkan kode dan periksa environment

Jika perbaikan masih berupa perubahan lokal, commit/push dari laptop atau salin file yang berubah, kemudian sinkronkan komputer training. `git pull` hanya membawa perubahan yang sudah tersedia di remote. Pertahankan instalasi CUDA yang bekerja. Jalankan dari root repository di PowerShell:

```powershell
$py = ".venv\Scripts\python.exe"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$env:PYTHONDONTWRITEBYTECODE = "1"
git status --short
git log -1 --oneline
& $py -c "import torch, transformers; print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'transformers', transformers.__version__); assert torch.cuda.is_available(), 'CUDA belum tersedia'; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 1024**3, 'GiB')"
& $py -m pip install -r requirements-m4human.txt
```

`$stamp` dipakai untuk output baru pada sesi ini. Jika mengulang command yang telah menghasilkan file/direktori, ganti nama output. Jangan menimpa bukti run sebelumnya. Jika CUDA belum tersedia, selesaikan environment sebelum profil/training GPU; tes CPU pada langkah berikut tetap dapat dijalankan.

### 0.2 Ulangi dua belas suite pada komputer training

```powershell
& $py -m eksperimen_model.test_m4human_pipeline --output "docs/m4human_verification_training_pc_$stamp.json"
```

Lanjutkan setelah `status: passed`, seluruh dua belas checks ber-exit-code 0, dan `source_changed_during_checks: false`. Suite ini memakai fixture, bukan dataset nyata. Kegagalan perlu diselesaikan sebelum training panjang. Dependency `psutil` diperlukan untuk memantau RAM termasuk worker.

### 0.3 Audit ulang dan tentukan target yang masih bisa dipakai

Contoh menggunakan metadata dari smoke sebelumnya di `D:/m4human_meta`; sesuaikan path dengan file aktual. Dataset tetap di `datasets/m4human`. Manifest input berikut sudah tersedia dari audit lama, sehingga mode draft tidak perlu diulang jika identitas dan split benar.

```powershell
$meta = "D:/m4human_meta"
$audit = "$meta/audit_$stamp"
& $py -m eksperimen_model.audit_m4human --mode verify --root datasets/m4human --schema "$meta/schema.json" --frame-manifest "$meta/audit/frames.jsonl" --joint-map "$meta/joint_map.json" --coordinate-audit "$meta/coordinate_audit.json" --output-dir $audit
```

Periksa `audit_report.json`: status verified, key coverage tanpa missing/unexpected, rejection counts, partition counts dan bukti semantik yang disuplai. `annotation_semantics_verified:false` pada audit struktural berarti anatomy/unit/kalibrasi masih perlu diperiksa melalui bukti/overlay; status verified tidak menggantikan pemeriksaan itu.

Bandingkan `frame_manifest_hash`, `joint_map_hash`, dan `coordinate_hash` dengan `lineage` dalam target `metadata.json`. Cocokkan juga paket sumber dan `export_recipe_hash` terhadap export config/aset yang dipin. Loader memeriksa hash arrays/manifest, count, dtype dan shape target. Jika salah satu input atau semantik berubah, ekspor ulang target ke direktori baru sesuai §2; jangan mengedit hash metadata agar cocok.

Buat config run `E.yaml` dari `eksperimen_model/configs/m4human_encoder_run.yaml` tanpa menimpa config yang masih dibutuhkan. Isi paths aktual; `paths.frame_manifest` menunjuk `$audit/frames.jsonl` dan `paths.audit_report` menunjuk `$audit/audit_report.json`. Target lama boleh tetap dipakai jika pemeriksaan di atas cocok. Untuk diagnosis awal, gunakan `training.precision: fp32`, microbatch4 × accumulation4 = effective_batch16, workers0, dan root threshold null. Hapus kedua mapping readiness yang masih menunjuk run/profil lama; field tersebut belum dibutuhkan pada diagnosis.

### 0.4 Tiny-overfit train-only dalam fp32

```powershell
& $py -m eksperimen_model.check_m4human_encoder_overfit --config E.yaml --output "$meta/overfit_E_$stamp.json" --device cuda:0 --samples 16 --updates 200 --loss-ratio-max 0.25
```

Periksa status, initial/final loss dan loss ratio. Default ratio 0.25 adalah diagnostic yang dipilih sebelum run, bukan threshold kualitas ilmiah universal. Jika gagal, periksa target, masks, unit/koordinat, input dan optimisasi sebelum melanjutkan. Command tidak menghasilkan checkpoint ilmiah dan tidak membuka validation/test samples.

### 0.5 Screening hardware, lalu profil pilihan minimal 100 update

Ikuti §7: screening awal E dapat menyapu batch efektif16/64/128/256/512/1024 dan microbatch4 hingga1024, workers0/2/4, fp32/AMP. Profiler menjaga sisa RAM2048MiB dan headroom CUDA1024MiB. Produk microbatch×accumulation yang berbeda dari16 adalah protokol baru: bandingkan melalui pilot train/val dengan exposure/update budget yang dicatat. Pilih setting yang lulus parity, stabil dan cepat.

Gunakan `--selected-config E_selected.yaml` untuk menulis config lengkap dari rekomendasi, atau salin **seluruh** `recommended_settings` ke bagian `training`. Pertahankan paths/data/encoder dan scientific loss policy yang sama dengan config smoke. Tetapkan semua field lain sebelum reprofile. Bila `scientific_gate_predeclared` diperlukan untuk pilot ilmiah, tetapkan sebelum profil; pilot diagnostic dan final tidak memerlukan flag pilot tersebut.

Jalankan profil ulang hanya pada setting terpilih dengan `--timed-steps 100`, sesuai §7. Cocokkan seluruh `recommended_settings` dan `readiness_config_hash` dengan config terpilih. Jika ada setting lain yang berubah, simpan config baru dan ulangi profil. Report CPU/sintetis atau screening10 update tidak menjadi bukti readiness final.

### 0.6 Smoke baru dan kelengkapan artefak

Gunakan config terpilih yang belum memiliki mapping readiness; sensor/source policy dan path spelling harus sama dengan run yang akan dilatih. CLI `--precision fp32` hanya mengubah precision run smoke, yang boleh berbeda dari profil AMP.

```powershell
$smoke = "docs/report_training/RUN_M4HUMAN_E_SMOKE_$stamp"
& $py -m eksperimen_model.train_m4human_encoder --config E_selected.yaml --output-dir $smoke --mode smoke --device cuda:0 --precision fp32
& $py -m eksperimen_model.validate_m4human_run $smoke
```

Smoke E menjalankan satu epoch penuh, bukan hanya beberapa batch. Lanjutkan setelah run complete dan validator memberi `analysis_ready:true` tanpa issues. Simpan best/last checkpoint, raw selected predictions, diagnostics dan witness. Bundle Git historis yang kehilangan checkpoint/predictions tidak dapat menggantikan smoke ini. Smoke selalu non-scientific, sekalipun error root rendah.

### 0.7 Pilot, threshold train/val, dan evidence untuk final

```powershell
& $py -m eksperimen_model.train_m4human_encoder --config E_selected.yaml --output-dir "docs/report_training/RUN_M4HUMAN_E_PILOT_$stamp" --mode pilot --device cuda:0
```

Pilot diagnostic dibatasi maksimal lima epoch dan dapat dijalankan tanpa readiness/root threshold. Periksa kurva train/val, global/relative/root error, overlay diagnostik, overflow retries, elapsed time serta allocated/reserved VRAM. Jika config/model/sumber diubah, ulangi langkah yang terkait; perubahan config selain pengecualian hash di §7 memerlukan profil terpilih baru. Threshold root ditentukan dari train/validation sebelum selected final run; test belum dibuka.

Setelah pilihan dipin, tambahkan `real_smoke_passed` dengan path `$smoke/metrics.json` dan SHA256 bytes, `resource_profile_passed` dengan report CUDA100+ terpilih dan SHA256 bytes, serta `training.root_quality_gate_m` yang telah ditentukan. Kedua mapping dan threshold dikecualikan dari readiness hash, sehingga penambahan ini tidak sendiri membutuhkan reprofile. Jangan menambahkan/mengubah flag pilot atau setting lain setelah profil tanpa memeriksa hash.

Checkpoint diagnostic pilot yang terlanjur non-scientific tidak berubah status setelah config diedit. Bila pilot hendak menjadi checkpoint ilmiah, flag pilot, threshold dan evidence harus sudah dideklarasikan sebelum run pilot tersebut. Workflow utama di sini memakai E final baru setelah pilot diagnosis.

### 0.8 E final, validasi, kemudian tahap downstream

```powershell
$final = "docs/report_training/RUN_M4HUMAN_E_FINAL_$stamp"
& $py -m eksperimen_model.train_m4human_encoder --config E_selected.yaml --output-dir $final --mode final --device cuda:0
& $py -m eksperimen_model.validate_m4human_run $final
```

Mulai final dari scratch sesuai protokol E; jangan meneruskan checkpoint lama dari kode/config/budget berbeda. Final preflight yang lulus belum menjamin error root checkpoint terbaik akan memenuhi threshold. Lanjutkan setelah artefak valid **dan** metrics/selected checkpoint menyatakan `scientific_freeze_eligible:true`/keputusan gate eligible. Jika gate kualitas gagal, kembali ke diagnosis train/val.

Ekstraksi memakai config final yang sama, tanpa mengubah setting setelah checkpoint dipilih; pilih batch ekstraksi melalui CLI bila perlu:

```powershell
& $py -m eksperimen_model.extract_m4human_states --config E_selected.yaml --checkpoint "$final/best.pt" --output-dir "$meta/states_$stamp" --device cuda:0 --batch-size 4
```

Lanjutkan **M → H → C_base/C_kin → probe → QA/projector → evaluasi test terkunci** sesuai §3–6. Isi lineage/path downstream dari cache baru dan gunakan output baru. Validasi kualitas serta numeric/resource gates tiap tahap; profil E tidak otomatis mensertifikasi M/C/probe/Qwen. C_base/C_kin dan projector conditions tetap matched. Cache diagnosis memakai `--allow-debug` dan tidak boleh menjadi upstream ilmiah.

Sebelum memindahkan hasil antarkomputer, validasi dan buat bundle lengkap menggunakan §8. Dataset dan target offline tetap eksternal; checkpoint/predictions yang diabaikan Git harus ikut dibawa melalui bundle.

## 1. Verifikasi kode tanpa dataset pada setiap environment

Jalankan dari root repository dengan Python virtual environment. Pada Windows, ganti `.venv/bin/python` dengan `.venv\Scripts\python.exe`, atau gunakan `uv run python` dengan environment yang sudah disiapkan.

```sh
.venv/bin/python -m eksperimen_model.test_m4human_pipeline --output docs/m4human_verification_local.json
```

Output harus memakai nama baru jika laporan sebelumnya sudah ada. Pemeriksaan memakai LMDB dan tensor sintetis, serta Qwen2 kecil berbobot acak dari Transformers yang dipin. Tidak mengunduh dataset, pretrained Qwen, atau aset SMPL-X. Laporan mencatat hasil setiap tes, versi runtime, durasi dan hash source. Versi terpasang pada laporan adalah bukti environment yang diuji; rentang dalam `requirements-m4human.txt` bukan klaim semua versi sudah diuji.

Dependency M4Human terpisah dari `requirements.txt` historis. Buat `.venv` Python 3.12, instal PyTorch yang sesuai CUDA komputer eksperimen, lalu instal `requirements-m4human.txt`. Jangan mengganti wheel CUDA yang sudah benar dengan wheel CPU. Transformers dipin `4.45.2` karena generation dari physical embeddings diuji terhadap implementasi tersebut.

## 2. Audit paket pada komputer dataset

Root default adalah `datasets/m4human`. Reader mendukung LMDB berupa file datar sebagaimana paket pengguna. `*.lmdb-lock` adalah berkas koordinasi LMDB, bukan sampel atau label. Input sensor hanya `radar_pc.lmdb`; `params` dan `calib` digunakan eksportir target offline. `bbox`, `radar_comp`, `rgb_feat` tidak menjadi input model P0.

Gunakan snapshot sumber yang tidak sedang ditulis proses lain. Semua artefak turunan berada di luar root dataset, default `derived_data/m4human/m4human_kinetok_v3/` (selanjutnya **D** dalam contoh). Jangan mengganti `audited:false` menjadi `true` untuk melewati pemeriksaan; flag harus disertai bukti pemeriksaan aktual.

```sh
.venv/bin/python -m eksperimen_model.audit_m4human --mode inspect --root datasets/m4human --output-dir D/inspection
```

Inspeksi hanya melaporkan inventori, contoh key dan header. Ia tidak menebak urutan byte, unit, frekuensi atau frame koordinat. Sebelum verifikasi, siapkan:

| Artefak | Isi yang harus diselesaikan berdasarkan paket aktual |
|---|---|
| `schema.json` | `audited`, `immutable_snapshot`, channel `[x,y,z,intensity]`, `xyz_unit:meter`, `byte_order`, batas ukuran, `sentinel_policy`, `max_gap_s`, source provenance |
| `frame_manifest.jsonl` | key sumber, frame UID, subject/action/recording/segment, nomor frame, waktu dalam detik, asal waktu, split subjek, status sensor |
| `joint_map.json` | 22 nama berurutan, indeks regressor, pelvis, `task_groups` arms/legs dengan elbow/wrist dan knee/ankle kiri/kanan |
| `coordinate_audit.json` | frame sumber/radar, unit, kalibrasi dan bukti origin/orientasi; unit translasi masing-masing transform |
| `target_export.json` | metode SMPL-X mesh/regressor atau direct-joints dengan bukti ekuivalensi, aset/hash/gender mapping/default parameter yang diaudit |

Struktur executable dari setiap artefak ada pada fixture di `test_m4human_encoder_contract.py`; nilainya sintetis dan **tidak boleh disalin sebagai hasil audit**. `source_provenance` memerlukan package ID, full upstream commit, serializer/calibration SHA-256 dan preprocessing RPC. Implementasi serializer dicocokkan dengan [utilitas LMDB resmi](https://github.com/FanJunqiao/M4Human/blob/main/dataset/lmdb_utils.py); kompatibilitas paket di komputer pengguna tetap perlu diperiksa.

Mode draft dapat menyusun indeks jika audit telah membuktikan satu take per subject/action dan grid waktu nominal:

```sh
.venv/bin/python -m eksperimen_model.audit_m4human --mode draft --root datasets/m4human --policy D/index_policy.json --output-dir D/draft
.venv/bin/python -m eksperimen_model.audit_m4human --mode verify --root datasets/m4human --schema D/schema.json --frame-manifest D/draft/draft_frames.jsonl --joint-map D/joint_map.json --coordinate-audit D/coordinate_audit.json --output-dir D/audit
.venv/bin/python -m eksperimen_model.export_m4human_targets --root datasets/m4human --manifest D/audit/frames.jsonl --export-config D/target_export.json --joint-map D/joint_map.json --coordinate-audit D/coordinate_audit.json --output-dir D/targets
```

Jika asumsi mode draft tidak benar, sediakan manifest eksplisit dari metadata recording/timestamp aktual. Gap membentuk segment baru; konteks tidak boleh melintasinya. Split subjek harus ditetapkan sebelum fitting normalizer, training atau seleksi model. Verifikasi struktur tidak otomatis membuktikan semantik joint/unit benar. Inspeksi visual overlay dan pemeriksaan kuantitatif target nyata tetap wajib.

## 3. Encoder common dan motion common

Bagian ini menjelaskan sambungan antartahap. Untuk memulai dari smoke lama, selesaikan urutan §0 dahulu. `E_RUN` pada ekstraksi berarti selected run yang telah memenuhi gate checkpoint; ordinary smoke/pilot diagnostic tidak memenuhi syarat itu.

Salin konfigurasi referensi ke konfigurasi run, isi seluruh path/hash/null berdasarkan audit. `joint_map_hash`, schema dan coordinate hash memakai SHA-256 **bytes file**; `recipe_hash` memakai canonical JSON dari recipe. Ambil lineage dari metadata artefak upstream agar tidak mencampur dua definisi hash.

```sh
.venv/bin/python -m eksperimen_model.train_m4human_encoder --config E.yaml --output-dir docs/report_training/RUN_M4HUMAN_TIMESTAMP_E_SMOKE --mode smoke --device cuda --precision fp32
.venv/bin/python -m eksperimen_model.train_m4human_encoder --config E.yaml --output-dir docs/report_training/RUN_M4HUMAN_TIMESTAMP_E_PILOT --mode pilot --device cuda
```

Selesaikan real smoke, tiny-overfit, target/root-quality gate dan profil resource sebelum `--mode final`. Deklarasikan threshold root berdasarkan train/val sebelum run yang akan dipilih; test tidak boleh menentukan threshold. Konfigurasi final memerlukan artefak nyata pada `real_smoke_passed` dan `resource_profile_passed` sebagaimana prosedur §7. Tidak ada angka performa target yang ditebak dari fixture. Encoder diinisialisasi dari scratch sesuai P0, tidak memuat checkpoint MM-Fi.

```sh
.venv/bin/python -m eksperimen_model.extract_m4human_states --config E.yaml --checkpoint E_RUN/best.pt --output-dir D/states --device cuda
.venv/bin/python -m eksperimen_model.train_m4human_motion --config M.yaml
.venv/bin/python -m eksperimen_model.extract_m4human_motion --config M.yaml --checkpoint M_RUN/M/best.pt --output D/H
```

Tidak memberikan `--split` saat ekstraksi menyimpan seluruh partition yang sudah ditetapkan; training tetap hanya membaca train/val. Ekstraksi E memakai RPC dan checkpoint beku tanpa membuka target/params. Untuk cache H berlabel offline, `M.yaml` menunjuk target cache; training loss membutuhkan target terpisah. Loader H dan ekstraksi U untuk inferensi tidak memerlukan target tersedia.

`M.yaml` memerlukan `max_gap_s`, `successful_updates`, lineage, joint map dan recipe yang diselesaikan. Normalizer dihitung pada train; empat kelompok loss fisik memiliki penyebut terpisah. `precision: fp16_amp` membutuhkan CUDA dan `precision_gate_passed:true` berdasarkan numeric gate aktual. Untuk M/C/probe, `scientific_gates_passed:true` adalah attestasi manual bahwa audit/smoke/tiny-overfit dan quality gates tahap telah selesai; bukan hasil otomatis profiler. Trainer juga memverifikasi complete scientific upstream cache dan mandatory lineage. Nilai template false/null harus diselesaikan beserta bukti di run notes; jangan hanya mengubah flag untuk melewati gate. Mulai dengan fp32 saat smoke jika precision gate belum tersedia.

## 4. Compressor dan probe matched

```sh
.venv/bin/python -m eksperimen_model.train_m4human_compressor --config C.yaml --condition C_base
.venv/bin/python -m eksperimen_model.train_m4human_compressor --config C.yaml --condition C_kin
.venv/bin/python -m eksperimen_model.extract_m4human_tokens --config C.yaml --checkpoint C_RUN/C_base/best.pt --output D/U_C_base_K16
.venv/bin/python -m eksperimen_model.extract_m4human_tokens --config C.yaml --checkpoint C_RUN/C_kin/best.pt --output D/U_C_kin_K16
.venv/bin/python -m eksperimen_model.train_m4human_probes --config PROBE_M.yaml --condition M
.venv/bin/python -m eksperimen_model.train_m4human_probes --config PROBE_BASE.yaml --condition U_base
.venv/bin/python -m eksperimen_model.train_m4human_probes --config PROBE_KIN.yaml --condition U_kin
```

Gunakan K16 primer. K8/K32 membutuhkan kontrak paired dan run baru. C_base dan C_kin memiliki initialization, normalizer, urutan data, update budget dan full-H fidelity yang sama. Perbedaan hanya sumber auxiliary M versus U; clipping compressor/fidelity terpisah dari auxiliary. Probe memakai inisialisasi baru dan kontrak paired tersendiri. Pada tiga config probe, bedakan token cache dan output sesuai kebutuhan; pertahankan training policy dan paired contract yang sama. `normalizer_path` menunjuk `M_RUN/M/normalizers.json`.

```sh
.venv/bin/python -m eksperimen_model.evaluate_m4human_physical predict --config PROBE_BASE.yaml --condition U_base --checkpoint PROBE_RUN/U_base/best.pt --split val --output D/physical_base_val.json
.venv/bin/python -m eksperimen_model.evaluate_m4human_physical predict --config PROBE_KIN.yaml --condition U_kin --checkpoint PROBE_RUN/U_kin/best.pt --split val --output D/physical_kin_val.json
.venv/bin/python -m eksperimen_model.evaluate_m4human_physical compare --base D/physical_base_val.json --kin D/physical_kin_val.json --output D/physical_comparison_val.json
```

`D_encoder_sg` menyediakan baseline predicted-pose + trailing derivative tanpa checkpoint probe. `--qa D/qa.jsonl` menghasilkan raw category predictions untuk kontrol rule/probe. Model yang tidak menghasilkan prediksi dicatat sebagai failure; estimasi error pasangan finite dilaporkan bersama jumlah kegagalan, bukan dianggap error nol.

Untuk diagnosis retensi, simpan pula prediksi probe kondisi `M`, lalu gunakan comparator dengan pasangan M→U_base dan M→U_kin. Selisih error positif berarti kandidat mempunyai error lebih rendah dari reference pada support berpasangan.

## 5. QA dan frozen Qwen

```sh
.venv/bin/python -m eksperimen_model.datasets.generate_m4human_qa --h-cache D/H --joint-map D/joint_map.json --output D/qa.jsonl
.venv/bin/python -m eksperimen_model.train_m4human_projector --config L.yaml --qa D/qa.jsonl --u-cache D/U_C_base_K16 --condition C_base --initial D/projector_initial.pt --initialize --output L_UNUSED --device cuda:0
.venv/bin/python -m eksperimen_model.train_m4human_projector --config L.yaml --qa D/qa.jsonl --u-cache D/U_C_base_K16 --condition C_base --initial D/projector_initial.pt --output L_BASE_RUN --device cuda:0
.venv/bin/python -m eksperimen_model.train_m4human_projector --config L.yaml --qa D/qa.jsonl --u-cache D/U_C_kin_K16 --condition C_kin --initial D/projector_initial.pt --output L_KIN_RUN --device cuda:0
```

Ganti revision Qwen yang null dengan full commit yang dipin. `--initialize` hanya menyimpan shared projector dan kalibrasi alpha train-only; argumen output tetap diminta parser; mode initialize tidak membuat run training. Kedua kondisi harus memuat initial yang sama. Hanya projector dilatih. Frozen Qwen tetap meneruskan gradien input ke projector. Generated validation dan parse coverage memilih checkpoint; jawaban teacher-forced bukan metrik reasoning.

QA memakai dua core task. `task_interval_s` menyatakan interval waktu pertanyaan setelah warmup; `label_support`/`evidence_support` mencatat indeks frame yang benar-benar lolos mask/guard untuk recipe. Interval nominal bukan janji seluruh frame valid. Evidence prediksi mengikuti interval query dan recipe yang sama; label support GT tidak dibocorkan sebagai input bahasa. Target undefined dikeluarkan dengan alasan; unknown tetap kategori jawaban sah. Derivative memakai tujuh sampel trailing, timestamp aktual dan reset warmup enam frame pada setiap jendela 32. Optional onset/radial tidak masuk generator P0 sebelum reliability gate tersendiri.

## 6. Lock test, inferensi dan laporan

Selesaikan pilot dan pin checkpoint, tugas, toleransi efek praktis serta kebijakan uncertainty **sebelum** menghasilkan held-out outcomes. Physical lock terpisah dari language lock; `--help` menjelaskan masing-masing argumen.

Physical lock menerima JSON spec dengan `held_out_outcomes_accessed:false`, `effect_tolerance` berupa mapping nama metrik ke toleransi numerik nonnegatif yang dipilih dari train/val, `uncertainty:{"method":"paired_subject_cluster_row_weighted","draws":2000,"seed":42}`, dan `entries` untuk `M`, `U_base`, `U_kin`. Setiap entry memuat `config`, `checkpoint` best.pt dan optional `qa` (jika category outputs akan dinilai). Ketiga run harus lengkap, matched dan memakai cohort identik. Tambahkan `D_encoder_sg` tanpa checkpoint jika baseline itu juga akan diuji.

```sh
.venv/bin/python -m eksperimen_model.evaluate_m4human_physical lock --spec D/physical_lock_spec.json --output D/physical_test_lock.json
.venv/bin/python -m eksperimen_model.evaluate_m4human_physical predict --config PROBE_BASE.yaml --condition U_base --checkpoint PROBE_RUN/U_base/best.pt --split test --test-contract D/physical_test_lock.json --output D/physical_base_test.json
```

Jika entry mengunci QA, sertakan file identik dengan `--qa`. Gunakan physical lock yang sama saat membandingkan prediksi test. Untuk language lock, `uncertainty.json` memakai format uncertainty yang sama; `effect_tolerance.json` menyimpan toleransi yang dipin sebelum hasil test.

```sh
.venv/bin/python -m eksperimen_model.evaluate_m4human --qa D/qa.jsonl --create-test-lock --config L.yaml --base-cache D/U_C_base_K16 --kin-cache D/U_C_kin_K16 --base-checkpoint L_BASE_RUN/best.pt --kin-checkpoint L_KIN_RUN/best.pt --effect-tolerance-json D/effect_tolerance.json --uncertainty-json D/uncertainty.json --output D/language_test_lock.json
.venv/bin/python -m eksperimen_model.evaluate_m4human_reasoning --config L.yaml --qa D/qa.jsonl --u-cache D/U_C_base_K16 --checkpoint L_BASE_RUN/best.pt --condition C_base --split test --test-contract D/language_test_lock.json --output D/base_test.json
```

Ulangi inference untuk C_kin dan `--text-only` untuk baseline teks, lalu gunakan `evaluate_m4human --base ... --kin ... --text ... --rule ... --probe ... --split test --test-contract ... --output ...`. Majority baseline berasal dari train. Donor harus same-activity, beda recording dan beda jawaban referensi dengan query/support kompatibel; laporkan coverage. Tidak ada klaim physical counterfactual dari shuffle.

```sh
.venv/bin/python -m eksperimen_model.validate_m4human_run RUN_DIRECTORY
```

Setiap run menyimpan config, environment/lineage, history/events, best/last, selected validation predictions, panel diagnostik NPZ+JSON, metrics, recomputation witness, report dan completion marker. Validator menghitung ulang metrik dari raw records dan QA reference, memeriksa nilai yang dilaporkan serta hash artefak. `analysis_ready` adalah kelengkapan/rekonsiliasi artefak, **bukan** keberhasilan hipotesis. Run nyata di bawah `docs/report_training/` menambah INDEX; fixture berada pada temporary directory.

Resume memakai `--resume RUN/last.pt` dan output baru. Pertahankan config yang memengaruhi eksperimen. M/C/probe melanjutkan checkpoint pada batas epoch yang tersimpan; perubahan urutan, source, budget atau normalizer ditolak. Determinisme bitwise yang diuji di CPU tidak menjamin identik lintas GPU/backend.

## 7. Profiling RTX 3060 12 GB dan gate checkpoint

Optimasi terbaru tidak menyimpan seluruh LMDB atau H/U ke RAM. Metadata frame/H/U/QA dibaca lazy, index ID berupa array ringkas yang dapat dibagi pada Windows spawn, target berupa memmap, dan satu file tensor dibaca sekali untuk verifikasi SHA sekaligus deserialisasi. RAM yang tersedia untuk filesystem cache dikelola sistem operasi. Ukuran arsitektur dan input RPC tetap mengikuti planning.

Semua sebelas YAML sekarang memasang `ram_reserve_mib: 2048`, `vram_reserve_mib: 1024`, `resource_poll_ms: 250`. Monitor berlaku pada training E/M/C/probe/L; jika reserve terlanggar atau pembacaan resource gagal, run gagal melalui mekanisme logger, sehingga tidak mendapat status complete. Sampling berkala dapat melewatkan lonjakan di antara poll; CUDA juga melaporkan peak allocator. RSS proses+worker adalah jumlah accounting yang dapat menghitung halaman bersama lebih dari sekali; batas RAM memakai `available` sistem, bukan menjumlahkan RSS sebagai penggunaan fisik. Headroom CUDA = free device + reserved allocator yang belum allocated. Ini mempertahankan cache PyTorch yang dapat dipakai ulang.

Screening E berikut menaikkan batch secara nyata untuk menemukan throughput terbaik. Config `E.yaml` terlebih dahulu harus berisi paths audit/target aktual dari §0. Tidak ada learning-rate scaling otomatis. `--effective-batches` dan `--effective-batch` saling eksklusif; jika keduanya tidak diberikan, produk batch dari config dipertahankan.

```powershell
& $py -m eksperimen_model.profile_m4human_encoder --config E.yaml --output "$meta/profile_E_sweep_$stamp.json" --selected-config E_selected.yaml --device cuda:0 --effective-batches 16 64 128 256 512 1024 --micro-batches 4 8 16 32 64 128 256 512 1024 --workers 0 2 4 --precisions fp32 fp16_amp_with_grad_scaler --warmup-steps 5 --timed-steps 10 --ram-reserve-mib 2048 --reserve-mib 1024
```

Kandidat yang OOM, gagal parity, timed overflow, atau melanggar reserve tidak direkomendasikan. `E_selected.yaml` hanya dibuat jika ada pilihan aman; file output harus baru. Periksa report sebelum training. Setelah pilot memilih protokol yang layak, ulangi hanya kombinasi terpilih minimal100 timed updates seperti contoh berikutnya.

Profil downstream juga tersedia; isi lineage, paths, successful-update budget dan scientific gates sesuai tahap sebelum mengeksekusi. Gunakan config lengkap baru untuk setiap hasil. Command tidak membuat checkpoint ilmiah atau membuka validation/test.

```powershell
& $py -m eksperimen_model.profile_m4human_stages --stage M --config M.yaml --output "$meta/profile_M_$stamp.json" --selected-config M_selected.yaml --device cuda:0 --micro-batches 4 8 16 32 --workers 0 2 4 --precisions fp32 fp16_amp --warmup-steps 5 --timed-steps 10
& $py -m eksperimen_model.profile_m4human_stages --stage C --condition C_base --config C.yaml --output "$meta/profile_Cbase_$stamp.json" --device cuda:0 --micro-batches 1 2 4 8 16 --workers 0 2 4 --precisions fp32 fp16_amp --warmup-steps 5 --timed-steps 10
& $py -m eksperimen_model.profile_m4human_stages --stage C --condition C_kin --config C.yaml --output "$meta/profile_Ckin_$stamp.json" --device cuda:0 --micro-batches 1 2 4 8 16 --workers 0 2 4 --precisions fp32 fp16_amp --warmup-steps 5 --timed-steps 10
& $py -m eksperimen_model.profile_m4human_stages --stage probe --condition U_base --config probe.yaml --output "$meta/profile_probe_$stamp.json" --device cuda:0 --micro-batches 1 2 4 8 16 --workers 0 2 4 --precisions fp32 fp16_amp --warmup-steps 5 --timed-steps 10
& $py -m eksperimen_model.profile_m4human_stages --stage L --condition C_base --config L.yaml --qa "$meta/qa.jsonl" --u-cache "$meta/U_base" --initial "$meta/projector_initial.pt" --output "$meta/profile_Lbase_$stamp.json" --device cuda:0 --micro-batches 1 2 4 8 16 --precisions fp16_amp_grad_scaler bf16 --generation-batches 1 2 4 8 16 --warmup-steps 5 --timed-steps 10
```

Untuk probe, profile `M`, `U_base`, `U_kin`; untuk L ulangi `C_kin` dengan exact-U cache kondisi tersebut dan shared initial projector yang sama. Pilih kombinasi yang aman pada **kedua** kondisi C/L dan semua probe dibandingkan, lalu pakai konfigurasi/budget yang matched. Jangan mengambil rekomendasi tercepat masing-masing secara terpisah jika setting perlakuannya berbeda. Perubahan paired config membutuhkan contract baru dan kedua kondisi dilatih ulang; profiler menandai regenerasi contract, tanpa mengesahkan gates. L memerlukan revision pretrained Qwen, lineage, dan shared initial yang dipin. `generation.batch_size` dari profil harus sama pada kedua kondisi. Model tetap frozen; SDPA dan checkpoint wrapper eksplisit mendukung microbatch lebih besar. Worker L tetap0 karena QA memakai loop langsung.

Kandidat terbaru: E128×1, M32×1, C/probe16×1, L4×4 serta generation8. Angka ini belum diukur pada RTX3060. M/C/probe/L mempertahankan effective batch sebelumnya; E128 tetap alternatif protokol terhadap baseline16. Profiler dapat menemukan pilihan lebih kecil atau lebih besar dari kandidat tersebut.

Seluruh perintah PowerShell berikut dijalankan dari root repository setelah audit dan path konfigurasi diselesaikan. `E.yaml` adalah config run aktual, bukan template dengan null. Jangan menimpa output lama.

```powershell
$py = ".venv\Scripts\python.exe"
$env:PYTHONDONTWRITEBYTECODE = "1"
# Baseline E: batch efektif 4 x 4 = 16; sweep hanya pembagi 16.
& $py -m eksperimen_model.profile_m4human_encoder --config E.yaml --output D:/m4human_meta/profile_E16.json --device cuda:0 --micro-batches 4 8 16 --workers 0 2 4 --precisions fp32 fp16_amp_with_grad_scaler --warmup-steps 5 --timed-steps 10 --reserve-mib 1024
# Kandidat protokol baru E128: batch efektif 128, bukan optimasi diam-diam E16.
& $py -m eksperimen_model.profile_m4human_encoder --config E128.yaml --output D:/m4human_meta/profile_E128.json --device cuda:0 --effective-batch 128 --micro-batches 16 32 64 128 --workers 0 2 4 --precisions fp32 fp16_amp_with_grad_scaler --warmup-steps 5 --timed-steps 10 --reserve-mib 1024
```

Untuk `E128.yaml`, salin kandidat `eksperimen_model/configs/m4human_encoder_rtx3060_candidate.yaml`, isi audit paths, lalu catat alasan perubahan batch efektif 16→128 dan pin protokol baru melalui pilot train/val. Learning rate tidak diskalakan otomatis. Report tidak menjamin kandidat terbesar tercepat atau muat: rekomendasi hanya kandidat dengan parity forward/loss/backward terhadap fp32, tanpa timed overflow, serta reserve memori terpenuhi. Report CPU/sintetis tidak memenuhi gate ilmiah CUDA. Profiler hanya membaca train dan fitting normalizer train dilakukan di luar timer; warmup tidak termasuk timed updates.

Sweep10 timed updates di atas hanya screening, tidak cukup untuk readiness final. Setelah memilih setting, simpan config lengkap sebagai `E_selected.yaml`, lalu reprofile **hanya** setting terpilih minimal100 timed updates. Contoh berikut memakai microbatch8/accumulation2 (E16), worker2, AMP; ganti tiga nilai dengan pilihan aktual dan pertahankan produk batch yang dipin:

Bagian `training` harus memuat seluruh `recommended_settings` dari screening, termasuk effective_batch dan persistent_workers yang diresolusikan profiler. Tetapkan paths, policy, field lain dan optional `scientific_gate_predeclared` sebelum profil100. Readiness hash masih mengikat flag pilot ini; mengubahnya setelah profil akan membuat evidence stale. Workflow diagnostic pilot → final boleh menghilangkan flag tersebut.

```powershell
& $py -m eksperimen_model.profile_m4human_encoder --config E_selected.yaml --output D:/m4human_meta/profile_E_selected100.json --device cuda:0 --micro-batches 8 --workers 2 --precisions fp16_amp_with_grad_scaler --warmup-steps 5 --timed-steps 100 --reserve-mib 1024
```

Run ini memerlukan setidaknya (5+100)×effective_batch sampel train, tidak membuka validation/test. Hanya report baru100+ timed updates dengan `successful_updates==timed_steps`, `samples==timed_steps×micro_batch×accumulation`, nol timed overflow dan parity/memory-safe yang memenuhi resource gate; warmup skips dicatat tetapi tidak dihitung sebagai timed updates. Nilai median/p95 berasal timed successful update durations jika tersedia, bukan estimasi laptop.

Pilih config kandidat **sebelum** profiler, atau buat config baru dengan seluruh `recommended_settings` lalu ulangi profiler pada config itu. `readiness_config_hash` mengikat config terpilih; `recommended_config_hash` adalah full canonical config dan tidak menggantikannya. Jika policy/config berubah, profil ulang. E smoke boleh mempunyai setting performa berbeda, tetapi sensor/source policy dan lineage schema, source, target, joint, coordinate, split serta recipe harus cocok dengan run terpilih. Laptop tidak menghasilkan klaim measured VRAM/throughput RTX 3060.

| Tahap | Batch efektif baseline | Kebijakan kandidat |
|---|---:|---|
| E | 16 | Kandidat 128 adalah protokol baru; pilot dan budget/exposure dicatat |
| M | 32 | Ubah microbatch/accumulation dengan produk tetap 32 |
| C_base/C_kin | 16 | Produk tetap 16 dan kedua treatment matched |
| Probe M/U_base/U_kin | 16 | Produk tetap 16, initialization/order/budget matched |
| L C_base/C_kin | 16 | Produk tetap 16; backend/checkpoint/precision sama |

Loader mendukung workers, spawn, pinning, nonblocking transfer, persistent workers dan bounded prefetch. Ukur worker0/2/4 pada perangkat tujuan; `persistent_workers` hanya untuk workers>0. Reader LMDB dan memmap dibuka per proses; epoch augmentation diteruskan ke persistent workers. Evaluasi scalar tidak wajib mengalokasikan seluruh prediction records; artefak terpilih tetap diekspor. `eval_batch_size` dan extraction batch/device adalah setting yang harus diprofilkan, bukan angka latency yang sudah diketahui. Panduan resmi: [DataLoader PyTorch 2.6](https://docs.pytorch.org/docs/2.6/data.html), [tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).

AMP memakai autocast untuk forward dan fp32 untuk loss fisik/reduksi, GradScaler unscale sebelum clipping, dan pemeriksaan gradien seluruh optimizer sebelum step. Overflow tidak dihitung sebagai successful update; retry terbatas pada batch akumulasi yang sama, lalu gagal jika tidak pulih. Scheduler/exposure budget mengikuti successful updates. Clipping C untuk compressor+fidelity dan auxiliary tetap terpisah. Loop C mengisi named telemetry `compressor_fidelity` dan `auxiliary` before/after clipping melalui `norms_out`; scalar maximum return helper sendiri tidak merepresentasikan setiap norm grup. Tahap lain hanya melaporkan field yang tersedia. [AMP PyTorch 2.6](https://docs.pytorch.org/docs/2.6/amp.html).

Loader workers/prefetch pada kandidat L tidak dipakai loop QA direct microbatch saat ini; jangan melaporkan manfaat DataLoader workers untuk projector. Optimasi L aktual ialah microbatch/accumulation, SDPA, activation checkpoint dan generation batch. Profiler L menolak workers>0. Numerical witness L memakai satu QA train fp32 dengan model reference dilepas sebelum model timed dibuat; batas ini eksplisit pada report dan tidak membuktikan ekuivalensi semua ukuran batch atau semua panjang QA. Parity dan generated validation aktual masih harus diperiksa pada komputer training.

Pada L, `llm.attention_backend: sdpa` memakai SDPA native PyTorch tanpa ekstensi FlashAttention. `training.activation_checkpointing: true` mengaktifkan wrapper checkpoint eksplisit pada forward frozen LLM; Qwen tetap eval dan bobot frozen, sementara gradient input diteruskan ke projector. Toggle Hugging Face yang hanya bekerja saat LLM training bukan bukti checkpointing aktif pada frozen eval. Bandingkan loss/gradient dan generation terhadap eager tanpa checkpoint; gunakan setting identik pada dua kondisi.

Tiny-overfit train-only tersedia sebagai diagnostic terpisah:

```powershell
& $py -m eksperimen_model.check_m4human_encoder_overfit --config E.yaml --output D:/m4human_meta/overfit_E_new.json --device cuda:0 --samples 16 --updates 200 --loss-ratio-max 0.25
```

Command membaca fixed train subset, fitting normalizer train dan membandingkan initial/final fp32 loss serta forward/loss/backward parity AMP jika precision AMP dipilih. Pada diagnosis awal §0 gunakan fp32. Diagnostic effective batch=16; ini bukan budget final E atau checkpoint scientific. `--loss-ratio-max` dipilih sebelum diagnostic sesuai kebutuhan train/val; nilai0.25 default bukan quality threshold ilmiah universal. Report tidak menggantikan real-smoke/resource readiness artifact atau root freeze gate. Setelah smoke, tiny-overfit, pilot dan profiling actual selesai, tambahkan mapping **file dan SHA256 bytes** ke config terpilih:

```powershell
(Get-FileHash -Algorithm SHA256 "E_SMOKE_RUN/metrics.json").Hash.ToLower()
(Get-FileHash -Algorithm SHA256 "D:/m4human_meta/profile_E_selected100.json").Hash.ToLower()
```

```yaml
real_smoke_passed:
  artifact_path: E_SMOKE_RUN/metrics.json
  artifact_sha256: REPLACE_WITH_ACTUAL_FILE_SHA256
resource_profile_passed:
  artifact_path: D:/m4human_meta/profile_E_selected100.json
  artifact_sha256: REPLACE_WITH_ACTUAL_FILE_SHA256
training:
  root_quality_gate_m: null # ganti threshold train/val yang dideklarasikan sebelum selected run
```

Cuplikan YAML hanya menunjukkan field yang digabung ke config lengkap; jangan mengganti seluruh `training` dengan cuplikan ini. Untuk pilot ilmiah, `scientific_gate_predeclared:true` harus sudah ditetapkan sebelum profil dan run; final tidak memerlukan flag pilot tersebut. Smoke baru jangan menunjuk bukti smoke/profil yang belum ada. Boolean `true` sebagai evidence atau SHA yang dikarang ditolak. Real smoke harus run E m4human complete dengan bundle tervalidasi, successful updates dan root support nonzero, nilai finite serta source/sensor lineage cocok. Resource report harus m4human, measured CUDA, matching recommended settings, parity `cuda_measured:true`, minimal100 complete timed updates dengan exact exposure, nol timed overflow dan reserved+reserve≤budget. `resource_safety` harus measured, CUDA terukur, minima RAMavailable/CUDAheadroom finite dan memenuhi reserve config. Hash normalizer profil harus cocok dengan normalizer aktual run; config tanpa normalizer yang dikunci difit dari train oleh profiler dan trainer. Semua artefak dicek bytes hash ketika spec gate dibangun.

```powershell
& $py -m eksperimen_model.train_m4human_encoder --config E_selected.yaml --output-dir docs/report_training/RUN_M4HUMAN_TIMESTAMP_E_FINAL --mode final --device cuda:0
& $py -m eksperimen_model.extract_m4human_states --config E_selected.yaml --checkpoint docs/report_training/RUN_M4HUMAN_TIMESTAMP_E_FINAL/best.pt --output-dir D:/m4human_meta/states --device cuda:0 --batch-size 4
```

Ekstraksi E dibatch fp32 (`--batch-size`, default `training.eval_batch_size`) meski training AMP. Ekstraksi H/U memakai `device` dan `eval_batch_size` dari config M/C; tidak ada CLI `--device`/`--batch-size` pada kedua entry point tersebut. Pilih batch ekstraksi dari measured memory profile.

Selected best checkpoint menyimpan mode, full resolved config hash, threshold, bukti readiness yang telah diverifikasi dan keputusan root gate. Smoke tidak pernah scientific; pilot hanya dapat memenuhi gate jika predeclared dan readiness lengkap. Mengubah threshold/config ekstraksi tidak mempromosikan checkpoint. Ekstraksi memverifikasi keputusan metadata checkpoint tanpa membuka kembali readiness files atau annotation arrays; `--allow-debug` pada checkpoint legacy/smoke menghasilkan cache non-scientific eksplisit.

## 8. Bundle, newline, dan bukti historis

```powershell
& $py -m eksperimen_model.validate_m4human_run E_RUN
& $py -m eksperimen_model.package_m4human_run --run-dir E_RUN --output D:/m4human_meta/E_RUN.zip --audit-metadata D:/m4human_meta/schema.json --audit-metadata D:/m4human_meta/coordinate_audit.json
```

Packaging memvalidasi run sebelum ZIP dan sesudah ekstraksi sementara, menyimpan best/last, predictions, witness, diagnostic arrays serta checksum bytes. Dataset/raw target arrays tidak dibundel default; audit metadata tambahan terbatas file teks kecil. QA reference harus run-local relative supaya bundle portable. Validasi integritas bukan validasi hipotesis atau pengganti audit semantik dataset.

Completion baru mendeklarasikan `text_hash_algorithm:utf8_lf_sha256` untuk metrics/history UTF-8, hanya menormalkan CRLF→LF. Marker lama menerima hash bytes LF/CRLF yang ekuivalen, bukan perubahan isi. Hash checkpoint/cache/source lineage tetap exact bytes melalui `file_sha256`. Windows `process_current_rss_bytes` berbeda dari `process_peak_rss_bytes` (`peak_wset` jika tersedia); peak yang tidak tersedia null. CUDA peak menyebut device.

`RUN_M4HUMAN_SMOKE` historis tetap tidak diubah. Bundle yang saat ini kehilangan checkpoint/predictions tidak lengkap walaupun summary/metrics tersedia; kompatibilitas newline tidak menciptakan file yang hilang atau bukti CUDA/data nyata. Package command menolak bundle tidak lengkap. Jangan merekonstruksi hasil atau model dari angka laporan.

## Gerbang yang masih harus dijalankan di komputer dataset

- Serializer/byte order, missing/sentinel, frame/recording/time, subject split, koordinat dan 22 joint pada paket nyata.
- SMPL-X asset/regressor, gender/defaults, unit/kalibrasi dan ekuivalensi direct-joints bila dipakai.
- Smoke, tiny-overfit dan kualitas E/M/probe pada train/val aktual.
- Pretrained Qwen/tokenizer yang dipin, numerical parity AMP, VRAM/RAM/latensi RTX 3060 dan budget yang layak.
- Seluruh training ilmiah, held-out evaluation dan interval kepercayaan hasil nyata.

Nilai `null`, `false` dan `not_run` pada template sengaja dipertahankan sampai ada bukti. Jangan menyebut hasil sintetis sebagai validasi ilmiah M4Human.
