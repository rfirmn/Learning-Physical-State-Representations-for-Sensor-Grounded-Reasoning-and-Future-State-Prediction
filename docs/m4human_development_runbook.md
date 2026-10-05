# Runbook development M4Human v3

Implementasi mengikuti lima dokumen `docs/planning_m4human/` dan proposal terbaru. Kontraknya `m4human_kinetok_v3`: E → M → compressor C → probe independen → projector ke Qwen beku. Ini implementasi P0 untuk membaca kinematika **jendela observasi**; bukan bukti forecasting atau reasoning counterfactual. Pipeline dan laporan MM-Fi tetap dipertahankan.

## 1. Verifikasi pada laptop tanpa dataset

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

Salin konfigurasi referensi ke konfigurasi run, isi seluruh path/hash/null berdasarkan audit. `joint_map_hash`, schema dan coordinate hash memakai SHA-256 **bytes file**; `recipe_hash` memakai canonical JSON dari recipe. Ambil lineage dari metadata artefak upstream agar tidak mencampur dua definisi hash.

```sh
.venv/bin/python -m eksperimen_model.train_m4human_encoder --config E.yaml --output-dir docs/report_training/RUN_M4HUMAN_TIMESTAMP_E_SMOKE --mode smoke --device cuda --precision fp32
.venv/bin/python -m eksperimen_model.train_m4human_encoder --config E.yaml --output-dir docs/report_training/RUN_M4HUMAN_TIMESTAMP_E_PILOT --mode pilot --device cuda
```

Selesaikan real smoke, tiny-overfit, target/root-quality gate dan profil resource sebelum `--mode final`. Pin threshold root dari train/val; konfigurasi final memerlukan `real_smoke_passed` dan `resource_profile_passed`. Tidak ada angka performa target yang ditebak dari fixture. Encoder diinisialisasi dari scratch sesuai P0, tidak memuat checkpoint MM-Fi.

```sh
.venv/bin/python -m eksperimen_model.extract_m4human_states --config E.yaml --checkpoint E_RUN/best.pt --output-dir D/states --device cuda
.venv/bin/python -m eksperimen_model.train_m4human_motion --config M.yaml
.venv/bin/python -m eksperimen_model.extract_m4human_motion --config M.yaml --checkpoint M_RUN/M/best.pt --output D/H
```

Tidak memberikan `--split` saat ekstraksi menyimpan seluruh partition yang sudah ditetapkan; training tetap hanya membaca train/val. Ekstraksi E memakai RPC dan checkpoint beku tanpa membuka target/params. Untuk cache H berlabel offline, `M.yaml` menunjuk target cache; training loss membutuhkan target terpisah. Loader H dan ekstraksi U untuk inferensi tidak memerlukan target tersedia.

`M.yaml` memerlukan `max_gap_s`, `successful_updates`, lineage, joint map dan recipe yang diselesaikan. Normalizer dihitung pada train; empat kelompok loss fisik memiliki penyebut terpisah. `precision: fp16_amp` membutuhkan CUDA dan numeric gate. Mulai dengan fp32 saat smoke jika gate belum tersedia.

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

Ganti revision Qwen yang null dengan full commit yang dipin. `--initialize` hanya menyimpan shared projector dan kalibrasi alpha train-only; argumen output tetap diminta parser tetapi run training belum dibuat. Kedua kondisi harus memuat initial yang sama. Hanya projector dilatih. Frozen Qwen tetap meneruskan gradien input ke projector. Generated validation dan parse coverage memilih checkpoint; jawaban teacher-forced bukan metrik reasoning.

QA memakai dua core task. Target undefined dikeluarkan dengan alasan; unknown tetap kategori jawaban sah. Derivative memakai tujuh sampel trailing, timestamp aktual dan reset warmup enam frame pada setiap jendela 32. Optional onset/radial tidak masuk generator P0 sebelum reliability gate tersendiri.

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

## Gerbang yang masih harus dijalankan di komputer dataset

- Serializer/byte order, missing/sentinel, frame/recording/time, subject split, koordinat dan 22 joint pada paket nyata.
- SMPL-X asset/regressor, gender/defaults, unit/kalibrasi dan ekuivalensi direct-joints bila dipakai.
- Smoke, tiny-overfit dan kualitas E/M/probe pada train/val aktual.
- Pretrained Qwen/tokenizer yang dipin, numerical parity AMP, VRAM/RAM/latensi RTX 3060 dan budget yang layak.
- Seluruh training ilmiah, held-out evaluation dan interval kepercayaan hasil nyata.

Nilai `null`, `false` dan `not_run` pada template sengaja dipertahankan sampai ada bukti. Jangan menyebut hasil sintetis sebagai validasi ilmiah M4Human.
