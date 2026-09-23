# Runbook pemulihan Tahap 4

Jalankan dari root repositori di **PowerShell Windows** pada mesin eksperimen memakai `.venv`. Checkout ini belum memuat fitur, GT raw, checkpoint Av2/Dynamics, atau hasil QA penuh; tidak ada angka empiris yang dapat diklaim di sini. Simpan seluruh hasil Batch 1. Perintah berikut memakai direktori fitur dan QA pemulihan, serta folder run baru untuk checkpoint/laporan.

## 0. Prasyarat, sanity check, dan jalur keluaran

```powershell
$py = ".venv\Scripts\python.exe"
$run = Join-Path "docs\report_training" ("stage4_recovery_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
$featuresDir = "datasets/MM-Fi_features_stage4_recovery"
$qaDir = "datasets/MM-Fi_grounded_qa_stage4_recovery"
$dynamicsDir = Join-Path $run "dynamics"
$dynamicsCheckpoint = Join-Path $dynamicsDir "best_dynamics_model.pth"
$dynamicsConfig = Join-Path $run "mmfi_dynamics_stage4_recovery.yaml"
$projectorConfig = Join-Path $run "mmfi_projector_stage4_recovery.yaml"
if ((Test-Path $featuresDir) -or (Test-Path $qaDir)) { throw "Arsipkan direktori recovery lama sebelum run baru." }
New-Item -ItemType Directory -Force -Path $run | Out-Null
$required = @("datasets\MM-Fi Dataset\filtered_mmwave", "datasets\MM-Fi Dataset\MMFi_action_segments.csv", "eksperimen_model\checkpoints\pose_estimation_v2\model_av2.pth")
foreach ($path in $required) { if (-not (Test-Path $path)) { throw "Artefak wajib tidak ada: $path" } }
& $py eksperimen_model/test_pipeline.py
```

**Gerbang:** semua artefak wajib tersedia dan sanity check lulus sebelum training panjang. Simpan checksum Av2, versi kode, seed, dan config di `$run`.

## 1. Ekstraksi fitur ke direktori pemulihan

```powershell
& $py eksperimen_model/datasets/extract_physical_features.py --config eksperimen_model/configs/mmfi_dynamics_v3.yaml --output_dir $featuresDir --seed 42
```

Periksa `$featuresDir/extraction_rejections.json`, `normalization_stats.pt`, dan jumlah berkas train/val/test. Ekstraktor menolak radar/GT tidak sah, menyimpan ID frame asli serta provenance, dan membangun statistik hanya dari berkas train yang tervalidasi. `--dry_run` dan `--max_actions` tidak membangun statistik penuh. Jika run harus diulang, arsipkan direktori recovery lebih dulu; jangan menimpa cache atau laporan Batch 1.

**Gerbang:** penyebab seluruh rekaman ditolak diaudit. Pastikan fitur memiliki hash encoder/config/sumber dan statistik memiliki `feature_provenance` serta `source_manifest_sha256` yang sesuai. Tidak ada cache lawas tercampur.

## 2. Manifest QA dan pembekuan target

```powershell
& $py eksperimen_model/datasets/generate_grounded_qa.py --features_dir $featuresDir --output_dir $qaDir --segments_csv "datasets/MM-Fi Dataset/MMFi_action_segments.csv" --raw_dataset_dir "datasets/MM-Fi Dataset/filtered_mmwave" --train_stride 4 --val_stride 8 --test_stride 8 --seed 42
Get-FileHash (Join-Path $qaDir "mmfi_grounded_qa_test.jsonl") -Algorithm SHA256
```

Periksa `$qaDir/qa_dataset_summary.json` dan tiga `*_rejections.jsonl`. Target saat ini `current_wrist_separation` memiliki kelas `narrower/wider`; target perubahan sampai t+8 `future_wrist_separation_change` memiliki kelas `closing/stable/opening`. Keduanya memakai jarak 3D relatif terhadap lebar bahu saat t, tanpa besaran absolut meter. Jendela lintas repetisi ditandai; rentang CSV terbalik dikarantina.

**Gerbang:** tinjau train/val per kelas, subjek, aksi, dan dekat ambang; periksa 16+8 frame berurutan, GT cocok dengan raw, dan subjek antarsplit terpisah. Tentukan ambang efek minimum, cakupan parse minimum, target primer, dan checksum manifest test **sebelum** menilai test. Jika label masa depan nyaris seluruhnya `stable`, hentikan klaim manfaat forecast.

## 3. Kecocokan Dynamics dengan fitur recovery

Checkpoint Dynamics lama tidak mencatat hash fitur maupun statistik normalisasi Tahap 3. Kesamaan path config saja tidak membuktikan bahwa fitur recovery cocok dengan input latihnya. Kode B4 saat ini menolak checkpoint tanpa lineage; jalur yang sudah diimplementasikan adalah **melatih ulang Dynamics** pada fitur recovery, tanpa menimpa checkpoint lama. Pemakaian checkpoint lama akan memerlukan prosedur audit dan migrasi lineage terpisah yang belum tersedia.

```powershell
& $py -c 'import sys,yaml; from pathlib import Path; src,dst,features=sys.argv[1:]; cfg=yaml.safe_load(Path(src).read_text(encoding="utf-8")); cfg["dataset"]["features_output_dir"]=features; Path(dst).write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")' "eksperimen_model/configs/mmfi_dynamics_best_tuned.yaml" $dynamicsConfig $featuresDir
& $py eksperimen_model/train_dynamics.py --config $dynamicsConfig --epochs 150 --batch_size 64 --output_dir $dynamicsDir
& $py eksperimen_model/evaluate_dynamics.py --config $dynamicsConfig --checkpoint $dynamicsCheckpoint --split val --batch_size 64 --output_dir (Join-Path $run "dynamics_val") --output_json (Join-Path $run "dynamics_val/dynamics_benchmark_results.json")
```

**Gerbang:** checkpoint baru ada, input memakai statistik train recovery, dan prediksi latent val dibanding persistensi pada horizon yang sama. Jika Dynamics gagal mengungguli persistensi untuk target relevan, perbaiki Tahap 3 sebelum B4. Simpan test Dynamics sampai keputusan target dan ambang dibekukan.

## 4. Probe B2 dan config Stage 4 yang cocok

```powershell
& $py eksperimen_model/train_probe.py --qa_dir $qaDir --features_dir $featuresDir --epochs 30 --batch_size 128 --seed 42 --output_dir (Join-Path $run "probe")
& $py -c 'import sys,yaml; from pathlib import Path; src,dst,features,qa,dynamics,dynamics_cfg=sys.argv[1:]; cfg=yaml.safe_load(Path(src).read_text(encoding="utf-8")); ds=cfg["dataset"]; ds.update(qa_dir=qa,features_dir=features,dynamics_checkpoint=dynamics,dynamics_config=dynamics_cfg,train_file=qa+"/mmfi_grounded_qa_train.jsonl",val_file=qa+"/mmfi_grounded_qa_val.jsonl",test_file=qa+"/mmfi_grounded_qa_test.jsonl"); Path(dst).write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")' "eksperimen_model/configs/mmfi_projector_qwen.yaml" $projectorConfig $featuresDir $qaDir $dynamicsCheckpoint $dynamicsConfig
& $py eksperimen_model/test_projector_pipeline.py
```

Periksa `$run/probe/best_probe_model.pth` dan `probe_benchmark_metrics.json` yang pada tahap ini hanya berisi metrik train/val. **Gerbang:** pada val, target terpilih terbaca dari z/riwayat dan melampaui baseline kelas mayoritas. Test B2 baru dihitung oleh evaluator pada langkah 6, setelah target dan ambang dibekukan. Pastikan salinan config projector menunjuk fitur, QA, statistik, dan Dynamics recovery sebelum training.

## 5. Projector B3, B3P, B4

```powershell
& $py eksperimen_model/train_projector.py --config $projectorConfig --condition B3 --output_dir (Join-Path $run "projector/b3")
& $py eksperimen_model/train_projector.py --config $projectorConfig --condition B3P --output_dir (Join-Path $run "projector/b3p")
& $py eksperimen_model/train_projector.py --config $projectorConfig --condition B4 --output_dir (Join-Path $run "projector/b4")
```

Ketiganya memakai 16 latent riwayat identik. B3P menambah delapan salinan state terakhir; B4 menambah delapan forecast dari Dynamics recovery beku. SLM, encoder, dan Dynamics tetap beku. **Gerbang:** ketiga `best_projector.pth` serta riwayat training ada, provenance cocok, dan checkpoint dipilih dengan metrik jawaban JSON pada val serta cakupan parse memadai. B5, bila diperlukan, harus memakai projector terlatih terpisah.

## 6. Evaluasi test beku dan laporan

```powershell
& $py eksperimen_model/evaluate_dynamics.py --config $dynamicsConfig --checkpoint $dynamicsCheckpoint --split test --batch_size 64 --output_dir (Join-Path $run "dynamics_test") --output_json (Join-Path $run "dynamics_test/dynamics_benchmark_results.json")
& $py eksperimen_model/evaluate_reasoning.py --config $projectorConfig --dynamics_checkpoint $dynamicsCheckpoint --dynamics_config $dynamicsConfig --checkpoint_b2 (Join-Path $run "probe/best_probe_model.pth") --checkpoint_b3 (Join-Path $run "projector/b3/best_projector.pth") --checkpoint_b3p (Join-Path $run "projector/b3p/best_projector.pth") --checkpoint_b4 (Join-Path $run "projector/b4/best_projector.pth") --output_json (Join-Path $run "stage4_reasoning_benchmark.json") --output_report (Join-Path $run "stage4_reasoning_benchmark.md")
```

Simpan respons mentah, status parse, sample ID, target/prediksi, checksum manifest, config, dan metrik dalam `$run`. Laporkan jumlah total/valid/terparse, distribusi kelas, accuracy dan macro-F1 per target/subjek, serta strata dalam/lintas repetisi. Perbandingan primer: **macro-F1 target masa depan B4 − B3P** pada sampel identik dengan interval berkelompok pada rekaman atau subjek. Shuffling adalah analisis sekunder. Angka Batch 1 tetap berstatus tidak konklusif untuk kontribusi Dynamics sampai pemulihan ini selesai dan metriknya dihitung.
