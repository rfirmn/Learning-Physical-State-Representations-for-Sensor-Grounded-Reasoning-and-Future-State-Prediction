# Runbook Pemulihan Tahap 4 — PowerShell Windows

Jalankan dari PowerShell pada mesin eksperimen Windows.

## Satu script untuk setiap tahap

Setiap tahap sudah memiliki script mandiri. Jalankan dari root repositori dan
tempel satu perintah sesuai tahap yang ingin dijalankan:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stage4_sanity_check.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\stage4_extract_features.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\stage4_generate_qa.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\stage4_train_dynamics.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\stage4_evaluate_dynamics.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\stage4_train_probe.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\stage4_sanity_projector.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\stage4_train_projector.ps1 -Condition B3
powershell -ExecutionPolicy Bypass -File .\scripts\stage4_train_projector.ps1 -Condition B3P
powershell -ExecutionPolicy Bypass -File .\scripts\stage4_train_projector.ps1 -Condition B4
powershell -ExecutionPolicy Bypass -File .\scripts\stage4_evaluate.ps1
```

Jalankan sesuai urutan. Setiap script berhenti saat menemukan error dan tidak
menimpa output recovery yang sudah ada.

> Batch 1 tidak diubah. Semua fitur dan QA pemulihan memakai direktori baru.
> Checkpoint Dynamics lama tidak memiliki lineage yang cukup untuk B4, sehingga
> Dynamics dilatih ulang menggunakan fitur pemulihan.

## 1. Persiapan dan sanity check

Pastikan terminal PowerShell sedang berada di root repositori. Salin Blok 1,
tekan Enter, lalu salin Blok 2.

### Blok 1 — persiapan dan pemeriksaan artefak

```powershell
$Py = ".venv\Scripts\python.exe"
$Run = Join-Path "docs\report_training" ("stage4_recovery_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
$Features = "datasets\MM-Fi_features_stage4_recovery"
$QA = "datasets\MM-Fi_grounded_qa_stage4_recovery"
$DynamicsDir = "eksperimen_model\checkpoints\dynamics_stage4_recovery"
$DynamicsCheckpoint = Join-Path $DynamicsDir "best_dynamics_model.pth"
$ProbeDir = Join-Path $Run "probe"
$ProjectorDir = Join-Path $Run "projector"

if (-not (Test-Path $Py)) { throw "Python .venv tidak ditemukan. Jalankan dari root repositori." }
if ((Test-Path $Features) -or (Test-Path $QA)) { throw "Direktori recovery lama sudah ada. Arsipkan atau gunakan nama direktori lain sebelum memulai." }
if (Test-Path $DynamicsDir) { throw "Checkpoint Dynamics recovery lama sudah ada. Arsipkan sebelum memulai." }

$Required = @(
    "datasets\MM-Fi Dataset\filtered_mmwave",
    "datasets\MM-Fi Dataset\MMFi_action_segments.csv",
    "eksperimen_model\checkpoints\pose_estimation_v2\model_av2.pth"
)
foreach ($Path in $Required) {
    if (-not (Test-Path $Path)) { throw "Artefak wajib tidak ditemukan: $Path" }
}

New-Item -ItemType Directory -Force -Path $Run | Out-Null
```

### Blok 2 — jalankan sanity check

```powershell
& $Py eksperimen_model\test_pipeline.py
if ($LASTEXITCODE -ne 0) {
    throw "Sanity check gagal dengan exit code $LASTEXITCODE. Hentikan proses dan perbaiki error sebelum training."
}
Write-Host "Sanity check lulus. Anda dapat melanjutkan ke ekstraksi fitur." -ForegroundColor Green
```

Blok 2 menggunakan variabel yang dibuat oleh Blok 1. Jangan lanjut ke tahap
berikutnya bila blok ini gagal.

## 2. Ekstraksi fitur recovery

```powershell
& $Py eksperimen_model\datasets\extract_physical_features.py `
    --config eksperimen_model\configs\mmfi_dynamics_v3.yaml `
    --output_dir $Features `
    --seed 42
```

Periksa hasilnya:

```powershell
Get-ChildItem $Features -Recurse -Filter *.pt | Measure-Object
Get-Content (Join-Path $Features "extraction_rejections.json") | Select-Object -First 20
if (-not (Test-Path (Join-Path $Features "normalization_stats.pt"))) {
    throw "normalization_stats.pt tidak dibuat. Jangan lanjut."
}
```

Jangan lanjut bila `normalization_stats.pt` tidak ada, atau alasan penolakan
rekaman belum ditinjau.

## 3. Bangun manifest QA yang tervalidasi

```powershell
& $Py eksperimen_model\datasets\generate_grounded_qa.py `
    --features_dir $Features `
    --output_dir $QA `
    --segments_csv "datasets\MM-Fi Dataset\MMFi_action_segments.csv" `
    --raw_dataset_dir "datasets\MM-Fi Dataset\filtered_mmwave" `
    --train_stride 4 `
    --val_stride 8 `
    --test_stride 8 `
    --seed 42

Get-Content (Join-Path $QA "qa_dataset_summary.json")
Get-FileHash (Join-Path $QA "mmfi_grounded_qa_test.jsonl") -Algorithm SHA256
```

Sebelum membuka hasil test, tinjau distribusi kelas train/val pada
`qa_dataset_summary.json`. Target yang digunakan adalah:

- `current_wrist_separation`: `narrower` atau `wider`.
- `future_wrist_separation_change`: `closing`, `stable`, atau `opening`.

Jika target masa depan hampir selalu `stable`, hentikan eksperimen B4 dan
revisi target pada train/val.

## 4. Latih dan validasi Dynamics pada fitur recovery

```powershell
& $Py eksperimen_model\train_dynamics.py `
    --config eksperimen_model\configs\mmfi_dynamics_best_tuned.yaml `
    --features_dir $Features `
    --epochs 150 `
    --batch_size 64 `
    --output_dir $DynamicsDir

& $Py eksperimen_model\evaluate_dynamics.py `
    --config eksperimen_model\configs\mmfi_dynamics_best_tuned.yaml `
    --features_dir $Features `
    --checkpoint $DynamicsCheckpoint `
    --split val `
    --batch_size 64 `
    --output_dir (Join-Path $Run "dynamics_val")
```

Pada validasi, bandingkan MSE latent Dynamics dengan persistensi. Jika
Dynamics tidak mengungguli persistensi pada target yang relevan, berhenti di
sini dan perbaiki Tahap 3.

## 5. Jalankan probe B2 dan sanity check projector

Probe hanya menyentuh train/val. Test tetap tertutup sampai langkah terakhir.

```powershell
& $Py eksperimen_model\train_probe.py `
    --qa_dir $QA `
    --features_dir $Features `
    --epochs 30 `
    --batch_size 128 `
    --seed 42 `
    --output_dir $ProbeDir

& $Py eksperimen_model\test_projector_pipeline.py
```

Lanjutkan bila probe pada validasi mengungguli baseline kelas mayoritas dan
sanity check projector lulus.

## 6. Latih B3, B3P, dan B4

Ketiga perintah menggunakan QA dan fitur recovery dari konfigurasi
`mmfi_projector_qwen.yaml`. Jangan mengubah epoch, batch, learning rate, atau
seed antar kondisi.

```powershell
& $Py eksperimen_model\train_projector.py `
    --config eksperimen_model\configs\mmfi_projector_qwen.yaml `
    --condition B3 `
    --output_dir (Join-Path $ProjectorDir "b3")

& $Py eksperimen_model\train_projector.py `
    --config eksperimen_model\configs\mmfi_projector_qwen.yaml `
    --condition B3P `
    --output_dir (Join-Path $ProjectorDir "b3p")

& $Py eksperimen_model\train_projector.py `
    --config eksperimen_model\configs\mmfi_projector_qwen.yaml `
    --condition B4 `
    --dynamics_checkpoint $DynamicsCheckpoint `
    --output_dir (Join-Path $ProjectorDir "b4")
```

- B3: 16 state riwayat.
- B3P: 16 state riwayat + 8 salinan state terakhir.
- B4: 16 state riwayat + 8 forecast Dynamics beku.

Pastikan tiga file berikut tersedia sebelum melanjutkan:

```powershell
$ProjectorCheckpoints = @(
    (Join-Path $ProjectorDir "b3\best_projector.pth"),
    (Join-Path $ProjectorDir "b3p\best_projector.pth"),
    (Join-Path $ProjectorDir "b4\best_projector.pth")
)
foreach ($Checkpoint in $ProjectorCheckpoints) {
    if (-not (Test-Path $Checkpoint)) { throw "Checkpoint projector tidak ditemukan: $Checkpoint" }
}
```

## 7. Evaluasi test beku

Jalankan langkah ini sekali setelah target, ambang, dan checkpoint sudah
dibekukan.

```powershell
& $Py eksperimen_model\evaluate_dynamics.py `
    --config eksperimen_model\configs\mmfi_dynamics_best_tuned.yaml `
    --features_dir $Features `
    --checkpoint $DynamicsCheckpoint `
    --split test `
    --batch_size 64 `
    --output_dir (Join-Path $Run "dynamics_test")

& $Py eksperimen_model\evaluate_reasoning.py `
    --config eksperimen_model\configs\mmfi_projector_qwen.yaml `
    --features_dir $Features `
    --dynamics_checkpoint $DynamicsCheckpoint `
    --checkpoint_b2 (Join-Path $ProbeDir "best_probe_model.pth") `
    --checkpoint_b3 (Join-Path $ProjectorDir "b3\best_projector.pth") `
    --checkpoint_b3p (Join-Path $ProjectorDir "b3p\best_projector.pth") `
    --checkpoint_b4 (Join-Path $ProjectorDir "b4\best_projector.pth") `
    --output_json (Join-Path $Run "stage4_reasoning_benchmark.json") `
    --output_report (Join-Path $Run "stage4_reasoning_benchmark.md")
```

Metrik utama adalah selisih macro-F1 target masa depan **B4 − B3P** pada panel
test yang sama. Laporan juga menyimpan respons mentah, status parse JSON,
strata subjek, batas repetisi, dan kontrol sensor shuffle.

Jangan menyimpulkan kontribusi Dynamics jika cakupan parse rendah, lineage
artefak tidak cocok, atau B4 tidak mengungguli B3P pada metrik utama.
