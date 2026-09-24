$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Py = ".venv\Scripts\python.exe"
$Features = "datasets\MM-Fi_features_stage4_recovery"
$QA = "datasets\MM-Fi_grounded_qa_stage4_recovery"

if (-not (Test-Path $Py)) { throw "Python .venv tidak ditemukan: $Py" }
if (-not (Test-Path $Features)) { throw "Fitur belum dibuat: $Features" }
if (Test-Path $QA) { throw "Direktori QA sudah ada: $QA" }

& $Py "eksperimen_model\datasets\generate_grounded_qa.py" `
    --features_dir $Features `
    --output_dir $QA `
    --segments_csv "datasets\MM-Fi Dataset\MMFi_action_segments.csv" `
    --raw_dataset_dir "datasets\MM-Fi Dataset\filtered_mmwave" `
    --train_stride 4 `
    --val_stride 8 `
    --test_stride 8 `
    --seed 42
if ($LASTEXITCODE -ne 0) { throw "Pembuatan QA gagal dengan exit code $LASTEXITCODE" }
Write-Host "Manifest QA selesai: $QA" -ForegroundColor Green
