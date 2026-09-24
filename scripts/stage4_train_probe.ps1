$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Py = ".venv\Scripts\python.exe"
$Features = "datasets\MM-Fi_features_stage4_recovery"
$QA = "datasets\MM-Fi_grounded_qa_stage4_recovery"
$Output = "eksperimen_model\checkpoints\probe_stage4_recovery"

if (-not (Test-Path $Py)) { throw "Python .venv tidak ditemukan: $Py" }
if (-not (Test-Path $Features)) { throw "Fitur belum dibuat: $Features" }
if (-not (Test-Path $QA)) { throw "QA belum dibuat: $QA" }
if (Test-Path $Output) { throw "Output probe sudah ada: $Output" }

& $Py "eksperimen_model\train_probe.py" `
    --qa_dir $QA `
    --features_dir $Features `
    --epochs 30 `
    --batch_size 128 `
    --seed 42 `
    --output_dir $Output
if ($LASTEXITCODE -ne 0) { throw "Training probe gagal dengan exit code $LASTEXITCODE" }
Write-Host "Training probe selesai: $Output" -ForegroundColor Green
