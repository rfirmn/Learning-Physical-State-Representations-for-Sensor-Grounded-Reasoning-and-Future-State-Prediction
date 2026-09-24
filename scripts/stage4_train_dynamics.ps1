$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Py = ".venv\Scripts\python.exe"
$Features = "datasets\MM-Fi_features_stage4_recovery"
$Output = "eksperimen_model\checkpoints\dynamics_stage4_recovery"

if (-not (Test-Path $Py)) { throw "Python .venv tidak ditemukan: $Py" }
if (-not (Test-Path $Features)) { throw "Fitur belum dibuat: $Features" }
if (Test-Path $Output) { throw "Output Dynamics sudah ada: $Output" }

& $Py "eksperimen_model\train_dynamics.py" `
    --config "eksperimen_model\configs\mmfi_dynamics_best_tuned.yaml" `
    --features_dir $Features `
    --epochs 150 `
    --batch_size 64 `
    --output_dir $Output
if ($LASTEXITCODE -ne 0) { throw "Training Dynamics gagal dengan exit code $LASTEXITCODE" }
Write-Host "Training Dynamics selesai: $Output" -ForegroundColor Green
