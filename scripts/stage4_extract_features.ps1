$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Py = ".venv\Scripts\python.exe"
$Features = "datasets\MM-Fi_features_stage4_recovery"

if (-not (Test-Path $Py)) { throw "Python .venv tidak ditemukan: $Py" }
if (Test-Path $Features) { throw "Direktori fitur sudah ada: $Features" }

& $Py "eksperimen_model\datasets\extract_physical_features.py" `
    --config "eksperimen_model\configs\mmfi_dynamics_v3.yaml" `
    --output_dir $Features `
    --seed 42
if ($LASTEXITCODE -ne 0) { throw "Ekstraksi fitur gagal dengan exit code $LASTEXITCODE" }
if (-not (Test-Path (Join-Path $Features "normalization_stats.pt"))) {
    throw "normalization_stats.pt tidak dibuat"
}
Write-Host "Ekstraksi fitur selesai: $Features" -ForegroundColor Green
