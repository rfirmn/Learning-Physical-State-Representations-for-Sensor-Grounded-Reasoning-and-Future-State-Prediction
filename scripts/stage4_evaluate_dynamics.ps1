$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Py = ".venv\Scripts\python.exe"
$Features = "datasets\MM-Fi_features_stage4_recovery"
$Checkpoint = "eksperimen_model\checkpoints\dynamics_stage4_recovery\best_dynamics_model.pth"
$Output = "docs\report_training\stage4_recovery_dynamics_test.json"
$OutputDir = "docs\report_training\stage4_recovery_dynamics_test"

if (-not (Test-Path $Py)) { throw "Python .venv tidak ditemukan: $Py" }
foreach ($Path in @($Features, $Checkpoint)) {
    if (-not (Test-Path $Path)) { throw "Artefak Dynamics tidak ditemukan: $Path" }
}
if (Test-Path $Output) { throw "Laporan Dynamics sudah ada: $Output" }

& $Py "eksperimen_model\evaluate_dynamics.py" `
    --config "eksperimen_model\configs\mmfi_dynamics_best_tuned.yaml" `
    --features_dir $Features `
    --checkpoint $Checkpoint `
    --split test `
    --batch_size 64 `
    --output_dir $OutputDir `
    --output_json $Output
if ($LASTEXITCODE -ne 0) { throw "Evaluasi Dynamics gagal dengan exit code $LASTEXITCODE" }
Write-Host "Evaluasi Dynamics selesai: $Output" -ForegroundColor Green
