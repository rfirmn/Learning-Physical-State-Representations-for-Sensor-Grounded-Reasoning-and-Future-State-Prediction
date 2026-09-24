$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Py = ".venv\Scripts\python.exe"
$Features = "datasets\MM-Fi_features_stage4_recovery"
$QA = "datasets\MM-Fi_grounded_qa_stage4_recovery"
$Dynamics = "eksperimen_model\checkpoints\dynamics_stage4_recovery\best_dynamics_model.pth"
$Probe = "eksperimen_model\checkpoints\probe_stage4_recovery\best_probe_model.pth"
$Projector = "eksperimen_model\checkpoints\projector_stage4_recovery"
$Output = "docs\report_training\stage4_recovery_reasoning_benchmark.json"
$Report = "docs\report_training\stage4_recovery_reasoning_benchmark.md"

if (-not (Test-Path $Py)) { throw "Python .venv tidak ditemukan: $Py" }
foreach ($Path in @($Features, $QA, $Dynamics, $Probe,
    (Join-Path $Projector "b3\best_projector.pth"),
    (Join-Path $Projector "b3p\best_projector.pth"),
    (Join-Path $Projector "b4\best_projector.pth"))) {
    if (-not (Test-Path $Path)) { throw "Artefak evaluasi tidak ditemukan: $Path" }
}
if (Test-Path $Output) { throw "Laporan sudah ada: $Output" }

& $Py "eksperimen_model\evaluate_reasoning.py" `
    --config "eksperimen_model\configs\mmfi_projector_qwen.yaml" `
    --features_dir $Features `
    --dynamics_checkpoint $Dynamics `
    --checkpoint_b2 $Probe `
    --checkpoint_b3 (Join-Path $Projector "b3\best_projector.pth") `
    --checkpoint_b3p (Join-Path $Projector "b3p\best_projector.pth") `
    --checkpoint_b4 (Join-Path $Projector "b4\best_projector.pth") `
    --output_json $Output `
    --output_report $Report
if ($LASTEXITCODE -ne 0) { throw "Evaluasi Stage 4 gagal dengan exit code $LASTEXITCODE" }
Write-Host "Evaluasi selesai: $Output" -ForegroundColor Green
