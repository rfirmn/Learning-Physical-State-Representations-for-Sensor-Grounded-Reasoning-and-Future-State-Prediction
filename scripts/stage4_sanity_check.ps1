$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Py = ".venv\Scripts\python.exe"

if (-not (Test-Path $Py)) { throw "Python .venv tidak ditemukan: $Py" }
foreach ($Path in @(
    "datasets\MM-Fi Dataset\filtered_mmwave",
    "datasets\MM-Fi Dataset\MMFi_action_segments.csv",
    "eksperimen_model\checkpoints\pose_estimation_v2\model_av2.pth"
)) {
    if (-not (Test-Path $Path)) { throw "Artefak wajib tidak ditemukan: $Path" }
}

& $Py "eksperimen_model\test_pipeline.py"
if ($LASTEXITCODE -ne 0) { throw "Sanity check gagal dengan exit code $LASTEXITCODE" }
Write-Host "Sanity check lulus." -ForegroundColor Green
