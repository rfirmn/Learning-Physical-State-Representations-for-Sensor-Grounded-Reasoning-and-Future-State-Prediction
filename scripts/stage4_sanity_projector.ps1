$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Py = ".venv\Scripts\python.exe"

if (-not (Test-Path $Py)) { throw "Python .venv tidak ditemukan: $Py" }
& $Py "eksperimen_model\test_projector_pipeline.py"
if ($LASTEXITCODE -ne 0) { throw "Sanity check projector gagal dengan exit code $LASTEXITCODE" }
Write-Host "Sanity check projector lulus." -ForegroundColor Green
