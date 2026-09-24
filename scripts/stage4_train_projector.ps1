param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("B3", "B3P", "B4")]
    [string]$Condition
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Py = ".venv\Scripts\python.exe"
$Features = "datasets\MM-Fi_features_stage4_recovery"
$QA = "datasets\MM-Fi_grounded_qa_stage4_recovery"
$Dynamics = "eksperimen_model\checkpoints\dynamics_stage4_recovery\best_dynamics_model.pth"
$Output = Join-Path "eksperimen_model\checkpoints\projector_stage4_recovery" $Condition.ToLower()

if (-not (Test-Path $Py)) { throw "Python .venv tidak ditemukan: $Py" }
if (-not (Test-Path $Features)) { throw "Fitur belum dibuat: $Features" }
if (-not (Test-Path $QA)) { throw "QA belum dibuat: $QA" }
if (Test-Path $Output) { throw "Output projector $Condition sudah ada: $Output" }

$Arguments = @(
    "eksperimen_model\train_projector.py",
    "--config", "eksperimen_model\configs\mmfi_projector_qwen.yaml",
    "--condition", $Condition,
    "--output_dir", $Output
)
if ($Condition -eq "B4") { $Arguments += @("--dynamics_checkpoint", $Dynamics) }

& $Py @Arguments
if ($LASTEXITCODE -ne 0) { throw "Training projector $Condition gagal dengan exit code $LASTEXITCODE" }
Write-Host "Training projector $Condition selesai: $Output" -ForegroundColor Green
