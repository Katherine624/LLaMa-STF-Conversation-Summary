$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvDir = Join-Path $ProjectDir ".venv"
$PythonExe = Join-Path $EnvDir "Scripts\python.exe"

Write-Host "[1/4] Creating an isolated Python environment at $EnvDir"
if (-not (Test-Path -LiteralPath $PythonExe)) {
    python -m venv $EnvDir
}

Write-Host "[2/4] Updating pip"
& $PythonExe -m pip install --upgrade pip

Write-Host "[3/4] Installing PyTorch 2.9.1 with CUDA 13.0"
& $PythonExe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130

Write-Host "[4/4] Installing fine-tuning packages"
& $PythonExe -m pip install -r (Join-Path $ProjectDir "requirements-local.txt")

Write-Host "Running environment checks"
& $PythonExe (Join-Path $ProjectDir "check_env.py")

Write-Host "Setup complete."


