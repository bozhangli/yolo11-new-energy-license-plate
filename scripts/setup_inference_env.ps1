param(
    [ValidateSet("gpu", "cpu")]
    [string]$Mode = "gpu",
    [string]$PythonCommand = "python",
    [string]$VenvPath = ".venv",
    [switch]$SkipTorchInstall,
    [switch]$SkipProjectInstall,
    [switch]$VerifyOnly
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvRoot = if ([System.IO.Path]::IsPathRooted($VenvPath)) { $VenvPath } else { Join-Path $repoRoot $VenvPath }
$venvPython = Join-Path $venvRoot "Scripts\python.exe"
$modelTest = Join-Path $repoRoot "examples\model_test.py"
$defaultModel = Join-Path $repoRoot "yolo11n.pt"

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-Python {
    param([string[]]$Arguments)
    & $venvPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $venvPython $($Arguments -join ' ')"
    }
}

Write-Step "Repository root: $repoRoot"
Write-Step "Virtual environment: $venvRoot"
Write-Step "Install mode: $Mode"

if (-not (Test-Path $venvPython)) {
    Write-Step "Creating virtual environment"
    & $PythonCommand -m venv $venvRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create virtual environment with '$PythonCommand -m venv $venvRoot'."
    }
} else {
    Write-Step "Reusing existing virtual environment"
}

if (-not $VerifyOnly) {
    Write-Step "Upgrading pip, setuptools, and wheel"
    Invoke-Python -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")

    if (-not $SkipTorchInstall) {
        if ($Mode -eq "gpu") {
            Write-Step "Installing CUDA-enabled torch and torchvision"
            Invoke-Python -Arguments @(
                "-m", "pip", "install", "--force-reinstall", "--no-deps", "--no-cache-dir",
                "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu128"
            )
        } else {
            Write-Step "Installing CPU torch and torchvision"
            Invoke-Python -Arguments @("-m", "pip", "install", "--upgrade", "torch", "torchvision")
        }
    } else {
        Write-Step "Skipping torch installation"
    }

    if (-not $SkipProjectInstall) {
        Write-Step "Installing repository in editable mode"
        Invoke-Python -Arguments @("-m", "pip", "install", "-e", $repoRoot)
    } else {
        Write-Step "Skipping editable project installation"
    }
} else {
    Write-Step "Verify-only mode, skipping package installation"
}

Write-Step "Verifying runtime"
Invoke-Python -Arguments @(
    "-c",
    "import torch; import ultralytics; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'); print('ultralytics', ultralytics.__version__)"
)

Write-Host "" 
Write-Host "Environment is ready." -ForegroundColor Green
Write-Host "Activate with:" -ForegroundColor Green
Write-Host "  & '$venvRoot\Scripts\Activate.ps1'"
Write-Host "Run inference with:" -ForegroundColor Green
Write-Host "  $venvPython $modelTest --model $defaultModel --source ultralytics/assets/bus.jpg --device $(if ($Mode -eq 'gpu') { '0' } else { 'cpu' }) --save"