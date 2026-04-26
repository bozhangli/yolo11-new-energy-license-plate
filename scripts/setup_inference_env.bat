@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "MODE=cpu"
set "PYTHON_CMD=python"
set "VENV_PATH=.venv"
set "SKIP_TORCH=0"
set "SKIP_PROJECT=0"
set "VERIFY_ONLY=0"

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--gpu" (
    set "MODE=gpu"
    shift
    goto parse_args
)
if /I "%~1"=="--cpu" (
    set "MODE=cpu"
    shift
    goto parse_args
)
if /I "%~1"=="--python" (
    if "%~2"=="" goto missing_python
    set "PYTHON_CMD=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--venv" (
    if "%~2"=="" goto missing_venv
    set "VENV_PATH=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--skip-torch" (
    set "SKIP_TORCH=1"
    shift
    goto parse_args
)
if /I "%~1"=="--skip-project" (
    set "SKIP_PROJECT=1"
    shift
    goto parse_args
)
if /I "%~1"=="--verify-only" (
    set "VERIFY_ONLY=1"
    shift
    goto parse_args
)
if /I "%~1"=="--help" goto usage
if /I "%~1"=="-h" goto usage

echo Unknown argument: %~1
goto usage

:missing_python
echo Missing value after --python
goto usage

:missing_venv
echo Missing value after --venv
goto usage

:args_done
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "SEARCH_DIR=%SCRIPT_DIR%"

:find_repo_root
if exist "%SEARCH_DIR%\pyproject.toml" (
    set "REPO_ROOT=%SEARCH_DIR%"
    goto repo_root_found
)
for %%I in ("%SEARCH_DIR%\..") do set "PARENT_DIR=%%~fI"
if /I "%PARENT_DIR%"=="%SEARCH_DIR%" goto repo_root_not_found
set "SEARCH_DIR=%PARENT_DIR%"
goto find_repo_root

:repo_root_not_found
echo Failed to locate pyproject.toml from %SCRIPT_DIR%
goto fail

:repo_root_found

if exist "%VENV_PATH%\Scripts\python.exe" (
    for %%I in ("%VENV_PATH%") do set "VENV_ROOT=%%~fI"
) else (
    set "VENV_ROOT=%REPO_ROOT%\%VENV_PATH%"
)

set "VENV_PYTHON=%VENV_ROOT%\Scripts\python.exe"
set "MODEL_TEST=%REPO_ROOT%\examples\model_test.py"
set "DEFAULT_MODEL=%REPO_ROOT%\yolo11n.pt"

call :step "Repository root: %REPO_ROOT%"
call :step "Virtual environment: %VENV_ROOT%"
call :step "Install mode: %MODE%"

if not exist "%VENV_PYTHON%" (
    call :step "Creating virtual environment"
    "%PYTHON_CMD%" -m venv "%VENV_ROOT%"
    if errorlevel 1 goto fail
) else (
    call :step "Reusing existing virtual environment"
)

if "%VERIFY_ONLY%"=="1" (
    call :step "Verify-only mode, skipping package installation"
) else (
    call :step "Upgrading pip, setuptools, and wheel"
    call :run_python -m pip install --upgrade pip setuptools wheel
    if errorlevel 1 goto fail

    if "%SKIP_TORCH%"=="1" (
        call :step "Skipping torch installation"
    ) else (
        if /I "%MODE%"=="gpu" (
            call :step "Installing CUDA-enabled torch and torchvision"
            call :run_python -m pip install --force-reinstall --no-deps --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128
            if errorlevel 1 goto fail
        ) else (
            call :step "Installing CPU torch and torchvision"
            call :run_python -m pip install --upgrade torch torchvision
            if errorlevel 1 goto fail
        )
    )

    if "%SKIP_PROJECT%"=="1" (
        call :step "Skipping editable project installation"
    ) else (
        call :step "Installing repository in editable mode"
        call :run_python -m pip install -e "%REPO_ROOT%"
        if errorlevel 1 goto fail
    )
)

call :step "Verifying runtime"
call :run_python -c "import torch; import ultralytics; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'); print('ultralytics', ultralytics.__version__)"
if errorlevel 1 goto fail

echo.
echo Environment is ready.
echo Activate with:
echo   %VENV_ROOT%\Scripts\Activate.ps1
echo Run inference with:
if /I "%MODE%"=="gpu" (
    echo   %VENV_PYTHON% %MODEL_TEST% --model %DEFAULT_MODEL% --source ultralytics/assets/bus.jpg --device 0 --save
) else (
    echo   %VENV_PYTHON% %MODEL_TEST% --model %DEFAULT_MODEL% --source ultralytics/assets/bus.jpg --device cpu --save
)
exit /b 0

:step
echo ==^> %~1
exit /b 0

:run_python
"%VENV_PYTHON%" %*
exit /b %errorlevel%

:fail
echo.
echo Setup failed.
exit /b 1

:usage
echo Usage: setup_inference_env.bat [--cpu ^| --gpu] [--python PYTHON] [--venv PATH] [--skip-torch] [--skip-project] [--verify-only]
echo.
echo Default mode is CPU.
exit /b 1