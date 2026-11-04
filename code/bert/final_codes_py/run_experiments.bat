@echo off
set PYTHONUTF8=1
echo ==============================================================================
echo MultiEMit Python Benchmark Runner
echo ==============================================================================

:: Find python
if exist "..\..\..\.conda\python.exe" (
    set "PYTHON_EXE=..\..\..\.conda\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo Using Python: %PYTHON_EXE%
echo.

%PYTHON_EXE% run_experiments.py %*

echo.
pause

