@echo off
chcp 65001 > nul
setlocal

echo ================================================================================
echo MULTIEMIT: STREAMLINED PAPER EXPERIMENTS RUNNER
echo Target: IEEE Transactions on Affective Computing Synthesis
echo ================================================================================

set PYTHON_EXE=..\..\..\.conda\python.exe
if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=python
)

"%PYTHON_EXE%" run_paper_experiments.py %*

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Experiments completed successfully.
) else (
    echo.
    echo An error occurred during execution (exit code %ERRORLEVEL%).
)

pause

