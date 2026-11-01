@echo off
cd /d "%~dp0"
echo ============================================================================
echo Running All Final MultiEMit Textual and Multimodal Dual-Encoder Experiments
echo ============================================================================
"..\..\..\.conda\python.exe" -u run_all_experiments.py %*
echo.
echo Finished! Unified results are saved in:
echo   - results\all_experiments_summary.csv
echo   - results\all_experiments_summary.xlsx
echo   - results\all_experiments_full_report.md
pause
