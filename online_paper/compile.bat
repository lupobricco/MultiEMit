@echo off
cd /d "%~dp0"
echo Compiling draft.tex...
pdflatex -interaction=nonstopmode draft.tex
if %ERRORLEVEL% equ 0 (
    echo.
    echo =======================================
    echo  Compilation SUCCESS! Output: draft.pdf
    echo =======================================
) else (
    echo.
    echo =======================================
    echo  Compilation FAILED! Check draft.log
    echo =======================================
)
