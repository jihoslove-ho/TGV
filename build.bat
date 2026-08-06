@echo off
echo ============================================
echo  TGV PSL Inspector - Build Start
echo ============================================
echo.

python -m pip install pyinstaller --quiet

echo [1/3] Cleaning old build files...
if exist build rmdir /s /q build
if exist dist  rmdir /s /q dist

echo [2/3] Building EXE... (please wait 2-5 min)
python -m PyInstaller ^
    --onedir ^
    --windowed ^
    --name "TGV_PSL_Inspector" ^
    --hidden-import cv2 ^
    --hidden-import PIL ^
    --hidden-import PIL.Image ^
    --hidden-import PIL.ImageTk ^
    --hidden-import pandas ^
    --hidden-import openpyxl ^
    --hidden-import numpy ^
    --hidden-import numpy.core ^
    --hidden-import tkinter ^
    --hidden-import tkinter.ttk ^
    --exclude-module matplotlib ^
    --exclude-module scipy ^
    --exclude-module PyQt5 ^
    tgv_v4.py

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed.
    pause
    exit /b 1
)

echo.
echo [3/3] Build complete!
echo.
echo Result: dist\TGV_PSL_Inspector\TGV_PSL_Inspector.exe
echo.
echo ============================================
pause
