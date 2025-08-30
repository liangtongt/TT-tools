@echo off
echo Installing ComfyUI Image Sequence Compressor Node...
echo.

REM 检查是否在ComfyUI custom_nodes目录中
if not exist "..\..\main.py" (
    echo Error: This script must be run from the ComfyUI custom_nodes directory
    echo Please navigate to your ComfyUI custom_nodes folder and run this script
    pause
    exit /b 1
)

echo Installing Python dependencies...
pip install -r requirements.txt

echo.
echo Installation completed!
echo Please restart ComfyUI to load the new node.
echo.
pause
