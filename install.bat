@echo off
chcp 65001 >nul
echo ================================================
echo TT img enc 节点安装程序 (Windows)
echo ================================================
echo.

echo 正在检查 Python 环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到 Python，请先安装 Python 3.7+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python 环境检查通过
echo.

echo 正在运行安装脚本...
python install.py

echo.
echo 安装完成！按任意键退出...
pause >nul
