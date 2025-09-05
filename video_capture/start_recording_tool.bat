@echo off
title OpenCap Sync Recording Tool
color 0A

echo ================================================================
echo               OpenCap 同步录制工具
echo ================================================================
echo.

cd /d "%~dp0"

echo 检查依赖项...
python -c "import fastapi, uvicorn, jinja2" 2>nul
if errorlevel 1 (
    echo ❌ 缺少依赖项，正在安装...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ 安装依赖项失败，请手动运行: pip install -r requirements.txt
        pause
        exit /b 1
    )
) else (
    echo ✅ 依赖项检查通过
)

echo.
echo 检查配置文件...
if not exist "stream_config.json" (
    echo ❌ 配置文件不存在，请先配置摄像头信息
    pause
    exit /b 1
) else (
    echo ✅ 配置文件存在
)

echo.
echo ================================================================
echo 🚀 启动服务器...
echo 💻 Web界面: http://localhost:8001
echo 📖 使用说明: README.md
echo ⏹️  按 Ctrl+C 停止服务器
echo ================================================================
echo.

python stream_processor.py

echo.
echo 服务器已停止
pause