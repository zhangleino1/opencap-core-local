# OpenCap 同步录制工具

这是一个专为 OpenCap 项目设计的同步录制工具，可以同时控制多个摄像头进行录制，非常适合动作捕获应用。

## 功能特点

- ✅ **同步录制**：一键启动/停止所有摄像头录制
- ✅ **实时监控**：显示每个摄像头的运行状态和录制时长
- ✅ **日志记录**：详细的操作日志和状态反馈
- ✅ **Web界面**：友好的浏览器界面，支持移动设备
- ✅ **自动分段**：支持按时间自动分段录制
- ✅ **灵活配置**：支持RTSP流地址和录制参数配置

## 快速开始

### 1. 安装依赖

```bash
pip install fastapi uvicorn python-multipart jinja2
```

### 2. 配置摄像头

编辑 `stream_config.json` 文件：

```json
{
    "streams": [
        {
            "name": "Camera 1",
            "rtsp_url": "rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101",
            "enabled": true
        },
        {
            "name": "Camera 2", 
            "rtsp_url": "rtsp://admin:password@192.168.1.101:554/Streaming/Channels/101",
            "enabled": true
        }
    ],
    "output_directory": "D:\\video",
    "segment_time": 30
}
```

### 3. 启动服务

```bash
cd video_capture
python stream_processor.py
```

### 4. 打开浏览器

访问 `http://localhost:8001` 即可使用录制界面。

## 界面功能

### 录制控制面板
- **开始录制**：同步启动所有启用的摄像头
- **停止录制**：同步停止所有摄像头录制
- **录制状态**：显示录制指示器和时长
- **摄像头统计**：显示运行中的摄像头数量

### 摄像头状态
- 实时显示每个摄像头的运行状态
- 显示录制时长
- 状态指示器（运行中/已停止/错误）

### 录制日志
- 实时显示录制操作日志
- 包含时间戳和详细信息
- 支持清空日志功能

### 系统配置
- 设置视频输出目录
- 配置分段录制时间
- 保存配置更改

## 输出文件结构

录制的视频文件按以下结构保存：

```
output_directory/
├── 2024-01-15/
│   ├── camera1/
│   │   ├── 2024-01-15_10-30-00.mp4
│   │   ├── 2024-01-15_10-30-30.mp4
│   │   └── ...
│   └── camera2/
│       ├── 2024-01-15_10-30-00.mp4
│       ├── 2024-01-15_10-30-30.mp4
│       └── ...
└── 2024-01-16/
    └── ...
```

## 配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `streams[].name` | 摄像头名称 | - |
| `streams[].rtsp_url` | RTSP流地址 | - |
| `streams[].enabled` | 是否启用 | true |
| `output_directory` | 输出目录 | "recordings" |
| `segment_time` | 分段时间(秒) | 60 |

## API 接口

### 录制控制
- `POST /api/recording/start` - 开始同步录制
- `POST /api/recording/stop` - 停止同步录制
- `GET /api/recording/status` - 获取录制状态

### 摄像头控制  
- `GET /api/streams` - 获取摄像头状态
- `POST /api/streams/{name}/start` - 启动单个摄像头
- `POST /api/streams/{name}/stop` - 停止单个摄像头

### 配置管理
- `GET /api/config` - 获取配置
- `POST /api/config` - 更新配置

## 与 OpenCap 集成

录制完成的视频文件可直接用于 OpenCap 处理流程：

1. 将录制的视频文件复制到 OpenCap 数据目录
2. 使用 `local_opencap_pipeline.py` 进行处理：

```python
from local_opencap_pipeline import run_local_opencap

# 处理录制的视频
success = run_local_opencap(
    video_dir="D:/video/2024-01-15",
    calibration_dir=None,  # 如果有标定视频
    pose_detector="OpenPose",
    resolution="1x736"
)
```

## 故障排除

### 摄像头连接失败
1. 检查RTSP URL是否正确
2. 确认网络连接正常
3. 验证摄像头用户名密码
4. 检查防火墙设置

### 录制文件损坏
1. 检查磁盘空间是否充足
2. 确认输出目录有写入权限
3. 检查网络稳定性

### 界面无响应
1. 刷新浏览器页面
2. 检查服务器日志
3. 重启服务进程

## 日志文件

- 服务器日志：`logs/video_YYYY-MM-DD.log`
- 包含详细的操作记录和错误信息
- 支持日志轮转和压缩

## 系统要求

- Python 3.7+
- FFmpeg (需要在系统PATH中)
- 至少2GB可用磁盘空间
- 稳定的网络连接

## 注意事项

1. 确保摄像头支持RTSP协议
2. 建议使用有线网络连接
3. 定期清理录制文件以释放空间
4. 录制前先测试摄像头连接状态