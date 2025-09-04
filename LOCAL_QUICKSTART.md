# 本地OpenCap快速开始指南

## 🚀 5分钟快速开始

### 1. 准备数据
```bash
# 创建数据目录
mkdir my_motion_capture
cd my_motion_capture

# 准备视频文件
mkdir videos calibration
# 将运动视频放入 videos/ 目录
# 将标定视频放入 calibration/ 目录 (可选)
```

### 2. 一行命令运行
```python
from local_opencap_pipeline import run_local_opencap

# 最简单的使用方式
success = run_local_opencap(
    video_dir="./videos",           # 运动视频目录
    calibration_dir="./calibration", # 标定视频目录 (可选)
    pose_detector="OpenPose"        # 或 "mmpose"
)
```

### 3. 查看结果
结果保存在 `Data/LocalSession_*/` 目录中：
- `*.trc` - 3D标记点数据
- `OpenSimData/` - 生物力学分析结果
- `OutputMedia*/` - 处理后的视频

---

## 📁 文件组织

### 推荐的目录结构
```
your_project/
├── videos/                    # 🎥 运动视频
│   ├── camera1_walk.mp4      
│   ├── camera2_walk.mp4      
│   └── ...
├── calibration/              # 📐 标定视频 (可选)
│   ├── camera1_calib.mp4     
│   ├── camera2_calib.mp4     
│   └── ...
└── config.yaml              # ⚙️ 配置文件 (可选)
```

### 视频文件命名规则
- 包含摄像头标识: `camera1_xxx.mp4`, `cam2_xxx.mp4`
- 或使用品牌型号: `iphone15_xxx.mp4`, `samsung_xxx.mp4`
- 系统会自动识别和分组

---

## ⚙️ 配置选项

### 创建配置文件
```python
from local_opencap_pipeline import create_config_template
create_config_template("my_config.yaml")
```

### 关键配置参数
```yaml
session:
  subject_mass: 70.0      # 受试者体重(kg)
  subject_height: 170.0   # 受试者身高(cm)

calibration:
  checkerboard:
    dimensions: [11, 8]   # 标定板内角点 [宽, 高]
    square_size: 60       # 正方形边长(mm)

processing:
  pose_detector: 'OpenPose'  # 'OpenPose' 或 'mmpose'
  resolution: '1x736'        # OpenPose分辨率
  augmenter_model: 'v0.3'    # LSTM模型版本
```

---

## 🔧 分辨率选择指南

| 分辨率 | 准确性 | 内存需求 | 处理速度 | 推荐用途 |
|--------|--------|----------|----------|----------|
| `default` | 低 | 4GB | 快 | 快速测试 |
| `1x736` | 中 | 4GB | 中 | 常规使用 (推荐) |
| `1x736_2scales` | 高 | 8GB | 慢 | 高质量分析 |
| `1x1008_4scales` | 最高 | 24GB | 很慢 | 研究级精度 |

---

## 📋 完整使用示例

### 基础示例
```python
from local_opencap_pipeline import LocalOpenCapPipeline

# 创建管道
pipeline = LocalOpenCapPipeline()

# 处理会话
success = pipeline.process_session(
    video_directory="./videos",
    calibration_directory="./calibration"
)
```

### 高级示例
```python
# 自定义配置
config = {
    'session': {
        'name': 'MyExperiment_Walking',
        'subject_mass': 75.0,
        'subject_height': 180.0
    },
    'processing': {
        'pose_detector': 'mmpose',
        'resolution': '1x736_2scales',
        'augmenter_model': 'v0.3'
    }
}

pipeline = LocalOpenCapPipeline(config_dict=config)
success = pipeline.process_session("./videos", "./calibration")
```

### 批量处理
```python
sessions = [
    {"name": "Walk_Trial1", "video_dir": "./walk1"},
    {"name": "Walk_Trial2", "video_dir": "./walk2"},
    {"name": "Run_Trial1", "video_dir": "./run1"}
]

for session in sessions:
    run_local_opencap(
        video_dir=session["video_dir"],
        **{"session.name": session["name"]}
    )
```

---

## 🛠️ 故障排除

### 常见问题

#### 1️⃣ 相机标定失败
```
❌ 标定失败: 只找到 X 幅有效图像，少于最低要求(10幅)
```
**解决方案:**
- 检查标定板参数是否正确
- 确保标定视频中棋盘格清晰可见
- 增加 `n_images` 参数值

#### 2️⃣ 姿态检测失败
```
❌ OpenPose/MMPose 未找到或配置错误
```
**解决方案:**
- 确保已安装OpenPose或配置MMPose
- 检查环境变量PATH设置
- 尝试重新安装依赖

#### 3️⃣ 内存不足
```
❌ CUDA out of memory
```
**解决方案:**
- 降低分辨率: `resolution: 'default'`
- 减少批处理大小
- 关闭其他GPU应用

#### 4️⃣ 找不到视频文件
```
❌ 在目录中没有找到.mp4文件
```
**解决方案:**
- 检查文件路径是否正确
- 确保视频格式为.mp4
- 检查文件权限

---

## 📊 输出文件说明

### 主要输出文件
- `*.trc` - 3D标记点轨迹 (可导入OpenSim/Visual3D)
- `*_keypoints.pkl` - 2D姿态检测结果
- `cameraIntrinsicsExtrinsics.pickle` - 相机参数

### OpenSim输出 (如启用)
- `*.osim` - 缩放后的人体模型
- `*.mot` - 逆向运动学结果  
- `*.json` - 3D可视化数据

### 调试输出
- `OutputMedia*/` - 带标记的处理视频
- `processing_report.yaml` - 处理摘要报告

---

## 🔄 与原版OpenCap的区别

| 功能 | 原版OpenCap | 本地版本 |
|------|------------|----------|
| 数据获取 | 云端下载 | ✅ 本地文件 |
| 相机标定 | 需要API | ✅ 完全本地 |
| 姿态检测 | 云端/本地 | ✅ 完全本地 |
| 3D重建 | 本地 | ✅ 完全本地 |
| 网络依赖 | ❌ 需要 | ✅ 无需 |
| API认证 | ❌ 需要 | ✅ 无需 |

---

## 🚀 性能优化建议

### GPU设置
- 确保CUDA版本匹配
- 监控GPU内存使用
- 适当调整批处理大小

### 存储优化
- 使用SSD存储数据
- 定期清理中间文件
- 压缩大型视频文件

### 内存管理
- 处理大型会话时重启Python
- 使用 `delete_intermediate=True`
- 监控系统内存使用

---

## 📞 获取帮助

### 日志分析
检查详细日志输出了解错误详情:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 社区支持
- 原项目: [OpenCap GitHub](https://github.com/stanfordnmbl/opencap-core)
- 文档: [OpenCap文档](https://www.opencap.ai/docs)

### 常用命令行
```bash
# 创建配置模板
python local_opencap_pipeline.py --create-config my_config.yaml

# 运行处理
python local_opencap_pipeline.py ./videos --calibration-dir ./calibration --config my_config.yaml

# 查看帮助
python local_opencap_pipeline.py --help
```

---

**🎉 现在你已经准备好使用完全本地化的OpenCap了！开始处理你的运动捕获数据吧！**