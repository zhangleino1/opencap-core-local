# 使用外部标定文件指南

## 概述

当本地标定结果不理想时，可以使用从OpenCap官网下载的标定文件来排除标定问题。此功能允许你跳过本地标定步骤，直接使用经过验证的官网标定结果。

## 使用场景

✅ **适用于:**
- 本地标定结果导致人物姿态异常
- 需要排除标定环节的问题
- 想要对比本地标定和官网标定的差异
- 快速测试其他处理步骤

❌ **不适用于:**
- 视频来源与官网标定不匹配
- 需要完全独立的本地处理
- 摄像头内参发生变化

## 准备工作

### 1. 从OpenCap官网下载标定文件

1. 访问 https://app.opencap.ai
2. 登录并选择一个已处理的会话
3. 下载会话数据
4. 解压后找到标定文件路径:
   ```
   OpenCapData_xxx/
   └── Videos/
       ├── Cam1/
       │   └── cameraIntrinsicsExtrinsics.pickle  ← 这个文件
       ├── Cam2/
       │   └── cameraIntrinsicsExtrinsics.pickle
       └── ...
   ```

### 2. 准备本地视频

确保你有需要处理的视频文件:
```
LocalData/
├── Videos/         # 动态视频
│   ├── Cam1_xxx.mp4
│   └── Cam2_xxx.mp4
└── Static/         # 静态视频（可选）
    ├── Cam1_static.mp4
    └── Cam2_static.mp4
```

## 配置方法

### 方法一: 交互式配置（推荐）

运行配置向导:
```bash
python example_use_external_calibration.py --setup
```

按照提示输入:
- 摄像头数量
- 每个摄像头的标定文件路径

配置向导会自动创建 `config_external_calib.yaml` 文件。

### 方法二: 手动编辑配置文件

创建或编辑 `config_external_calib.yaml`:

```yaml
calibration:
  # 启用外部标定文件
  use_external_calibration: true

  # 指定每个摄像头的标定文件路径
  external_calibration_files:
    Cam1: "D:/path/to/downloaded/Videos/Cam1/cameraIntrinsicsExtrinsics.pickle"
    Cam2: "D:/path/to/downloaded/Videos/Cam2/cameraIntrinsicsExtrinsics.pickle"
    # 根据实际摄像头数量添加或删除

  # 其他标定参数（用于元数据）
  checkerboard:
    dimensions: [5, 4]
    square_size: 35
    placement: backWall

  # 不需要交互式选择（使用外部标定文件）
  interactive_selection: false

# 处理配置
processing:
  pose_detector: mmpose
  resolution: 1x736
  image_upsample_factor: 4
  augmenter_model: v0.3

# 会话配置
session:
  name: session_external_calib
  description: 使用外部标定文件的OpenCap会话
  subject_mass: 67.0
  subject_height: 170.0

# 目录配置
directories:
  input_videos: ./LocalData/Videos
  static_videos: ./LocalData/Static
```

## 运行处理

### 使用示例脚本

```bash
python example_use_external_calibration.py
```

### 使用主管道

```python
from local_opencap_pipeline import run_local_opencap

success = run_local_opencap(
    video_dir="./LocalData/Videos",
    calibration_dir=None,  # 不需要提供
    static_dir="./LocalData/Static",
    config_file="config_external_calib.yaml"
)
```

## 处理流程

启用外部标定后，处理流程如下:

1. ✅ **跳过标定试验** - 不进行本地标定
2. ✅ **复制外部标定文件** - 将官网标定文件复制到会话目录
3. ✅ **处理静态试验** - 使用外部标定进行静态姿态分析
4. ✅ **处理动态试验** - 使用外部标定进行运动分析
5. ✅ **生成结果** - TRC文件、OpenSim模型等

## 验证结果

### 1. 检查标定文件复制

处理日志应显示:
```
📂 使用外部标定文件
✅ Cam1: 已复制外部标定文件
   源文件: D:/path/to/downloaded/Videos/Cam1/cameraIntrinsicsExtrinsics.pickle
   目标: Data/session_xxx/Videos/Cam1/cameraIntrinsicsExtrinsics.pickle
   大小: xxxxx bytes
```

### 2. 检查姿态结果

在OpenSim中打开生成的模型和运动文件:
- `Data/session_xxx/OpenSimData/Model/*_scaled.osim`
- `Data/session_xxx/OpenSimData/Kinematics/*_ik.mot`

检查人物姿态是否正常:
- ✅ 人物直立
- ✅ 手臂和腿部姿势正确
- ✅ 运动轨迹合理

### 3. 对比分析

如果姿态正常，说明问题出在本地标定:
- 检查棋盘格放置方式
- 检查标定图像质量
- 检查标定方案选择

如果姿态仍然异常，说明问题可能在:
- 姿态检测阶段
- 3D重建阶段
- OpenSim缩放和IK阶段

## 常见问题

### Q1: 标定文件路径错误
**错误信息:** `❌ Cam1: 外部标定文件不存在`

**解决方法:**
- 检查路径是否正确
- 确认文件确实存在
- Windows路径使用 `/` 或 `\\`（YAML中使用 `/` 更安全）

### Q2: 摄像头名称不匹配
**错误信息:** `❌ Cam1: 未指定外部标定文件`

**解决方法:**
- 确认配置文件中的摄像头名称（Cam1, Cam2等）
- 确认视频文件名包含正确的摄像头标识
- 检查 `_organize_videos_by_camera` 的识别逻辑

### Q3: 标定文件不兼容
**错误信息:** 处理过程中出现pickle加载错误

**解决方法:**
- 确认标定文件来自OpenCap官方系统
- 检查OpenCap版本兼容性
- 尝试使用相同版本的OpenCap处理

### Q4: 姿态仍然异常
如果使用外部标定后姿态仍然异常:

1. **检查视频匹配性:**
   - 外部标定文件是否来自相同的摄像头设置？
   - 视频分辨率是否一致？
   - 摄像头位置是否改变？

2. **检查其他处理参数:**
   - 姿态检测器设置
   - 棋盘格放置方式 (placement)
   - OpenSim模型参数

3. **查看处理日志:**
   - 检查3D重建的残差
   - 检查姿态检测的置信度
   - 检查OpenSim缩放因子

## 高级用法

### 混合使用本地和外部标定

可以为不同摄像头使用不同来源的标定:

```yaml
calibration:
  use_external_calibration: true
  external_calibration_files:
    Cam1: "path/to/external/Cam1/calib.pickle"  # 外部标定
    # Cam2 不指定，将使用本地标定（如果有）
```

### 批量处理多个会话

```python
import glob
from local_opencap_pipeline import LocalOpenCapPipeline

# 获取所有下载的标定文件
calib_sessions = glob.glob("Downloads/OpenCapData_*/Videos/Cam1")

for calib_dir in calib_sessions:
    session_id = calib_dir.split('/')[-3]

    # 创建配置
    config = {
        'calibration': {
            'use_external_calibration': True,
            'external_calibration_files': {
                'Cam1': f"{calib_dir}/cameraIntrinsicsExtrinsics.pickle",
                'Cam2': f"{calib_dir}/../Cam2/cameraIntrinsicsExtrinsics.pickle"
            }
        },
        # ... 其他配置
    }

    # 运行处理
    pipeline = LocalOpenCapPipeline(config_dict=config)
    pipeline.process_session(
        video_directory=f"Videos/{session_id}",
        static_directory=f"Static/{session_id}"
    )
```

## 总结

使用外部标定文件是排查本地标定问题的有效方法:

1. ✅ **快速验证** - 快速确定问题是否在标定环节
2. ✅ **对比分析** - 对比本地和官网标定的差异
3. ✅ **学习参考** - 学习官网标定的最佳实践
4. ✅ **混合使用** - 可以混合使用本地和外部标定

记住，外部标定文件只是临时解决方案，最终目标应该是:
- 优化本地标定流程
- 提高标定图像质量
- 正确选择标定方案
- 理解坐标系转换原理