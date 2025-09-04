# 双摄像头标定教程

## 概述
本教程介绍如何使用OpenCap Core对两个摄像头进行标定，获取内参和外参用于3D重建。

## 准备工作

### 1. 硬件要求
- 2个摄像头（相同或不同型号）
- 标定板（棋盘格）
- 稳定的拍摄环境

### 2. 标定板参数
- 默认规格：11x8个内角点，60mm边长
- 可在sessionMetadata.yaml中自定义

### 3. 数据目录结构
```
Data/
└── IntrinsicCaptures/
    └── DualCamera_720_60FPS_20241209/  # 会话名称
        ├── sessionMetadata.yaml         # 标定板参数
        ├── trial_uuid_1/               # 试验1
        │   ├── cam1_calibration.mp4    # 摄像头1视频
        │   └── cam2_calibration.mp4    # 摄像头2视频
        ├── trial_uuid_2/               # 试验2
        │   ├── cam1_calibration.mp4
        │   └── cam2_calibration.mp4
        └── trial_uuid_3/               # 试验3（建议3+次试验）
            ├── cam1_calibration.mp4
            └── cam2_calibration.mp4
```

## 配置标定脚本

### 1. 编辑 main_calcIntrinsics.py

修改以下关键参数：

```python
# 会话名称
sessionName = 'DualCamera_720_60FPS_20241209'

# 标定板参数（如有sessionMetadata.yaml会自动覆盖）
CheckerBoardParams = {'dimensions':(11,8),'squareSize':60}

# 试验列表（每个试验包含双摄像头视频）
trials = ['trial_uuid_1', 'trial_uuid_2', 'trial_uuid_3']

# 保存选项
loadTrialInfo = False  # 是否从文件加载试验信息
saveIntrinsicsForDeployment = True  # 保存到部署目录

# 部署目录名（用于保存标定结果）
deployedFolderNames = ['Deployed_720_60fps','Deployed']
```

### 2. 创建sessionMetadata.yaml

```yaml
checkerBoard:
  black2BlackCornersWidth_n: 11    # 棋盘格宽度内角点数
  black2BlackCornersHeight_n: 8    # 棋盘格高度内角点数  
  squareSideLength_mm: 60          # 棋盘格边长(毫米)
```

## 拍摄标定视频

### 1. 拍摄要求
- 每个试验同时录制两个摄像头
- 视频分辨率保持一致（如720p）
- 帧率保持一致（如60fps）
- 充足光照，避免过曝和阴影

### 2. 标定板移动模式
- **平移**：在图像平面内移动标定板
- **旋转**：绕x、y、z轴旋转标定板
- **深度变化**：改变标定板与摄像头的距离
- **边缘覆盖**：确保标定板出现在图像边缘区域

### 3. 质量检查
- 标定板完全可见且清晰
- 角点检测无遮挡
- 每个摄像头视频包含50+个有效帧

## 运行标定

### 1. 执行标定脚本
```bash
cd d:\work\researchcode\opencap-core
python main_calcIntrinsics.py
```

### 2. 标定流程
脚本会自动：
1. 读取配置参数
2. 检测每个视频中的角点
3. 计算每个摄像头的内参
4. 平均多次试验的结果
5. 保存标定参数到指定目录

### 3. 输出文件
```
CameraIntrinsics/
├── Camera1Model/
│   ├── Deployed/
│   │   └── cameraIntrinsics.pickle
│   └── Deployed_720_60fps/
│       └── cameraIntrinsics.pickle
└── Camera2Model/
    ├── Deployed/
    │   └── cameraIntrinsics.pickle
    └── Deployed_720_60fps/
        └── cameraIntrinsics.pickle
```

## 验证标定结果

### 1. 检查重投影误差
标定过程中会输出每个摄像头的重投影误差，通常应小于1像素。

### 2. 使用测试脚本验证
```bash
cd d:\work\researchcode\opencap-core
python tests\test_read_camera_intrinsics.py
```

### 3. 参数合理性检查
- **焦距**：应接近摄像头实际焦距
- **主点**：应接近图像中心
- **畸变系数**：径向畸变为主，切向畸变较小

## 常见问题

### 1. 角点检测失败
- 检查光照条件
- 确保标定板平整
- 调整图像对比度

### 2. 重投影误差过大
- 增加标定图像数量
- 改善标定板姿态多样性
- 检查标定板尺寸测量精度

### 3. 摄像头模型识别错误
- 确保视频文件命名包含摄像头型号信息
- 手动指定摄像头模型参数

## 后续使用

标定完成后，内参文件可用于：
- 3D重建和三角化
- 立体视觉深度估计
- 摄像头标定精度评估
- OpenCap运动捕获流水线

双摄像头标定为后续的多视角3D重建提供了精确的几何基础。