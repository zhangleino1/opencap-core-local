"""
本地OpenCap处理管道使用示例
展示如何使用完全本地化的OpenCap流程处理运动捕获数据
"""

import os
# 设置本地模式环境变量，确保在导入其他模块前设置
os.environ['OPENCAP_LOCAL_MODE'] = 'true'
# 设置UTF-8编码，避免Windows下的GBK编码问题
os.environ['PYTHONIOENCODING'] = 'utf-8'

from local_opencap_pipeline import LocalOpenCapPipeline, run_local_opencap, create_config_template

# ==================== 示例1: 最简单的使用方式 ====================
def example_simple_usage():
    """最简单的使用方式 - 只需提供视频目录"""
    
    video_directory = "./LocalData/Videos"  # 包含运动视频的目录
    calibration_directory = "./LocalData/Calibration"  # 可选：标定视频目录
    
    # 一行代码处理整个会话
    success = run_local_opencap(
        video_dir=video_directory,
        calibration_dir=calibration_directory,  # 可选
        pose_detector='OpenPose',  # 或 'mmpose'
        resolution='1x736'  # OpenPose分辨率
    )
    
    if success:
        print("✅ 处理成功！检查Data目录中的结果")
    else:
        print("❌ 处理失败，请检查日志")

# ==================== 示例2: 使用配置文件 ====================
def example_with_config():
    """使用配置文件的方式"""
    
    # 1. 创建配置文件模板
    config_path = "local_config_template.yaml"
    create_config_template(config_path)
    print(f"配置文件模板已创建: {config_path}")
    print("请根据需要修改配置，然后重新运行")
    
    # 2. 使用配置文件处理
    success = run_local_opencap(
        video_dir="./LocalData/Videos",
        calibration_dir="./LocalData/Calibration",
        config_file=config_path
    )
    
    return success

# ==================== 示例3: 高级用法 - 分步处理 ====================
def example_advanced_usage():
    """高级用法 - 完全控制处理流程"""
    
    # 自定义配置
    config = {
        'session': {
            'name': 'MyCustomSession_20241209',
            'description': '我的运动捕获试验',
            'subject_mass': 75.0,  # 受试者体重(kg)
            'subject_height': 175.0  # 受试者身高(cm)
        },
        'calibration': {
            'checkerboard': {
                'dimensions': [9, 6],  # 不同的标定板规格
                'square_size': 50  # 毫米
            },
            'n_images': 30  # 使用较少标定图像
        },
        'processing': {
            'pose_detector': 'mmpose',  # 使用MMPose
            'bbox_threshold': 0.9,  # 更严格的检测阈值
            'augmenter_model': 'v0.2',  # 使用旧版本LSTM模型
            'resolution': 'default'
        }
    }
    
    # 创建管道实例
    pipeline = LocalOpenCapPipeline(config_dict=config)
    
    # 分步执行
    try:
        # 1. 创建会话元数据
        pipeline.create_session_metadata()
        
        # 2. 设置视频数据
        trial_name, camera_names = pipeline.setup_from_videos(
            video_directory="./LocalData/Videos",
            calibration_directory="./LocalData/Calibration"
        )
        
        # 3. 运行相机标定
        calib_success, camera_model = pipeline.run_calibration()
        
        # 4. 处理运动试验
        if calib_success:
            success = pipeline.process_trial(trial_name, camera_names, 'dynamic')
            
            if success:
                print("处理完成！")
                return True
        
        return False
        
    except Exception as e:
        print(f"处理异常: {e}")
        return False

# ==================== 示例4: 批量处理多个会话 ====================
def example_batch_processing():
    """批量处理多个会话"""
    
    sessions = [
        {
            'name': 'Subject01_Walking',
            'video_dir': './data/subject01/walking',
            'calib_dir': './data/subject01/calibration',
            'mass': 70.0,
            'height': 170.0
        },
        {
            'name': 'Subject01_Running', 
            'video_dir': './data/subject01/running',
            'calib_dir': None,  # 使用之前的标定
            'mass': 70.0,
            'height': 170.0
        }
    ]
    
    results = []
    for session in sessions:
        print(f"\n处理会话: {session['name']}")
        
        success = run_local_opencap(
            video_dir=session['video_dir'],
            calibration_dir=session['calib_dir'],
            **{
                'session.name': session['name'],
                'session.subject_mass': session['mass'],
                'session.subject_height': session['height']
            }
        )
        
        results.append({
            'session': session['name'],
            'success': success
        })
    
    # 打印结果摘要
    print("\n" + "="*50)
    print("批量处理结果:")
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"{status} {result['session']}")

# ==================== 数据目录结构示例 ====================
def show_directory_structure():
    """显示预期的目录结构"""
    structure = """
推荐的数据目录结构:

your_project/
├── videos/                    # 运动视频目录
│   ├── camera1_walking.mp4   # 摄像头1的运动视频
│   ├── camera2_walking.mp4   # 摄像头2的运动视频
│   └── ...
├── calibration/              # 标定视频目录（可选）
│   ├── camera1_calib.mp4     # 摄像头1的标定视频
│   ├── camera2_calib.mp4     # 摄像头2的标定视频
│   └── ...
└── config.yaml              # 配置文件（可选）

处理后的输出结构:
Data/
└── YourSessionName/
    ├── sessionMetadata.yaml
    ├── Videos/
    │   ├── camera1/
    │   └── camera2/
    ├── MarkerAugmenter/
    │   └── *.trc            # 3D标记点数据
    ├── OpenSimData/
    │   ├── *.osim          # OpenSim模型
    │   ├── *.mot           # 运动数据
    │   └── *.json          # 可视化数据
    └── OutputMedia*/
        └── *.mp4           # 处理后的视频
    """
    print(structure)

# ==================== 故障排除指南 ====================
def troubleshooting_guide():
    """故障排除指南"""
    guide = """
常见问题解决方案:

1. 相机标定失败:
   - 检查标定板参数(dimensions, square_size)是否正确
   - 确保标定视频包含清晰的棋盘格图像
   - 尝试增加n_images参数获取更多标定图像

2. 姿态检测失败:
   - 确保已正确安装OpenPose或配置MMPose
   - 检查视频中人体是否清晰可见
   - 尝试降低分辨率设置

3. 3D重建失败:
   - 确保至少有2个摄像头的视频
   - 检查相机标定是否成功
   - 验证视频同步是否正确

4. OpenSim分析失败:
   - 确保已安装OpenSim 4.4
   - 检查受试者质量和身高设置
   - 验证3D数据质量

5. 内存不足:
   - 降低OpenPose分辨率设置
   - 减少同时处理的视频数量
   - 关闭不必要的应用程序
    """
    print(guide)

if __name__ == "__main__":
    print("本地OpenCap处理管道使用示例")
    print("="*50)
    
    # 显示目录结构
    show_directory_structure()
    
    # 根据需要取消注释运行相应示例
    
    # 示例1: 简单使用
    # example_simple_usage()
    
    # 示例2: 配置文件
    example_with_config()
    
    # 示例3: 高级用法
    # example_advanced_usage()
    
    # 示例4: 批量处理
    # example_batch_processing()
    
    # 故障排除指南
    # troubleshooting_guide()