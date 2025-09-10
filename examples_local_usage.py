"""
本地OpenCap处理管道使用示例
展示如何使用本地化的OpenCap流程处理运动捕获数据
"""

import os
# 设置本地模式环境变量
os.environ['OPENCAP_LOCAL_MODE'] = 'true'
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 导入本地管道
from local_opencap_pipeline import LocalOpenCapPipeline, run_local_opencap, create_config_template

def example_simple_usage():
    """最简单的使用方式"""
    
    success = run_local_opencap(
        video_dir="./LocalData/Videos",
        calibration_dir="./LocalData/Calibration",  # 确保提供标定目录
        static_dir="./LocalData/Static",
        pose_detector='OpenPose',
        resolution='1x736'
    )
    
    if success:
        print("✅ 处理成功！")
    else:
        print("❌ 处理失败")

def example_with_config():
    """使用配置文件"""
    
    config_path = "my_config.yaml"
    create_config_template(config_path)
    print(f"配置文件已创建: {config_path}")
    
    success = run_local_opencap(
        video_dir="./LocalData/Videos",
        calibration_dir="./LocalData/Calibration",  # 确保提供标定目录
        static_dir="./LocalData/Static",
        config_file=config_path
    )
    
    return success

def show_directory_structure():
    """显示目录结构"""
    print("""
    推荐的数据目录结构:

    your_project/
    ├── videos/                    # 运动视频目录
    │   ├── camera1_walking.mp4   
    │   ├── camera2_walking.mp4   
    │   └── ...
    ├── calibration/              # 标定视频目录（可选）
    │   ├── camera1_calib.mp4     
    │   ├── camera2_calib.mp4     
    │   └── ...
    ├── static/                   # 静态姿态视频目录（可选）
    │   ├── camera1_static.mp4    
    │   ├── camera2_static.mp4    
    │   └── ...
    └── config.yaml              # 配置文件（可选）

    处理后的输出结构:
    Data/
    └── YourSessionName/
        ├── sessionMetadata.yaml
        ├── Videos/
        ├── MarkerData/          # 3D标记点数据
        ├── OpenSimData/         # OpenSim模型和运动数据
        └── VisualizerVideos/    # 处理后的视频
        """)

if __name__ == "__main__":
    print("本地OpenCap处理管道使用示例")
    show_directory_structure()
    print("\n📋 配置文件使用方式:")
    example_with_config()
    
    # print("\n🔧 简单使用方式:")
    # example_simple_usage()
    
    print("✅ 示例运行完成！")
    print("📖 本地管道提供了完整的OpenCap功能。")