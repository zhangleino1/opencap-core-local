"""
快速启动脚本 - 使用外部标定文件
快速配置并运行使用外部标定文件的处理流程
"""

import os
import sys
import yaml
import glob

def check_external_calibration_files():
    """
    检查从OpenCap下载的标定文件
    """
    print("="*80)
    print("📂 查找OpenCap下载的标定文件")
    print("="*80)

    # 搜索下载目录
    search_patterns = [
        "Data/OpenCapData_*/OpenCapData_*/Videos/Cam*/cameraIntrinsicsExtrinsics.pickle",
        "Downloads/OpenCapData_*/OpenCapData_*/Videos/Cam*/cameraIntrinsicsExtrinsics.pickle",
        "*/OpenCapData_*/Videos/Cam*/cameraIntrinsicsExtrinsics.pickle"
    ]

    found_files = {}

    for pattern in search_patterns:
        files = glob.glob(pattern)
        for file in files:
            # 提取摄像头名称
            parts = file.split(os.sep)
            for i, part in enumerate(parts):
                if part.startswith('Cam'):
                    cam_name = part
                    if cam_name not in found_files:
                        found_files[cam_name] = file
                    break

    if found_files:
        print("\n✅ 找到以下标定文件:")
        for cam_name, file_path in sorted(found_files.items()):
            print(f"   {cam_name}: {file_path}")
        return found_files
    else:
        print("\n⚠️  未自动找到标定文件")
        print("请手动指定标定文件路径")
        return None


def create_quick_config(external_calibration_files):
    """
    创建快速配置文件
    """
    config = {
        'calibration': {
            'use_external_calibration': True,
            'external_calibration_files': external_calibration_files,
            'checkerboard': {
                'dimensions': [5, 4],
                'square_size': 35,
                'placement': 'backWall'
            },
            'interactive_selection': False
        },
        'processing': {
            'pose_detector': 'mmpose',
            'resolution': '1x736',
            'image_upsample_factor': 4,
            'augmenter_model': 'v0.3',
            'bbox_threshold': 0.8
        },
        'session': {
            'name': 'session_external_calib_quick',
            'description': '使用外部标定文件的快速测试',
            'subject_mass': 67.0,
            'subject_height': 170.0
        },
        'directories': {
            'input_videos': './LocalData/Videos',
            'static_videos': './LocalData/Static',
            'calibration_videos': './LocalData/Calibration',
            'output': './LocalData/Results'
        },
        'output': {
            'delete_intermediate': False,
            'generate_opensim': True,
            'save_videos': True
        }
    }

    config_file = 'my_config.yaml'
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"\n✅ 配置文件已创建: {config_file}")
    return config_file


def main():
    print("="*80)
    print("🚀 OpenCap 外部标定文件 - 快速启动")
    print("="*80)

    # 1. 自动查找标定文件
    auto_found_files = check_external_calibration_files()

    # 2. 获取标定文件路径
    calibration_files = {}

    if auto_found_files:
        use_auto = input("\n是否使用自动找到的标定文件? (y/n): ").strip().lower()
        if use_auto == 'y':
            calibration_files = auto_found_files
        else:
            print("\n请手动输入标定文件路径")

    # 如果没有自动找到或用户选择手动输入
    if not calibration_files:
        print("\n" + "="*80)
        print("📝 手动配置标定文件")
        print("="*80)
        print("\n请输入从OpenCap官网下载的标定文件路径")
        print("示例: D:/path/to/OpenCapData_xxx/Videos/Cam1/cameraIntrinsicsExtrinsics.pickle")

        # 获取摄像头数量
        while True:
            try:
                num_cams = int(input("\n请输入摄像头数量 (1-4): ").strip())
                if 1 <= num_cams <= 4:
                    break
                else:
                    print("请输入1到4之间的数字")
            except ValueError:
                print("请输入有效的数字")

        # 获取每个摄像头的标定文件
        for i in range(1, num_cams + 1):
            cam_name = f"Cam{i}"
            while True:
                path = input(f"\n{cam_name} 的标定文件路径: ").strip().strip('"')
                if os.path.exists(path):
                    calibration_files[cam_name] = path
                    print(f"✅ {cam_name}: {path}")
                    break
                else:
                    print(f"❌ 文件不存在: {path}")
                    retry = input("重新输入? (y/n): ").strip().lower()
                    if retry != 'y':
                        print(f"跳过 {cam_name}")
                        break

    if not calibration_files:
        print("\n❌ 未配置任何标定文件，退出")
        return False

    # 3. 创建配置文件
    print("\n" + "="*80)
    print("⚙️  创建配置文件")
    print("="*80)
    config_file = create_quick_config(calibration_files)

    # 4. 检查视频目录
    print("\n" + "="*80)
    print("📹 检查视频目录")
    print("="*80)

    video_dir = "./LocalData/Videos"
    static_dir = "./LocalData/Static"

    video_exists = os.path.exists(video_dir)
    static_exists = os.path.exists(static_dir)

    print(f"\n动态视频目录: {video_dir} {'✓' if video_exists else '✗'}")
    print(f"静态视频目录: {static_dir} {'✓' if static_exists else '✗'}")

    if not video_exists:
        print("\n❌ 未找到视频目录，请确保视频文件在正确位置")
        print("目录结构应为:")
        print("  LocalData/")
        print("  ├── Videos/     # 动态视频")
        print("  └── Static/     # 静态视频(可选)")
        return False

    # 5. 运行处理
    print("\n" + "="*80)
    print("🎬 开始处理")
    print("="*80)

    run_now = input("\n是否立即开始处理? (y/n): ").strip().lower()

    if run_now != 'y':
        print("\n配置已完成，稍后可以运行:")
        print("  python examples_local_usage.py")
        return True

    print("\n正在启动处理流程...\n")

    from local_opencap_pipeline import run_local_opencap

    try:
        success = run_local_opencap(
            video_dir=video_dir,
            calibration_dir=None,  # 使用外部标定，不需要标定目录
            static_dir=static_dir if static_exists else None,
            config_file=config_file
        )

        if success:
            print("\n" + "="*80)
            print("✅ 处理成功完成！")
            print("="*80)
            print("\n📊 后续步骤:")
            print("  1. 检查 Data/ 目录下的处理结果")
            print("  2. 在OpenSim中打开生成的模型和运动文件")
            print("  3. 检查人物姿态是否正常")
            print("\n💡 分析建议:")
            print("  - 如果姿态正常: 说明本地标定存在问题")
            print("  - 如果姿态异常: 问题可能在其他处理环节")
        else:
            print("\n" + "="*80)
            print("❌ 处理失败")
            print("="*80)
            print("\n请检查日志了解详细错误信息")

        return success

    except Exception as e:
        print(f"\n❌ 处理出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 设置环境变量
    os.environ['OPENCAP_LOCAL_MODE'] = 'true'
    os.environ['PYTHONIOENCODING'] = 'utf-8'

    success = main()
    sys.exit(0 if success else 1)