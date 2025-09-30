"""
使用外部标定文件进行OpenCap处理的示例
此脚本演示如何使用从OpenCap官网下载的标定文件来排除本地标定问题
"""

import os
import sys

# 设置本地模式环境变量
os.environ['OPENCAP_LOCAL_MODE'] = 'true'
os.environ['PYTHONIOENCODING'] = 'utf-8'

from local_opencap_pipeline import run_local_opencap

def example_with_external_calibration():
    """
    使用外部标定文件的示例

    步骤:
    1. 从OpenCap官网下载已处理的会话数据
    2. 提取其中的 cameraIntrinsicsExtrinsics.pickle 文件
    3. 在配置文件中指定这些外部标定文件的路径
    4. 运行处理，跳过本地标定步骤
    """

    print("="*80)
    print("🎯 使用外部标定文件进行OpenCap处理")
    print("="*80)
    print("\n此示例将:")
    print("  1. 跳过本地标定步骤")
    print("  2. 使用从OpenCap官网下载的标定文件")
    print("  3. 直接处理静态和动态试验")
    print("\n优点:")
    print("  ✅ 排除本地标定问题")
    print("  ✅ 使用经过验证的官网标定结果")
    print("  ✅ 快速测试其他处理步骤")
    print("="*80 + "\n")

    # 配置文件路径
    config_file = "config_external_calib.yaml"

    # 检查配置文件是否存在
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        print("请先创建配置文件并指定外部标定文件路径")
        return False

    # 视频目录
    video_dir = "./LocalData/Videos"
    static_dir = "./LocalData/Static"

    # 检查视频目录
    if not os.path.exists(video_dir):
        print(f"❌ 视频目录不存在: {video_dir}")
        return False

    print(f"📁 配置文件: {config_file}")
    print(f"📁 视频目录: {video_dir}")
    print(f"📁 静态目录: {static_dir}")
    print("\n开始处理...\n")

    try:
        # 运行处理流程
        success = run_local_opencap(
            video_dir=video_dir,
            calibration_dir=None,  # 不提供标定目录，因为使用外部标定文件
            static_dir=static_dir,
            config_file=config_file
        )

        if success:
            print("\n" + "="*80)
            print("✅ 处理成功完成！")
            print("="*80)
            print("\n📊 结果分析:")
            print("  - 如果姿态仍然异常，说明问题可能不在标定环节")
            print("  - 如果姿态正常，说明问题出在本地标定")
            print("\n💡 后续步骤:")
            print("  1. 检查 Data/ 目录下的处理结果")
            print("  2. 查看 OpenSim 中的人物姿态")
            print("  3. 对比本地标定和官网标定的差异")
        else:
            print("\n" + "="*80)
            print("❌ 处理失败")
            print("="*80)
            print("\n请检查:")
            print("  - 外部标定文件路径是否正确")
            print("  - 标定文件是否与当前视频匹配")
            print("  - 视频文件是否完整")

        return success

    except Exception as e:
        print(f"\n❌ 处理出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def setup_external_calibration_config():
    """
    交互式创建外部标定配置文件
    """
    import yaml

    print("="*80)
    print("🔧 配置外部标定文件")
    print("="*80)

    # 从官网下载的标定文件路径
    print("\n请输入从OpenCap官网下载的标定文件路径:")
    print("示例: D:/path/to/downloaded/OpenCapData_xxx/Videos/Cam1/cameraIntrinsicsExtrinsics.pickle")
    print()

    calibration_files = {}

    # 获取摄像头数量
    while True:
        try:
            num_cams = int(input("请输入摄像头数量 (1-4): ").strip())
            if 1 <= num_cams <= 4:
                break
            else:
                print("请输入1到4之间的数字")
        except ValueError:
            print("请输入有效的数字")

    # 获取每个摄像头的标定文件路径
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
                    break

    if not calibration_files:
        print("\n❌ 未配置任何标定文件")
        return False

    # 创建配置
    config = {
        'calibration': {
            'use_external_calibration': True,
            'external_calibration_files': calibration_files,
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
            'augmenter_model': 'v0.3'
        },
        'session': {
            'name': f"session_external_calib",
            'description': '使用外部标定文件的OpenCap会话',
            'subject_mass': 67.0,
            'subject_height': 170.0
        },
        'directories': {
            'input_videos': './LocalData/Videos',
            'static_videos': './LocalData/Static'
        }
    }

    # 保存配置
    config_file = 'config_external_calib.yaml'
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print("\n" + "="*80)
    print(f"✅ 配置文件已创建: {config_file}")
    print("="*80)
    print("\n配置内容:")
    print(yaml.dump(config, default_flow_style=False, allow_unicode=True))

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='使用外部标定文件进行OpenCap处理')
    parser.add_argument('--setup', action='store_true',
                       help='交互式创建外部标定配置文件')

    args = parser.parse_args()

    if args.setup:
        # 创建配置文件
        if setup_external_calibration_config():
            print("\n💡 配置完成！现在可以运行:")
            print("   python example_use_external_calibration.py")
        sys.exit(0)

    # 运行处理
    success = example_with_external_calibration()
    sys.exit(0 if success else 1)