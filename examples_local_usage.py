"""
本地OpenCap处理管道使用示例
展示如何使用本地化的OpenCap流程处理运动捕获数据
"""

import os
import sys
import signal
import threading
import traceback
import time
from datetime import datetime

# 设置本地模式环境变量
os.environ['OPENCAP_LOCAL_MODE'] = 'true'
os.environ['PYTHONIOENCODING'] = 'utf-8'

def print_all_threads():
    """打印所有线程的栈信息"""
    print("\n" + "="*60)
    print(f"🔍 线程栈追踪 - {datetime.now().strftime('%H:%M:%S')}")
    print("="*60)
    
    for thread_id, frame in sys._current_frames().items():
        thread = None
        for t in threading.enumerate():
            if t.ident == thread_id:
                thread = t
                break
        
        thread_name = thread.name if thread else f"Unknown-{thread_id}"
        print(f"\n📍 线程: {thread_name} (ID: {thread_id})")
        print("-" * 40)
        
        # 打印栈信息
        stack = traceback.format_stack(frame)
        for line in stack[-10:]:  # 只显示最近10层调用
            print(line.strip())

def signal_handler(signum, frame):
    """信号处理器，用于打印栈信息"""
    print(f"\n⚠️  收到信号 {signum}")
    print_all_threads()
    
    # 询问是否退出
    try:
        response = input("\n❓ 是否退出程序? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            print("🚪 正在退出...")
            os._exit(1)
    except KeyboardInterrupt:
        print("\n🚪 强制退出...")
        os._exit(1)

def setup_debug_handlers():
    """设置调试信号处理器"""
    # Windows 和 Linux 通用的信号
    if hasattr(signal, 'SIGINT'):
        signal.signal(signal.SIGINT, signal_handler)
    
    # Linux/Mac 特有信号
    if hasattr(signal, 'SIGUSR1'):
        signal.signal(signal.SIGUSR1, signal_handler)
    
    print("🔧 调试模式已启用:")
    print("   - 按 Ctrl+C 查看线程栈并选择是否退出")
    if hasattr(signal, 'SIGUSR1'):
        print("   - 发送 SIGUSR1 信号查看线程栈")
    print()

def monitor_progress(func, *args, **kwargs):
    """监控函数执行进度"""
    start_time = time.time()
    result = None
    exception = None
    
    def target():
        nonlocal result, exception
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            exception = e
    
    # 启动目标线程
    thread = threading.Thread(target=target, name="OpenCap-Main")
    thread.daemon = True
    thread.start()
    
    # 监控线程
    check_interval = 30  # 30秒检查一次
    last_check = start_time
    
    while thread.is_alive():
        time.sleep(1)
        current_time = time.time()
        
        if current_time - last_check > check_interval:
            elapsed = int(current_time - start_time)
            print(f"⏱️  程序运行中... 已用时: {elapsed//60}:{elapsed%60:02d}")
            last_check = current_time
    
    # 等待线程完成
    thread.join()
    
    if exception:
        raise exception
    
    return result

# 导入本地管道
from local_opencap_pipeline import LocalOpenCapPipeline, run_local_opencap, create_config_template

def example_simple_usage():
    """最简单的使用方式"""
    print("🚀 开始简单使用示例...")
    
    def run_pipeline():
        return run_local_opencap(
            video_dir="./LocalData/Videos",
            calibration_dir="./LocalData/Calibration",  # 确保提供标定目录
            static_dir="./LocalData/Static",
            pose_detector='OpenPose',
            resolution='1x736'
        )
    
    try:
        success = monitor_progress(run_pipeline)
        
        if success:
            print("✅ 处理成功！")
        else:
            print("❌ 处理失败")
        return success
    except KeyboardInterrupt:
        print("\n⚠️  用户中断了处理过程")
        return False
    except Exception as e:
        print(f"❌ 处理出错: {str(e)}")
        print_all_threads()
        return False

def example_with_config():
    """使用配置文件"""
    print("🚀 开始配置文件示例...")
    
    config_path = "my_config.yaml"
    create_config_template(config_path)
    print(f"配置文件已创建: {config_path}")
    
    def run_pipeline():
        return run_local_opencap(
            video_dir="./LocalData/Videos",
            calibration_dir="./LocalData/Calibration",  # 确保提供标定目录
            static_dir="./LocalData/Static",
            config_file=config_path
        )
    
    try:
        success = monitor_progress(run_pipeline)
        return success
    except KeyboardInterrupt:
        print("\n⚠️  用户中断了处理过程")
        return False
    except Exception as e:
        print(f"❌ 处理出错: {str(e)}")
        print_all_threads()
        return False

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
    print("=" * 50)
    
    # 设置调试处理器
    setup_debug_handlers()
    
    show_directory_structure()
    
    print("\n📋 配置文件使用方式:")
    try:
        example_with_config()
    except Exception as e:
        print(f"配置文件示例失败: {str(e)}")
        print_all_threads()
    
    # print("\n🔧 简单使用方式:")
    # example_simple_usage()
    
    print("✅ 示例运行完成！")
    print("📖 本地管道提供了完整的OpenCap功能。")