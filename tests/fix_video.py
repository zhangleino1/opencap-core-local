#!/usr/bin/env python3
"""
视频修复工具 - 使用ffmpeg修复指定目录及子目录下的所有视频文件
"""

import os
import sys
import subprocess
import glob

def fix_video_with_ffmpeg(input_path, output_path=None):
    """使用ffmpeg修复视频文件，保持原始质量"""
    
    if output_path is None:
        # 默认输出路径：原文件名_fixed.mp4
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_fixed{ext}"
    
    print(f"修复视频: {input_path}")
    print(f"输出到: {output_path}")
    print("=" * 50)
    
    if not os.path.exists(input_path):
        print(f"[错误] 输入文件不存在: {input_path}")
        return False
    
    # 构建ffmpeg命令 - 无损修复，保持原始质量
    # -err_detect ignore_err: 忽略错误继续处理
    # -c copy: 复制流不重新编码，保持原始质量
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-err_detect', 'ignore_err',
        '-c', 'copy',  # 复制流，不重新编码
        '-y',  # 覆盖输出文件
        output_path
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    print()
    
    try:
        # 运行ffmpeg
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            print(f"[成功] 视频修复完成: {output_path}")
            
            # 检查输出文件大小
            if os.path.exists(output_path):
                input_size = os.path.getsize(input_path)
                output_size = os.path.getsize(output_path)
                print(f"原文件大小: {input_size:,} bytes")
                print(f"修复后大小: {output_size:,} bytes")
                print(f"大小变化: {((output_size-input_size)/input_size*100):+.1f}%")
            
            return True
        else:
            print(f"[错误] ffmpeg执行失败 (返回码: {result.returncode})")
            if result.stderr:
                print(f"错误信息: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("[错误] 找不到ffmpeg程序")
        print("请确保ffmpeg已安装并在PATH环境变量中")
        print("下载地址: https://ffmpeg.org/download.html")
        return False
    except Exception as e:
        print(f"[错误] 执行异常: {e}")
        return False

def find_video_files(directory):
    """在指定目录及子目录中查找所有视频文件"""
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm', '*.m4v']
    video_files = []
    
    for ext in video_extensions:
        # 递归查找所有子目录中的视频文件
        pattern = os.path.join(directory, '**', ext)
        video_files.extend(glob.glob(pattern, recursive=True))
    
    return sorted(video_files)

def process_directory(directory):
    """处理指定目录下的所有视频文件"""
    if not os.path.exists(directory):
        print(f"[错误] 目录不存在: {directory}")
        return False
    
    if not os.path.isdir(directory):
        print(f"[错误] 路径不是目录: {directory}")
        return False
    
    print(f"搜索目录: {directory}")
    video_files = find_video_files(directory)
    
    if not video_files:
        print("未找到视频文件")
        return True
    
    print(f"找到 {len(video_files)} 个视频文件:")
    for i, video_file in enumerate(video_files, 1):
        print(f"  {i}. {video_file}")
    print()
    
    success_count = 0
    failed_count = 0
    
    for i, video_file in enumerate(video_files, 1):
        print(f"处理 {i}/{len(video_files)}: {video_file}")
        
        # 跳过已经修复的文件
        if '_fixed' in os.path.basename(video_file):
            print("跳过已修复文件")
            print()
            continue
        
        success = fix_video_with_ffmpeg(video_file)
        if success:
            success_count += 1
        else:
            failed_count += 1
        print()
    
    print("=" * 50)
    print(f"处理完成!")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {failed_count} 个文件")
    
    return failed_count == 0

def main():
    if len(sys.argv) < 2:
        print("用法:")
        print("  python fix_video.py <目录路径>")
        print()
        print("功能:")
        print("  修复指定目录及其所有子目录中的视频文件")
        print("  保持原始视频质量，不进行压缩")
        print()
        print("示例:")
        print("  python fix_video.py /path/to/video/directory")
        print("  python fix_video.py C:\\Videos\\MyProject")
        sys.exit(1)
    
    directory = sys.argv[1]
    success = process_directory(directory)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()