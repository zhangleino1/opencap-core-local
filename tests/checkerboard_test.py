#!/usr/bin/env python3
"""
棋盘格检测测试工具 - 测试不同尺寸的棋盘格检测
"""

import cv2
import numpy as np
import os
import sys

def test_checkerboard_detection(video_path, dimensions_list=None):
    """测试不同棋盘格尺寸的检测效果"""
    
    if dimensions_list is None:
        # 常见的棋盘格尺寸
        dimensions_list = [
            (5, 4),    # 正确配置
            (4, 5),    # 旋转90度
            (6, 4),    # 可能的尺寸
            (7, 5),    # 可能的尺寸
            (8, 6),    # 较大尺寸
            (9, 6),    # 较大尺寸
        ]
    
    print(f"测试视频: {video_path}")
    print("=" * 60)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[错误] 无法打开视频")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    test_frames = min(10, total_frames)  # 测试前10帧
    
    # 测试每种棋盘格尺寸
    for dimensions in dimensions_list:
        print(f"\n[测试] 棋盘格尺寸: {dimensions[0]}x{dimensions[1]}")
        
        detected_count = 0
        
        for frame_idx in range(0, test_frames * 10, 10):  # 每10帧测试一次
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret or frame is None:
                continue
            
            # 转换为灰度图
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 检测棋盘格
            ret, corners = cv2.findChessboardCorners(frame, dimensions, None)
            
            if ret:
                detected_count += 1
                print(f"  帧 {frame_idx}: 检测到棋盘格 [OK]")
            else:
                print(f"  帧 {frame_idx}: 未检测到棋盘格 [FAIL]")
        
        detection_rate = detected_count / test_frames * 100
        print(f"  检测成功率: {detection_rate:.1f}% ({detected_count}/{test_frames})")
        
        if detection_rate >= 80:
            print(f"  [推荐] 该尺寸检测效果很好！")
        elif detection_rate >= 50:
            print(f"  [可用] 该尺寸检测效果一般")
        else:
            print(f"  [不推荐] 该尺寸检测效果较差")
    
    cap.release()

def extract_sample_frame(video_path, output_path="sample_frame.jpg"):
    """提取一帧样本图像用于手动检查"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    
    # 跳到中间位置
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    
    ret, frame = cap.read()
    cap.release()
    
    if ret and frame is not None:
        cv2.imwrite(output_path, frame)
        print(f"样本帧已保存到: {output_path}")
        return True
    
    return False

def main():


    video_path = 'E:/guge/opencap-core-local/Data/LocalSession_20250906_200947/Videos/camera1/calibration.mp4'


    test_checkerboard_detection(video_path)

if __name__ == "__main__":
    main()