"""
    本地版本相机内参标定脚本 - 无需远程API
    @authors: Scott Uhlrich, Antoine Falisse, Łukasz Kidziński
    
    基于main_calcIntrinsics.py修改，移除所有远程API依赖，支持完全本地标定
"""

import os 
import yaml
import pickle
import glob
import cv2
import numpy as np
from datetime import datetime

from utils import importMetadata
from utils import getDataDirectory
from utilsCameraPy3 import Camera

# 本地版本的内参计算函数
def computeAverageIntrinsicsLocal(session_path, trialIDs, CheckerBoardParams, nImages=25, cameraModel=None):
    """
    本地版本的平均内参计算 - 无需API调用
    
    Args:
        session_path: 会话目录路径
        trialIDs: 试验ID列表
        CheckerBoardParams: 标定板参数
        nImages: 每个视频使用的图像数量
        cameraModel: 手动指定摄像头型号（可选）
    
    Returns:
        CamParamsAverage: 平均内参
        CamParamList: 各试验内参列表
        intrinsicComparisons: 内参比较数据
        detectedCameraModel: 检测到的摄像头型号
    """
    
    CamParamList = []
    camModels = []
    intrinsicComparisons = {}
    
    print(f"开始处理 {len(trialIDs)} 个试验...")
    
    for i, trial_id in enumerate(trialIDs):
        trial_path = os.path.join(session_path, trial_id)
        
        if not os.path.exists(trial_path):
            print(f"警告: 试验目录不存在: {trial_path}")
            continue
            
        # 查找视频文件
        video_files = glob.glob(os.path.join(trial_path, "*.mp4"))
        if not video_files:
            print(f"警告: 试验 {trial_id} 中没有找到视频文件")
            continue
            
        # 处理每个视频（支持多摄像头）
        for video_file in video_files:
            print(f"处理视频: {os.path.basename(video_file)}")
            
            # 从视频文件名推断摄像头型号（如果未手动指定）
            if cameraModel is None:
                detected_model = extractCameraModelFromFilename(video_file)
                if detected_model:
                    camModels.append(detected_model)
                else:
                    # 使用默认型号
                    camModels.append(f"Camera_{i+1}")
            else:
                camModels.append(cameraModel)
            
            # 提取标定图像并计算内参
            intrinsic_data = calibrateCameraFromVideo(
                video_file, 
                CheckerBoardParams, 
                nImages
            )
            
            if intrinsic_data is not None:
                CamParamList.append(intrinsic_data)
                
                # 记录内参比较数据
                trial_key = f"{trial_id}_{os.path.basename(video_file)}"
                intrinsicComparisons[trial_key] = {
                    'reprojection_error': intrinsic_data.get('reprojectionError', 0),
                    'focal_length': [intrinsic_data['intrinsicMat'][0,0], intrinsic_data['intrinsicMat'][1,1]],
                    'principal_point': [intrinsic_data['intrinsicMat'][0,2], intrinsic_data['intrinsicMat'][1,2]],
                    'distortion_coeffs': intrinsic_data['distortion'].flatten().tolist()
                }
            else:
                print(f"标定失败: {video_file}")
    
    if not CamParamList:
        raise Exception("没有成功标定任何摄像头！请检查视频文件和标定板参数。")
    
    # 计算平均内参
    CamParamsAverage = computeAverageParameters(CamParamList)
    
    # 确定最终的摄像头型号
    if camModels:
        detectedCameraModel = max(set(camModels), key=camModels.count)  # 使用最常见的型号
    else:
        detectedCameraModel = cameraModel or f"LocalCamera_{datetime.now().strftime('%Y%m%d')}"
    
    print(f"标定完成!")
    print(f"检测到摄像头型号: {detectedCameraModel}")
    print(f"成功标定 {len(CamParamList)} 个视频")
    print(f"平均重投影误差: {np.mean([c.get('reprojectionError', 0) for c in CamParamList]):.2f} 像素")
    
    return CamParamsAverage, CamParamList, intrinsicComparisons, detectedCameraModel

def extractCameraModelFromFilename(video_file):
    """从视频文件名推断摄像头型号"""
    filename = os.path.basename(video_file).lower()
    
    # 常见摄像头型号匹配模式
    camera_patterns = {
        'iphone': r'iphone[\d\w,\.]+',
        'samsung': r'samsung[\d\w]+',
        'pixel': r'pixel[\d\w]+',
        'camera': r'camera[\d]+',
        'cam': r'cam[\d]+',
    }
    
    import re
    for brand, pattern in camera_patterns.items():
        match = re.search(pattern, filename)
        if match:
            return match.group(0).replace(',', '_').replace('.', '_')
    
    return None

def calibrateCameraFromVideo(video_file, CheckerBoardParams, nImages):
    """
    从单个视频文件标定摄像头内参
    
    Args:
        video_file: 视频文件路径
        CheckerBoardParams: 标定板参数
        nImages: 使用的图像数量
    
    Returns:
        dict: 内参数据 或 None（如果失败）
    """
    
    # 打开视频
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_file}")
        return None
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"  视频信息: {width}x{height}, {fps}fps, {total_frames}帧")
    
    # 设置标定参数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # 准备标定板角点
    objp = np.zeros((CheckerBoardParams['dimensions'][0] * CheckerBoardParams['dimensions'][1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:CheckerBoardParams['dimensions'][0], 0:CheckerBoardParams['dimensions'][1]].T.reshape(-1,2)
    objp = objp * CheckerBoardParams['squareSize']
    
    # 存储角点
    objpoints = []  # 3D点
    imgpoints = []  # 2D图像点
    
    # 选择均匀分布的帧
    frame_indices = np.linspace(0, total_frames-1, nImages*2, dtype=int)  # 多取一些以防检测失败
    
    valid_images = 0
    for frame_idx in frame_indices:
        if valid_images >= nImages:
            break
            
        # 跳转到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 寻找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, CheckerBoardParams['dimensions'], None)
        
        if ret:
            # 细化角点位置
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners2)
            valid_images += 1
    
    cap.release()
    
    if valid_images < 10:  # 最少需要10幅图像
        print(f"  标定失败: 只找到 {valid_images} 幅有效图像，少于最低要求(10幅)")
        return None
    
    print(f"  找到 {valid_images} 幅有效标定图像")
    
    # 标定摄像头
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (width, height), None, None
    )
    
    if not ret:
        print("  标定计算失败")
        return None
    
    # 计算重投影误差
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    
    reprojection_error = total_error / len(objpoints)
    
    print(f"  重投影误差: {reprojection_error:.2f} 像素")
    
    # 构造返回数据
    intrinsic_data = {
        'intrinsicMat': mtx,
        'distortion': dist,
        'imageSize': np.array([[width], [height]], dtype=np.float64),
        'reprojectionError': reprojection_error,
        'valid_images': valid_images,
        'rvecs': rvecs,
        'tvecs': tvecs
    }
    
    return intrinsic_data

def computeAverageParameters(CamParamList):
    """计算多次标定的平均内参"""
    
    if len(CamParamList) == 1:
        return CamParamList[0]
    
    # 平均内参矩阵
    intrinsic_mats = [params['intrinsicMat'] for params in CamParamList]
    avg_intrinsic = np.mean(intrinsic_mats, axis=0)
    
    # 平均畸变系数
    distortions = [params['distortion'] for params in CamParamList]
    avg_distortion = np.mean(distortions, axis=0)
    
    # 使用第一个的图像尺寸（假设都相同）
    image_size = CamParamList[0]['imageSize']
    
    # 平均重投影误差
    avg_reprojection_error = np.mean([params.get('reprojectionError', 0) for params in CamParamList])
    
    return {
        'intrinsicMat': avg_intrinsic,
        'distortion': avg_distortion,  
        'imageSize': image_size,
        'reprojectionError': avg_reprojection_error
    }

def saveCameraParametersLocal(filename, cameraParams):
    """保存摄像头参数到文件"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 只保存必要的参数
    params_to_save = {
        'intrinsicMat': cameraParams['intrinsicMat'],
        'distortion': cameraParams['distortion'],
        'imageSize': cameraParams['imageSize']
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(params_to_save, f)
    
    print(f"参数已保存到: {filename}")

# %% 主要配置参数 - 修改这些参数以适应你的设置
if __name__ == "__main__":
    # 基本配置
    sessionName = 'DualCamera_Local_20241209'  # 修改为你的会话名
    CheckerBoardParams = {'dimensions':(11,8),'squareSize':60}  # 标定板参数
    
    # 试验列表 - 修改为你的实际试验目录名
    trials = ['trial1', 'trial2', 'trial3']
    
    # 可选：手动指定摄像头型号（如果不指定将从文件名自动推断）  
    manualCameraModel = None  # 例如: 'iPhone15' 或 'CustomCamera'
    
    # 处理参数
    nImages = 50  # 每个视频使用的标定图像数量
    saveIntrinsicsForDeployment = True
    deployedFolderNames = ['Deployed_720_60fps','Deployed']
    
    # 数据路径
    dataDir = os.path.join(getDataDirectory(),'Data')
    sessionDir = os.path.join(dataDir,'IntrinsicCaptures', sessionName)
    
    print("="*60)
    print("本地相机内参标定")
    print("="*60)
    print(f"会话目录: {sessionDir}")
    print(f"试验数量: {len(trials)}")
    print(f"标定板规格: {CheckerBoardParams['dimensions'][0]}x{CheckerBoardParams['dimensions'][1]}")
    print(f"正方形边长: {CheckerBoardParams['squareSize']}mm")
    print("="*60)
    
    # 检查会话目录是否存在
    if not os.path.exists(sessionDir):
        print(f"错误: 会话目录不存在: {sessionDir}")
        print("请先创建目录结构并放入标定视频")
        exit(1)
    
    # 从元数据文件读取标定板参数（如果存在）
    metadataPath = os.path.join(sessionDir,'sessionMetadata.yaml')
    if os.path.exists(metadataPath):
        try:
            with open(metadataPath, 'r', encoding='utf-8') as f:
                sessionMetadata = yaml.safe_load(f)
            
            if 'checkerBoard' in sessionMetadata:
                CheckerBoardParams = {
                    'dimensions': (sessionMetadata['checkerBoard']['black2BlackCornersWidth_n'],
                                   sessionMetadata['checkerBoard']['black2BlackCornersHeight_n']),
                    'squareSize': sessionMetadata['checkerBoard']['squareSideLength_mm']}
                print('标定板参数已从元数据文件更新')
                print(f"更新后参数: {CheckerBoardParams}")
        except Exception as e:
            print(f"读取元数据文件时出错: {e}")
    
    try:
        # 执行本地标定
        CamParamsAverage, CamParamList, intrinsicComparisons, cameraModel = computeAverageIntrinsicsLocal(
            sessionDir, trials, CheckerBoardParams, nImages, manualCameraModel
        )
        
        # 保存标定结果
        if saveIntrinsicsForDeployment:
            for deployedFolderName in deployedFolderNames:
                permIntrinsicDir = os.path.join(os.getcwd(), 'CameraIntrinsics',
                                                cameraModel, deployedFolderName)
                intrinsicFile = os.path.join(permIntrinsicDir, 'cameraIntrinsics.pickle')
                saveCameraParametersLocal(intrinsicFile, CamParamsAverage)
        
        # 保存试验信息
        trialFile = os.path.join(sessionDir, 'trialInfo.yaml')
        trialInfo = {
            'trials': trials,
            'nSquaresWidth': CheckerBoardParams['dimensions'][0],
            'nSquaresHeight': CheckerBoardParams['dimensions'][1],
            'squareSize': CheckerBoardParams['squareSize'],
            'cameraModel': cameraModel,
            'calibration_date': datetime.now().isoformat(),
            'calibration_type': 'local',
            'total_videos_processed': len(CamParamList)
        }
        
        with open(trialFile, 'w', encoding='utf-8') as f:
            yaml.dump(trialInfo, f, allow_unicode=True)
        
        # 保存内参比较数据  
        intrinsicComparisonFile = os.path.join(sessionDir, 'intrinsicComparison.pkl')
        with open(intrinsicComparisonFile, 'wb') as f:
            pickle.dump(intrinsicComparisons, f)
        
        print("\n" + "="*60)
        print("✅ 本地标定成功完成!")
        print("="*60)
        print(f"摄像头型号: {cameraModel}")
        print(f"处理视频数: {len(CamParamList)}")
        print(f"平均重投影误差: {CamParamsAverage.get('reprojectionError', 0):.2f} 像素")
        print(f"焦距: fx={CamParamsAverage['intrinsicMat'][0,0]:.2f}, fy={CamParamsAverage['intrinsicMat'][1,1]:.2f}")
        print(f"主点: cx={CamParamsAverage['intrinsicMat'][0,2]:.2f}, cy={CamParamsAverage['intrinsicMat'][1,2]:.2f}")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 标定失败: {str(e)}")
        print("请检查:")
        print("1. 视频文件是否存在于试验目录中") 
        print("2. 标定板参数是否正确")
        print("3. 视频中是否包含清晰的标定板图像")
        exit(1)