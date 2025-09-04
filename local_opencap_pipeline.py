"""
完全本地化的OpenCap处理管道
@authors: 基于OpenCap Core修改

本脚本实现了完全离线的运动捕获处理流程，包括：
1. 本地相机标定 (支持单摄像头/多摄像头)
2. 姿态检测 (OpenPose/MMPose)
3. 视频同步
4. 3D三角化重建
5. 标记点增强 (LSTM)
6. OpenSim生物力学分析

无需网络连接，无需API认证，完全本地运行。
"""

import os
import sys
import glob
import shutil
import yaml
import pickle
import numpy as np
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入OpenCap核心模块
from utils import getDataDirectory, importMetadata
from main import main as opencap_main

# 导入本地标定模块
from main_calcIntrinsics_local import (
    computeAverageIntrinsicsLocal, 
    saveCameraParametersLocal,
    extractCameraModelFromFilename
)

class LocalOpenCapPipeline:
    """完全本地化的OpenCap处理管道"""
    
    def __init__(self, config_file=None, config_dict=None):
        """
        初始化本地OpenCap管道
        
        Args:
            config_file: YAML配置文件路径
            config_dict: 配置字典（如果不使用文件）
        """
        self.config = self._load_config(config_file, config_dict)
        self._validate_config()
        self._setup_directories()
        
    def _load_config(self, config_file, config_dict):
        """加载配置"""
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        elif config_dict:
            return config_dict
        else:
            return self._get_default_config()
    
    def _get_default_config(self):
        """获取默认配置"""
        return {
            'session': {
                'name': f'LocalSession_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'description': '本地OpenCap会话',
                'subject_mass': 70.0,  # kg
                'subject_height': 170.0  # cm
            },
            'calibration': {
                'checkerboard': {
                    'dimensions': [11, 8],  # [width, height] 内角点数
                    'square_size': 60  # mm
                },
                'n_images': 50,
                'deployed_folders': ['Deployed_720_60fps', 'Deployed']
            },
            'processing': {
                'pose_detector': 'OpenPose',  # 'OpenPose' or 'mmpose'
                'resolution': '1x736',  # OpenPose分辨率
                'bbox_threshold': 0.8,  # mmpose边界框阈值
                'image_upsample_factor': 4,
                'augmenter_model': 'v0.3',  # LSTM增强模型版本
                'filter_frequency': 'default',
                'scaling_setup': 'upright_standing_pose'
            },
            'output': {
                'save_videos': True,
                'delete_intermediate': False,
                'generate_opensim': True
            },
            'directories': {
                'input_videos': './LocalData/Videos',
                'calibration_videos': './LocalData/Calibration',
                'output': './LocalData/Results'
            }
        }
    
    def _validate_config(self):
        """验证配置参数"""
        required_keys = ['session', 'calibration', 'processing', 'directories']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"配置中缺少必需的键: {key}")
        
        # 验证姿态检测器
        valid_detectors = ['OpenPose', 'mmpose']
        detector = self.config['processing']['pose_detector']
        if detector not in valid_detectors:
            raise ValueError(f"不支持的姿态检测器: {detector}. 支持: {valid_detectors}")
    
    def _setup_directories(self):
        """设置目录结构"""
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 数据目录
        data_dir = getDataDirectory()
        self.session_dir = os.path.join(data_dir, 'Data', self.config['session']['name'])
        
        # 创建目录结构
        dirs_to_create = [
            self.session_dir,
            os.path.join(self.session_dir, 'Videos'),
            os.path.join(self.session_dir, 'CalibrationImages'),
            os.path.join(self.session_dir, 'MarkerAugmenter'),
            os.path.join(self.session_dir, 'OpenSimData')
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
        
        logger.info(f"会话目录已创建: {self.session_dir}")
    
    def create_session_metadata(self):
        """创建会话元数据文件"""
        metadata = {
            'sessionWithCalibration': True,
            'mass_kg': self.config['session']['subject_mass'],
            'height_m': self.config['session']['subject_height'] / 100.0,
            'sessionDate': datetime.now().isoformat(),
            'sessionDescription': self.config['session']['description'],
            
            # 标定板参数
            'checkerBoard': {
                'black2BlackCornersWidth_n': self.config['calibration']['checkerboard']['dimensions'][0],
                'black2BlackCornersHeight_n': self.config['calibration']['checkerboard']['dimensions'][1],
                'squareSideLength_mm': self.config['calibration']['checkerboard']['square_size']
            },
            
            # 处理参数
            'poseDetector': self.config['processing']['pose_detector'],
            'resolutionPoseDetection': self.config['processing']['resolution'],
            'augmenter_model': self.config['processing']['augmenter_model'],
            'imageUpsampleFactor': self.config['processing']['image_upsample_factor'],
            
            # 本地处理标识
            'localProcessing': True,
            'created_by': 'LocalOpenCapPipeline'
        }
        
        metadata_path = os.path.join(self.session_dir, 'sessionMetadata.yaml')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            yaml.dump(metadata, f, allow_unicode=True, default_flow_style=False)
        
        logger.info(f"会话元数据已保存: {metadata_path}")
        return metadata_path
    
    def setup_from_videos(self, video_directory, calibration_directory=None):
        """
        从视频目录设置会话数据
        
        Args:
            video_directory: 包含运动视频的目录
            calibration_directory: 包含标定视频的目录（可选）
        """
        if not os.path.exists(video_directory):
            raise FileNotFoundError(f"视频目录不存在: {video_directory}")
        
        # 复制运动视频
        video_files = glob.glob(os.path.join(video_directory, "*.mp4"))
        if not video_files:
            raise FileNotFoundError(f"在 {video_directory} 中没有找到.mp4文件")
        
        # 按摄像头组织视频文件
        cameras = self._organize_videos_by_camera(video_files)
        
        # 创建试验结构
        trial_name = f"trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        trial_dir = os.path.join(self.session_dir, trial_name)
        os.makedirs(trial_dir, exist_ok=True)
        
        # 为每个摄像头创建目录并复制视频
        for camera_name, video_file in cameras.items():
            camera_dir = os.path.join(self.session_dir, 'Videos', camera_name)
            os.makedirs(camera_dir, exist_ok=True)
            
            dest_path = os.path.join(camera_dir, f"{trial_name}.mp4")
            shutil.copy2(video_file, dest_path)
            logger.info(f"视频已复制: {video_file} -> {dest_path}")
        
        # 处理标定视频（如果提供）
        if calibration_directory and os.path.exists(calibration_directory):
            self._setup_calibration_videos(calibration_directory, cameras.keys())
        
        return trial_name, list(cameras.keys())
    
    def _organize_videos_by_camera(self, video_files):
        """根据文件名将视频按摄像头分组"""
        cameras = {}
        
        for video_file in video_files:
            filename = os.path.basename(video_file)
            
            # 尝试从文件名推断摄像头名称
            camera_name = extractCameraModelFromFilename(video_file)
            
            if camera_name is None:
                # 使用文件名前缀作为摄像头名称
                base_name = os.path.splitext(filename)[0]
                # 移除常见的后缀
                suffixes_to_remove = ['_video', '_capture', '_recording']
                for suffix in suffixes_to_remove:
                    if base_name.endswith(suffix):
                        base_name = base_name[:-len(suffix)]
                        break
                camera_name = base_name or f"camera_{len(cameras)+1}"
            
            cameras[camera_name] = video_file
        
        logger.info(f"检测到 {len(cameras)} 个摄像头: {list(cameras.keys())}")
        return cameras
    
    def _setup_calibration_videos(self, calibration_directory, camera_names):
        """设置标定视频"""
        calib_files = glob.glob(os.path.join(calibration_directory, "*.mp4"))
        
        if not calib_files:
            logger.warning(f"标定目录中没有找到.mp4文件: {calibration_directory}")
            return
        
        # 创建标定试验目录
        calib_trial = "calibration"
        calib_dir = os.path.join(self.session_dir, calib_trial)
        os.makedirs(calib_dir, exist_ok=True)
        
        # 匹配标定视频到摄像头
        for camera_name in camera_names:
            # 寻找匹配的标定视频
            matching_file = None
            for calib_file in calib_files:
                if camera_name.lower() in os.path.basename(calib_file).lower():
                    matching_file = calib_file
                    break
            
            if matching_file is None and len(calib_files) == len(camera_names):
                # 如果数量匹配，按顺序分配
                matching_file = calib_files[list(camera_names).index(camera_name)]
            
            if matching_file:
                camera_calib_dir = os.path.join(self.session_dir, 'Videos', camera_name)
                dest_path = os.path.join(camera_calib_dir, f"{calib_trial}.mp4")
                shutil.copy2(matching_file, dest_path)
                logger.info(f"标定视频已复制: {matching_file} -> {dest_path}")
    
    def run_calibration(self, trial_names=None):
        """
        运行相机标定
        
        Args:
            trial_names: 用于标定的试验名称列表，None表示自动检测
        """
        logger.info("开始相机标定...")
        
        if trial_names is None:
            # 查找标定视频
            video_dirs = glob.glob(os.path.join(self.session_dir, 'Videos', '*'))
            trial_names = []
            
            for video_dir in video_dirs:
                calib_files = glob.glob(os.path.join(video_dir, "*calibration*.mp4"))
                if calib_files:
                    trial_names.extend([os.path.splitext(os.path.basename(f))[0] for f in calib_files])
            
            if not trial_names:
                # 使用所有可用视频进行标定
                trial_names = [os.path.splitext(os.path.basename(f))[0] 
                             for f in glob.glob(os.path.join(self.session_dir, 'Videos', '*', '*.mp4'))]
        
        if not trial_names:
            raise FileNotFoundError("没有找到可用于标定的视频文件")
        
        # 使用本地标定函数
        CheckerBoardParams = {
            'dimensions': tuple(self.config['calibration']['checkerboard']['dimensions']),
            'squareSize': self.config['calibration']['checkerboard']['square_size']
        }
        
        try:
            CamParamsAverage, CamParamList, intrinsicComparisons, cameraModel = computeAverageIntrinsicsLocal(
                self.session_dir,
                trial_names,
                CheckerBoardParams,
                nImages=self.config['calibration']['n_images']
            )
            
            # 保存标定结果
            for deployedFolderName in self.config['calibration']['deployed_folders']:
                permIntrinsicDir = os.path.join(
                    self.base_dir, 'CameraIntrinsics',
                    cameraModel, deployedFolderName
                )
                intrinsicFile = os.path.join(permIntrinsicDir, 'cameraIntrinsics.pickle')
                saveCameraParametersLocal(intrinsicFile, CamParamsAverage)
            
            logger.info(f"相机标定完成 - 型号: {cameraModel}")
            logger.info(f"平均重投影误差: {CamParamsAverage.get('reprojectionError', 0):.2f} 像素")
            
            return True, cameraModel
            
        except Exception as e:
            logger.error(f"相机标定失败: {str(e)}")
            return False, None
    
    def process_trial(self, trial_name, camera_names=None, trial_type='dynamic'):
        """
        处理单个试验
        
        Args:
            trial_name: 试验名称
            camera_names: 摄像头名称列表，None表示使用全部
            trial_type: 试验类型 ('dynamic', 'static', 'calibration')
        """
        logger.info(f"开始处理试验: {trial_name} ({trial_type})")
        
        try:
            # 运行OpenCap主流程
            success = opencap_main(
                sessionName=self.config['session']['name'],
                trialName=trial_name,
                trial_id=trial_name,  # 本地模式下trial_id=trialName
                cameras_to_use=camera_names or ['all'],
                intrinsicsFinalFolder='Deployed',
                isDocker=False,
                poseDetector=self.config['processing']['pose_detector'],
                resolutionPoseDetection=self.config['processing']['resolution'],
                bbox_thr=self.config['processing']['bbox_threshold'],
                augmenter_model=self.config['processing']['augmenter_model'],
                imageUpsampleFactor=self.config['processing']['image_upsample_factor'],
                filter_frequency=self.config['processing']['filter_frequency'],
                scaling_setup=self.config['processing']['scaling_setup'],
                scaleModel=self.config['output']['generate_opensim'],
                generateVideo=self.config['output']['save_videos']
            )
            
            if success is not False:  # main()通常不返回False，异常时会抛出
                logger.info(f"试验 {trial_name} 处理完成")
                return True
            else:
                logger.error(f"试验 {trial_name} 处理失败")
                return False
                
        except Exception as e:
            logger.error(f"试验 {trial_name} 处理异常: {str(e)}")
            return False
    
    def process_session(self, video_directory, calibration_directory=None):
        """
        处理完整会话
        
        Args:
            video_directory: 运动视频目录
            calibration_directory: 标定视频目录（可选）
        """
        logger.info("="*60)
        logger.info("开始本地OpenCap会话处理")
        logger.info("="*60)
        
        try:
            # 1. 创建会话元数据
            self.create_session_metadata()
            
            # 2. 设置视频数据
            trial_name, camera_names = self.setup_from_videos(video_directory, calibration_directory)
            
            # 3. 运行相机标定（如果有标定视频）
            if calibration_directory:
                calib_success, camera_model = self.run_calibration()
                if not calib_success:
                    logger.warning("相机标定失败，尝试使用现有标定参数")
            
            # 4. 处理运动试验
            success = self.process_trial(trial_name, camera_names, 'dynamic')
            
            if success:
                logger.info("="*60)
                logger.info("✅ 会话处理成功完成!")
                logger.info(f"结果保存在: {self.session_dir}")
                logger.info("="*60)
                
                # 生成处理报告
                self._generate_report(trial_name, camera_names)
            else:
                logger.error("❌ 会话处理失败")
            
            return success
            
        except Exception as e:
            logger.error(f"会话处理异常: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_report(self, trial_name, camera_names):
        """生成处理报告"""
        report = {
            'session_name': self.config['session']['name'],
            'processing_date': datetime.now().isoformat(),
            'trial_name': trial_name,
            'camera_names': camera_names,
            'configuration': self.config,
            'output_files': self._list_output_files()
        }
        
        report_path = os.path.join(self.session_dir, 'processing_report.yaml')
        with open(report_path, 'w', encoding='utf-8') as f:
            yaml.dump(report, f, allow_unicode=True, default_flow_style=False)
        
        logger.info(f"处理报告已保存: {report_path}")
    
    def _list_output_files(self):
        """列出输出文件"""
        output_files = {}
        
        # 扫描各种输出文件
        file_patterns = {
            'trc_files': '**/*.trc',
            'opensim_files': '**/OpenSimData/*',
            'videos': '**/OutputMedia*/*',
            'pose_data': '**/*keypoints*.pkl',
            'calibration': '**/cameraIntrinsicsExtrinsics.pickle'
        }
        
        for file_type, pattern in file_patterns.items():
            files = glob.glob(os.path.join(self.session_dir, pattern), recursive=True)
            output_files[file_type] = [os.path.relpath(f, self.session_dir) for f in files]
        
        return output_files

# 便捷函数
def create_config_template(output_path):
    """创建配置文件模板"""
    pipeline = LocalOpenCapPipeline()
    config = pipeline._get_default_config()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    print(f"配置文件模板已创建: {output_path}")

def run_local_opencap(video_dir, calibration_dir=None, config_file=None, **kwargs):
    """
    便捷函数：运行本地OpenCap处理
    
    Args:
        video_dir: 运动视频目录
        calibration_dir: 标定视频目录（可选）
        config_file: 配置文件路径（可选）
        **kwargs: 其他配置参数
    """
    
    # 合并配置
    config = {}
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    
    # 应用kwargs覆盖
    for key, value in kwargs.items():
        if '.' in key:
            # 支持嵌套键如 'processing.pose_detector'
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        else:
            config[key] = value
    
    # 创建并运行管道
    pipeline = LocalOpenCapPipeline(config_dict=config)
    return pipeline.process_session(video_dir, calibration_dir)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='本地OpenCap处理管道')
    parser.add_argument('video_dir', help='运动视频目录')
    parser.add_argument('--calibration-dir', '-c', help='标定视频目录')
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--pose-detector', choices=['OpenPose', 'mmpose'], 
                      default='OpenPose', help='姿态检测器')
    parser.add_argument('--resolution', default='1x736', 
                      help='OpenPose分辨率')
    parser.add_argument('--create-config', help='创建配置文件模板到指定路径')
    
    args = parser.parse_args()
    
    # 创建配置模板
    if args.create_config:
        create_config_template(args.create_config)
        sys.exit(0)
    
    # 运行处理
    success = run_local_opencap(
        video_dir=args.video_dir,
        calibration_dir=args.calibration_dir,
        config_file=args.config,
        pose_detector=args.pose_detector,
        resolution=args.resolution
    )
    
    sys.exit(0 if success else 1)