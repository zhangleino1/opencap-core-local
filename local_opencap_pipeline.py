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

# 设置本地模式环境变量，跳过API认证
os.environ['OPENCAP_LOCAL_MODE'] = 'true'
# 设置UTF-8编码，避免Windows下的GBK编码问题
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 强制设置Python默认编码为UTF-8
import locale
import sys
if sys.platform.startswith('win'):
    try:
        # 在Windows上强制设置UTF-8编码
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
        sys.stdin.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python 3.6及以下版本的兼容性
        pass

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入OpenCap核心模块
# 延迟导入utils以避免API token要求
def getDataDirectory():
    """获取数据目录，避免API token问题"""
    import os
    try:
        from utils import getDataDirectory as _getDataDirectory
        return _getDataDirectory()
    except:
        # 如果utils导入失败，使用默认本地路径
        return os.path.dirname(os.path.abspath(__file__))

def importMetadata(sessionDir):
    """导入元数据，避免API token问题"""
    try:
        from utils import importMetadata as _importMetadata
        return _importMetadata(sessionDir)
    except:
        # 如果utils导入失败，返回None
        return None

def opencap_main(*args, **kwargs):
    """延迟导入main函数以避免API token问题"""
    try:
        from main import main as _main
        return _main(*args, **kwargs)
    except Exception as e:
        logger.error(f"Failed to import or run main: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

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
            return self._load_template_config()
    
    def _load_template_config(self):
        """从模板配置文件加载默认配置"""
        template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'local_config_template.yaml')
        
        if os.path.exists(template_path):
            with open(template_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                # 确保会话名称是最新的
                if 'session' in config:
                    config['session']['name'] = f'LocalSession_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
                return config
        else:
            # 如果模板文件不存在，返回最基本的配置
            return {
                'session': {
                    'name': f'LocalSession_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                    'description': '本地OpenCap会话',
                    'subject_mass': 67.0,
                    'subject_height': 170.0
                },
                'calibration': {
                    'checkerboard': {
                        'dimensions': [5, 4],
                        'square_size': 35
                    },
                    'n_images': 50,
                    'deployed_folders': ['Deployed_720_60fps', 'Deployed']
                },
                'processing': {
                    'pose_detector': 'OpenPose',
                    'resolution': '1x736',
                    'bbox_threshold': 0.8,
                    'image_upsample_factor': 4,
                    'augmenter_model': 'v0.3',
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
                'squareSideLength_mm': self.config['calibration']['checkerboard']['square_size'],
                'placement': 'backWall'  # 添加棋盘格放置位置参数
            },
            
            # 处理参数
            'poseDetector': self.config['processing']['pose_detector'],
            'resolutionPoseDetection': self.config['processing']['resolution'],
            'augmenter_model': self.config['processing']['augmenter_model'],
            'imageUpsampleFactor': self.config['processing']['image_upsample_factor'],
            
            # 多摄像头型号支持 - 每个摄像头有独立的型号标识 (支持各种品牌摄像头)
            'cameraModel': {},
            
            # OpenSim模型配置
            'openSimModel': 'LaiUhlrich2022',
            
            # 标记点增强配置 - 必需的配置项
            'markerAugmentationSettings': {
                'markerAugmenterModel': 'LSTM'  # 模型类型固定为LSTM，版本由augmenter_model指定
            },
            
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
        
        # 为每个摄像头创建目录并复制视频 - 使用官方OpenCap目录结构
        for camera_name, video_file in cameras.items():
            camera_dir = os.path.join(self.session_dir, 'Videos', camera_name)
            # 创建符合官方OpenCap期望的目录结构: Videos/CameraName/InputMedia/trialName/
            input_media_dir = os.path.join(camera_dir, 'InputMedia', trial_name)
            os.makedirs(input_media_dir, exist_ok=True)
            
            # 视频文件放在 InputMedia/trialName/trial_id.mp4
            dest_path = os.path.join(input_media_dir, f"{trial_name}.mp4")
            shutil.copy2(video_file, dest_path)
            logger.info(f"视频已复制: {video_file} -> {dest_path}")
        
        # 处理标定视频（如果提供）- 创建标定试验
        calib_trial_name = None
        if calibration_directory and os.path.exists(calibration_directory):
            calib_trial_name = self._setup_calibration_trial(calibration_directory, cameras.keys())
        
        return trial_name, list(cameras.keys()), calib_trial_name
    
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
    
    def _setup_calibration_trial(self, calibration_directory, camera_names):
        """设置标定试验 - 创建符合官方OpenCap结构的标定试验"""
        calib_files = glob.glob(os.path.join(calibration_directory, "*.mp4"))
        
        if not calib_files:
            logger.warning(f"标定目录中没有找到.mp4文件: {calibration_directory}")
            return None
        
        # 创建标定试验名称
        calib_trial_name = "calibration"
        
        # 匹配标定视频到摄像头
        for camera_name in camera_names:
            # 寻找匹配的标定视频 - 支持多种命名方式
            matching_file = None
            camera_base_name = camera_name.lower()
            
            # 尝试多种匹配模式
            for calib_file in calib_files:
                calib_filename = os.path.basename(calib_file).lower()
                
                # 模式1: 直接包含摄像头名称
                if camera_base_name in calib_filename:
                    matching_file = calib_file
                    break
                
                # 模式2: 提取摄像头编号进行匹配 (如 camera1_walk -> camera1)
                import re
                camera_number_match = re.search(r'camera(\d+)', camera_base_name)
                if camera_number_match:
                    camera_num = camera_number_match.group(1)
                    if f'camera{camera_num}' in calib_filename:
                        matching_file = calib_file
                        break
                
                # 模式3: 提取纯数字编号进行匹配
                number_match = re.search(r'(\d+)', camera_base_name)
                if number_match:
                    cam_num = number_match.group(1)
                    if cam_num in calib_filename:
                        matching_file = calib_file
                        break
            
            # 如果仍然没有匹配，且文件数量相同，按顺序分配
            if matching_file is None and len(calib_files) == len(camera_names):
                sorted_calib_files = sorted(calib_files)
                sorted_camera_names = sorted(camera_names)
                matching_file = sorted_calib_files[sorted_camera_names.index(camera_name)]
            
            if matching_file:
                # 创建标定试验的官方目录结构: Videos/CameraName/InputMedia/calibration/
                camera_dir = os.path.join(self.session_dir, 'Videos', camera_name)
                calib_input_media_dir = os.path.join(camera_dir, 'InputMedia', calib_trial_name)
                os.makedirs(calib_input_media_dir, exist_ok=True)
                
                # 标定视频放在 InputMedia/calibration/calibration.mp4
                dest_path = os.path.join(calib_input_media_dir, f"{calib_trial_name}.mp4")
                shutil.copy2(matching_file, dest_path)
                logger.info(f"标定视频已复制: {matching_file} -> {dest_path}")
        
        return calib_trial_name
    
    def run_calibration(self, trial_names=None):
        """
        运行相机标定
        
        Args:
            trial_names: 用于标定的试验名称列表，None表示自动检测
        """
        logger.info("开始相机标定...")
        
        if trial_names is None:
            # 查找标定视频 - 现在在InputMedia/calibration/目录下
            trial_names = ["calibration"]  # 直接使用标定试验名称
            
            # 验证标定文件存在
            calib_exists = False
            video_dirs = glob.glob(os.path.join(self.session_dir, 'Videos', '*'))
            for video_dir in video_dirs:
                # 检查新的目录结构: Videos/CameraName/InputMedia/calibration/calibration.mp4
                calib_files = glob.glob(os.path.join(video_dir, "InputMedia", "calibration", "calibration.mp4"))
                if calib_files:
                    calib_exists = True
                    break
            
            if not calib_exists:
                logger.error("未找到标定视频文件！标定需要包含棋盘格的视频。")
                return False, None
        
        if not trial_names:
            raise FileNotFoundError("没有找到可用于标定的视频文件")
        
        # 使用本地标定函数
        CheckerBoardParams = {
            'dimensions': tuple(self.config['calibration']['checkerboard']['dimensions']),
            'squareSize': self.config['calibration']['checkerboard']['square_size']
        }
        
        try:
            # 新的多摄像头分别标定逻辑
            videos_dir = os.path.join(self.session_dir, 'Videos')
            camera_models = {}
            session_metadata = {}
            
            # 获取所有摄像头目录
            camera_dirs = glob.glob(os.path.join(videos_dir, "*"))
            camera_dirs = [d for d in camera_dirs if os.path.isdir(d)]
            
            if not camera_dirs:
                raise FileNotFoundError("没有找到摄像头目录")
            
            logger.info(f"开始为 {len(camera_dirs)} 个摄像头分别标定...")
            
            for camera_dir in camera_dirs:
                camera_name = os.path.basename(camera_dir)
                logger.info(f"标定摄像头: {camera_name}")
                
                # 查找该摄像头的标定视频 - 使用新的目录结构
                camera_trial_videos = []
                for trial_name in trial_names:
                    # 新目录结构: Videos/CameraName/InputMedia/calibration/calibration.mp4
                    video_pattern = os.path.join(camera_dir, "InputMedia", trial_name, f"{trial_name}.mp4")
                    matching_files = glob.glob(video_pattern)
                    camera_trial_videos.extend(matching_files)
                
                if not camera_trial_videos:
                    logger.warning(f"摄像头 {camera_name} 没有找到标定视频")
                    continue
                
                # 为每个摄像头单独标定
                CamParams, intrinsicComparison, detectedModel = self._calibrate_single_camera(
                    camera_trial_videos, CheckerBoardParams
                )
                
                if CamParams is None:
                    logger.error(f"摄像头 {camera_name} 标定失败")
                    continue
                
                # 生成该摄像头的型号名称
                camera_model = f"{detectedModel}_{camera_name}" if detectedModel else f"Camera_{camera_name}"
                camera_models[camera_name] = camera_model
                
                # 保存该摄像头的内参
                for deployedFolderName in self.config['calibration']['deployed_folders']:
                    permIntrinsicDir = os.path.join(
                        self.base_dir, 'CameraIntrinsics',
                        camera_model, deployedFolderName
                    )
                    intrinsicFile = os.path.join(permIntrinsicDir, 'cameraIntrinsics.pickle')
                    saveCameraParametersLocal(intrinsicFile, CamParams)
                
                logger.info(f"摄像头 {camera_name} 标定完成 - 型号: {camera_model}")
                logger.info(f"摄像头 {camera_name} 重投影误差: {CamParams.get('reprojectionError', 0):.2f} 像素")
            
            # 更新sessionMetadata中的iphoneModel字段
            self._update_session_metadata_with_camera_models(camera_models)
            
            if not camera_models:
                raise Exception("所有摄像头标定都失败了")
            
            logger.info(f"多摄像头标定完成，共标定 {len(camera_models)} 个摄像头")
            
            return True, camera_models
            
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
                scaleModel=self.config['output']['generate_opensim']
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
            trial_name, camera_names, calib_trial_name = self.setup_from_videos(video_directory, calibration_directory)
            
            # 3. 运行相机标定（如果有标定视频）
            if calibration_directory:
                calib_success, camera_model = self.run_calibration()
                if not calib_success:
                    logger.warning("相机标定失败，尝试使用现有标定参数")
                
                # 处理标定试验以计算外参
                if calib_trial_name:
                    logger.info("处理标定试验以计算外参...")
                    calib_success = self.process_trial(calib_trial_name, camera_names, 'calibration')
                    if not calib_success:
                        logger.warning("标定试验处理失败，但继续处理运动试验")
            
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
    
    def _calibrate_single_camera(self, video_files, CheckerBoardParams):
        """
        为单个摄像头标定内参
        
        Args:
            video_files: 该摄像头的标定视频文件列表
            CheckerBoardParams: 标定板参数
            
        Returns:
            CamParams: 摄像头参数
            intrinsicComparison: 内参比较数据  
            detectedModel: 检测到的摄像头型号
        """
        from main_calcIntrinsics_local import calibrateCameraFromVideo, extractCameraModelFromFilename
        
        CamParamList = []
        detected_models = []
        
        for video_file in video_files:
            logger.info(f"  处理视频: {os.path.basename(video_file)}")
            
            # 检测摄像头型号
            detected_model = extractCameraModelFromFilename(video_file)
            if detected_model:
                detected_models.append(detected_model)
            
            # 标定该视频
            try:
                CamParams = calibrateCameraFromVideo(
                    video_file, CheckerBoardParams, 
                    nImages=self.config['calibration']['n_images']
                )
                
                if CamParams is not None:
                    CamParamList.append(CamParams)
                    reprojection_error = CamParams.get('reprojectionError', 0)
                    logger.info(f"    视频信息: {CamParams.get('resolution', 'Unknown')}, "
                              f"{CamParams.get('frameRate', 'Unknown')}fps, "
                              f"{CamParams.get('nFrames', 'Unknown')}帧")
                    logger.info(f"    CheckerBoardParams使用 {CheckerBoardParams} 张图像进行标定")
                    logger.info(f"    找到 {CamParams.get('nImages', 0)} 幅有效标定图像")
                    logger.info(f"    重投影误差: {reprojection_error:.2f} 像素")
                
            except Exception as e:
                logger.warning(f"    视频 {os.path.basename(video_file)} 标定失败: {str(e)}")
                continue
        
        if not CamParamList:
            logger.error("    没有成功标定的视频")
            return None, None, None
        
        # 如果有多个视频，计算平均内参
        if len(CamParamList) > 1:
            # 简单平均所有内参矩阵和畸变参数
            avg_params = CamParamList[0].copy()
            
            # 平均内参矩阵K
            K_sum = sum([params['K'] for params in CamParamList])
            avg_params['K'] = K_sum / len(CamParamList)
            
            # 平均畸变参数
            if 'distCoeff' in avg_params:
                distCoeff_sum = sum([params['distCoeff'] for params in CamParamList])
                avg_params['distCoeff'] = distCoeff_sum / len(CamParamList)
            
            # 平均重投影误差
            avg_error = sum([params.get('reprojectionError', 0) for params in CamParamList]) / len(CamParamList)
            avg_params['reprojectionError'] = avg_error
            
            logger.info(f"    平均重投影误差: {avg_error:.2f} 像素")
        else:
            avg_params = CamParamList[0]
        
        # 确定摄像头型号 - 支持各种品牌摄像头
        if detected_models:
            # 使用最常见的型号
            detectedModel = max(set(detected_models), key=detected_models.count)
        else:
            # 如果无法从文件名检测，使用通用命名
            detectedModel = "GenericCamera"
        
        return avg_params, {}, detectedModel
    
    def _update_session_metadata_with_camera_models(self, camera_models):
        """
        更新sessionMetadata中的cameraModel字段 - 支持各种品牌摄像头
        
        Args:
            camera_models: {camera_name: camera_model} 字典
        """
        metadata_path = os.path.join(self.session_dir, 'sessionMetadata.yaml')
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = yaml.safe_load(f)
        else:
            metadata = {}
        
        # 更新cameraModel字段 (支持各种品牌摄像头，不仅仅是iPhone)
        metadata['cameraModel'] = camera_models
        
        # 保存更新后的元数据
        with open(metadata_path, 'w', encoding='utf-8') as f:
            yaml.dump(metadata, f, allow_unicode=True, default_flow_style=False)
        
        logger.info(f"已更新sessionMetadata，摄像头型号: {camera_models}")
    
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
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'local_config_template.yaml')
    
    if os.path.exists(template_path):
        # 检查是否是同一个文件
        if os.path.abspath(template_path) != os.path.abspath(output_path):
            # 复制现有模板文件
            shutil.copy2(template_path, output_path)
        
        # 更新会话名称为当前时间
        with open(output_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if 'session' in config:
            config['session']['name'] = f'LocalSession_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    else:
        # 如果模板不存在，创建一个最基本的模板
        pipeline = LocalOpenCapPipeline()
        config = pipeline._load_template_config()
        
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