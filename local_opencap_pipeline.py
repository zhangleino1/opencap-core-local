"""
本地OpenCap处理管道
@authors: 基于OpenCap Core修改

本脚本实现了完全离线的运动捕获处理流程，包括：
1. 本地相机标定 (支持单摄像头/多摄像头)
2. 姿态检测 (OpenPose/MMPose)
3. 视频同步
4. 3D三角化重建
5. 标记点增强 (LSTM)
6. OpenSim生物力学分析

主要特性：
- 符合官方OpenCap API规范
- 完整的试验间数据继承机制
- 智能错误处理和数据清理
- 无需网络连接，完全本地运行
"""

import os
import sys
import glob
import shutil
import yaml
import json
import logging
import argparse
import traceback
from pathlib import Path
from datetime import datetime
import numpy as np

# 设置本地模式环境变量，跳过API认证
os.environ['OPENCAP_LOCAL_MODE'] = 'true'
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 强制设置Python默认编码为UTF-8
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')  
        sys.stdin.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入OpenCap核心模块  
def getDataDirectory():
    """获取数据目录，避免API token问题"""
    try:
        from utils import getDataDirectory as _getDataDirectory
        return _getDataDirectory()
    except:
        return os.path.dirname(os.path.abspath(__file__))

def importMetadata(sessionDir):
    """导入元数据，避免API token问题"""
    try:
        from utils import importMetadata as _importMetadata
        return _importMetadata(sessionDir)
    except:
        return None

def opencap_main(*args, **kwargs):
    """延迟导入main函数以避免API token问题"""
    try:
        from main import main as _main
        return _main(*args, **kwargs)
    except Exception as e:
        logger.error(f"Failed to import or run main: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

# 导入本地标定模块
from main_calcIntrinsics_local import (
    computeAverageIntrinsicsLocal, 
    saveCameraParametersLocal,
    extractCameraModelFromFilename
)

class LocalOpenCapPipeline:
    """本地OpenCap处理管道 - 符合官方逻辑"""
    
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
        self._load_default_settings()
        
        # 存储试验间共享的数据
        self.calibration_options = None
        self.session_metadata = None
        self.static_trial_name = None
        
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
                # 更新会话名称为当前时间
                config['session']['name'] = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                return config
        else:
            raise FileNotFoundError(f"配置模板文件不存在: {template_path}")
            
    def _validate_config(self):
        """验证配置参数"""
        required_keys = ['session', 'calibration', 'processing', 'directories']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"配置文件缺少必需的键: {key}")
        
        # 验证姿态检测器
        valid_detectors = ['OpenPose', 'mmpose']
        detector = self.config['processing']['pose_detector']
        if detector not in valid_detectors:
            raise ValueError(f"不支持的姿态检测器: {detector}")
    
    def _load_default_settings(self):
        """加载默认OpenCap设置"""
        settings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'defaultOpenCapSettings.json')
        
        try:
            with open(settings_path, 'r', encoding='utf-8') as f:
                self.default_settings = json.load(f)
                logger.info(f"已加载默认设置: {self.default_settings}")
        except Exception as e:
            logger.warning(f"无法加载默认设置文件 {settings_path}: {str(e)}")
            self.default_settings = {
                'openpose': '1x736',
                'hrnet': 0.8
            }
    
    def _setup_directories(self):
        """设置目录结构"""
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 数据目录
        data_dir = getDataDirectory()
        self.session_name = self.config['session']['name']
        self.session_dir = os.path.join(data_dir, 'Data', self.session_name)
        
        # 创建目录结构
        dirs_to_create = [
            self.session_dir,
            os.path.join(self.session_dir, 'Videos'),
            os.path.join(self.session_dir, 'CalibrationImages'),
            os.path.join(self.session_dir, 'MarkerData'),
            os.path.join(self.session_dir, 'OpenSimData'),
            os.path.join(self.session_dir, 'VisualizerVideos'),
            os.path.join(self.session_dir, 'VisualizerJsons'),
            os.path.join(self.session_dir, 'NeutralPoseImages')
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
        
        logger.info(f"会话目录已创建: {self.session_dir}")
    
    def _cleanup_previous_outputs(self, trial_type, trial_name=None):
        """
        清理之前的输出文件 - 只清理处理结果，保留输入视频
        
        Args:
            trial_type: 试验类型 ('calibration', 'static', 'dynamic')
            trial_name: 试验名称
        """
        logger.info(f"清理之前的输出文件: {trial_type}")
        
        try:
            if trial_type == 'calibration':
                self._delete_calibration_outputs()
            elif trial_type == 'static':
                self._delete_static_outputs(trial_name or 'neutral')
        except Exception as e:
            logger.warning(f"清理输出文件时出错: {str(e)}")
    
    def _delete_calibration_outputs(self):
        """删除标定输出文件 - 只删除处理结果，保留输入视频"""
        # 删除标定图像
        cal_image_path = os.path.join(self.session_dir, 'CalibrationImages')
        if os.path.exists(cal_image_path):
            shutil.rmtree(cal_image_path)
            os.makedirs(cal_image_path, exist_ok=True)
        
        # 删除相机目录中的标定结果文件
        cam_dirs = glob.glob(os.path.join(self.session_dir, 'Videos', 'Cam*'))
        
        for cam_dir in cam_dirs:
            # 删除相机内外参文件
            ext_path = os.path.join(cam_dir, 'cameraIntrinsicsExtrinsics.pickle')
            if os.path.exists(ext_path):
                os.remove(ext_path)
                logger.info(f"已删除标定结果文件: {ext_path}")
            
            # 删除OutputPkl目录中的处理结果
            output_pkl_path = os.path.join(cam_dir, 'OutputPkl')
            if os.path.exists(output_pkl_path):
                shutil.rmtree(output_pkl_path)
                logger.info(f"已删除处理结果目录: {output_pkl_path}")
    
    def _delete_static_outputs(self, static_trial_name='neutral'):
        """删除静态试验输出文件 - 只删除处理结果，保留输入视频"""
        # 删除标记数据
        marker_dirs = glob.glob(os.path.join(self.session_dir, 'MarkerData*'))
        for marker_dir in marker_dirs:
            if os.path.exists(marker_dir):
                marker_files = glob.glob(os.path.join(marker_dir, '*'))
                for marker_file in marker_files:
                    if static_trial_name in os.path.basename(marker_file):
                        os.remove(marker_file)
                        logger.info(f"已删除标记文件: {marker_file}")
        
        # 删除OpenSim数据（静态是第一个保存的OpenSim数据）
        opensim_dir = os.path.join(self.session_dir, 'OpenSimData')
        if os.path.exists(opensim_dir):
            shutil.rmtree(opensim_dir)
            os.makedirs(opensim_dir, exist_ok=True)
            logger.info("已删除OpenSim数据目录")
        
        # 删除相机目录中的输出文件
        cam_dirs = glob.glob(os.path.join(self.session_dir, 'Videos', 'Cam*'))
        for cam_dir in cam_dirs:
            # 删除OutputPkl目录中与静态试验相关的文件
            output_pkl_path = os.path.join(cam_dir, 'OutputPkl', static_trial_name)
            if os.path.exists(output_pkl_path):
                shutil.rmtree(output_pkl_path)
                logger.info(f"已删除静态试验输出: {output_pkl_path}")
    
    def _cleanup_previous_results(self, trial_type, trial_name=None):
        """
        清理之前的结果文件 - 基于官方deleteCalibrationFiles和deleteStaticFiles
        
        Args:
            trial_type: 试验类型 ('calibration', 'static', 'dynamic')
            trial_name: 试验名称
        """
        logger.info(f"清理之前的结果文件: {trial_type}")
        
        try:
            if trial_type == 'calibration':
                self._delete_calibration_files()
            elif trial_type == 'static':
                self._delete_static_files(trial_name or 'neutral')
        except Exception as e:
            logger.warning(f"清理文件时出错: {str(e)}")
    
    def _delete_calibration_files(self):
        """删除标定文件 - 基于官方deleteCalibrationFiles"""
        # 删除标定图像
        cal_image_path = os.path.join(self.session_dir, 'CalibrationImages')
        if os.path.exists(cal_image_path):
            shutil.rmtree(cal_image_path)
            os.makedirs(cal_image_path, exist_ok=True)
        
        # 删除相机目录中的标定文件
        cam_dirs = glob.glob(os.path.join(self.session_dir, 'Videos', 'Cam*'))
        
        for cam_dir in cam_dirs:
            # 删除相机内外参文件
            ext_path = os.path.join(cam_dir, 'cameraIntrinsicsExtrinsics.pickle')
            if os.path.exists(ext_path):
                os.remove(ext_path)
                logger.info(f"已删除标定文件: {ext_path}")
    
    def _delete_static_files(self, static_trial_name='neutral'):
        """删除静态文件 - 基于官方deleteStaticFiles"""
        # 删除相机目录中的静态试验
        cam_dirs = glob.glob(os.path.join(self.session_dir, 'Videos', 'Cam*'))
        
        for cam_dir in cam_dirs:
            media_dirs = glob.glob(os.path.join(cam_dir, '*'))
            for med_dir in media_dirs:
                static_path = os.path.join(med_dir, static_trial_name)
                if os.path.exists(static_path):
                    shutil.rmtree(static_path)
                    logger.info(f"已删除静态试验目录: {static_path}")
        
        # 删除标记数据
        marker_dirs = glob.glob(os.path.join(self.session_dir, 'MarkerData*'))
        for marker_dir in marker_dirs:
            if os.path.exists(marker_dir):
                marker_files = glob.glob(os.path.join(marker_dir, '*'))
                for marker_file in marker_files:
                    if static_trial_name in os.path.basename(marker_file):
                        os.remove(marker_file)
                        logger.info(f"已删除标记文件: {marker_file}")
        
        # 删除OpenSim数据（静态是第一个保存的OpenSim数据）
        opensim_dir = os.path.join(self.session_dir, 'OpenSimData')
        if os.path.exists(opensim_dir):
            shutil.rmtree(opensim_dir)
            os.makedirs(opensim_dir, exist_ok=True)
            logger.info("已删除OpenSim数据目录")
    
    def _get_calibration_data(self, trial_type='dynamic'):
        """
        获取标定数据 - 本地版本的getCalibration
        
        Args:
            trial_type: 试验类型，用于确定是否需要标定选项
            
        Returns:
            calibration_options: 标定选项（如果需要）
        """
        logger.info(f"获取标定数据用于 {trial_type} 试验")
        
        # 查找标定文件
        cam_dirs = glob.glob(os.path.join(self.session_dir, 'Videos', 'Cam*'))
        calibration_files = []
        
        for cam_dir in cam_dirs:
            calib_file = os.path.join(cam_dir, 'cameraIntrinsicsExtrinsics.pickle')
            if os.path.exists(calib_file):
                calibration_files.append(calib_file)
        
        if not calibration_files:
            logger.warning("未找到标定文件，可能需要先运行标定试验")
            return None
        
        logger.info(f"找到 {len(calibration_files)} 个标定文件")
        
        # 对于静态试验，不返回标定选项（本地处理只有单一解决方案）
        if trial_type == 'static':
            # 本地处理中，我们已经有了标定文件，不需要选择多个方案
            # 返回None以跳过自动选择外参的过程
            return None
        
        return None
    
    def _get_model_and_metadata(self):
        """
        获取模型和元数据 - 本地版本的getModelAndMetadata
        
        查找静态试验生成的缩放模型
        """
        logger.info("获取模型和元数据")
        
        # 查找缩放后的模型
        model_dir = os.path.join(self.session_dir, 'OpenSimData', 'Model')
        scaled_models = glob.glob(os.path.join(model_dir, '*_scaled.osim'))
        
        if scaled_models:
            logger.info(f"找到缩放模型: {scaled_models}")
            return True
        else:
            logger.warning("未找到缩放模型，可能需要先运行静态试验")
            return False
    
    def _apply_pose_detector_settings(self, pose_detector):
        """
        根据姿态检测器应用默认设置 - 基于官方逻辑
        
        Args:
            pose_detector: 姿态检测器名称
            
        Returns:
            dict: 更新后的处理参数
        """
        params = {}
        
        if pose_detector.lower() == 'openpose':
            params['resolutionPoseDetection'] = self.config['processing']['resolution']
                
        elif pose_detector.lower() == 'mmpose':
            params['bbox_thr'] = self.config['processing']['bbox_threshold']
        
        logger.info(f"应用 {pose_detector} 设置: {params}")
        return params

    def _interactive_calibration_selection(self):
        """
        交互式选择标定方案

        Returns:
            list: 需要使用备选方案的摄像头列表
        """
        import subprocess
        import platform

        print("\n" + "="*60)
        print("🎯 标定方案选择")
        print("="*60)
        print("⚠️  重要提示：此选择将保存并用于后续所有试验（静态、动态）")
        print("    一旦选择完成，后续试验将自动使用保存的方案，无需再次选择")
        print("="*60)

        # 查找标定图像
        cal_image_dir = os.path.join(self.session_dir, 'CalibrationImages')
        if not os.path.exists(cal_image_dir):
            logger.warning("未找到标定图像目录，跳过交互式选择")
            return None

        cam_dirs = glob.glob(os.path.join(self.session_dir, 'Videos', 'Cam*'))
        alternate_cams = []

        for cam_dir in cam_dirs:
            cam_name = os.path.basename(cam_dir)

            # 查找标定方案图像
            solution1_img = os.path.join(cal_image_dir, f'extrinsicCalib_{cam_name}.jpg')
            solution2_img = os.path.join(cal_image_dir, f'extrinsicCalib_altSoln_{cam_name}.jpg')

            if os.path.exists(solution1_img) and os.path.exists(solution2_img):
                print(f"\n📷 {cam_name} 标定方案选择:")
                print(f"   方案0 (默认): {solution1_img}")
                print(f"   方案1 (备选): {solution2_img}")

                # 尝试自动打开图像供用户查看
                try:
                    if platform.system() == 'Windows':
                        subprocess.run(['start', solution1_img], shell=True, check=False)
                        subprocess.run(['start', solution2_img], shell=True, check=False)
                    elif platform.system() == 'Darwin':  # macOS
                        subprocess.run(['open', solution1_img], check=False)
                        subprocess.run(['open', solution2_img], check=False)
                    elif platform.system() == 'Linux':
                        subprocess.run(['xdg-open', solution1_img], check=False)
                except:
                    pass

                print("\n请查看两个标定方案图像:")
                print("- 方案0: 默认标定方案 (对应上面的第一个图像)")
                print("- 方案1: 备选标定方案 (对应上面的第二个图像)")
                print("\n正确的标定方案应该：")
                print("✅ Z轴(深蓝色箭头)垂直指向标定板平面 (dark blue axis pointing into the board)")
                print("✅ X轴(红色箭头)和Y轴(绿色箭头)平行于标定板平面")
                print("✅ 坐标轴清晰可见，没有明显的几何扭曲")
                print("❌ 如果Z轴指向相反方向，应选择备选方案")

                while True:
                    choice = input(f"\n请选择 {cam_name} 的标定方案 (0: 默认/1: 备选): ").strip()
                    if choice == '0':
                        print(f"✅ {cam_name} 使用方案0 (默认)")
                        break
                    elif choice == '1':
                        print(f"✅ {cam_name} 使用方案1 (备选)")
                        alternate_cams.append(cam_name)
                        break
                    else:
                        print("❌ 请输入 0 或 1")
            else:
                logger.warning(f"未找到 {cam_name} 的标定方案图像，使用默认方案")

        print("\n" + "="*60)
        if alternate_cams:
            print(f"📋 最终选择: {alternate_cams} 使用备选标定方案")
        else:
            print("📋 所有摄像头使用默认标定方案")
        print("="*60 + "\n")

        return alternate_cams if alternate_cams else None

    def _apply_calibration_selection(self, alternate_cams):
        """
        应用用户的标定方案选择

        Args:
            alternate_cams: 需要使用备选方案的摄像头列表
        """
        import shutil

        logger.info(f"🔄 应用标定方案选择: {alternate_cams}")

        cam_dirs = glob.glob(os.path.join(self.session_dir, 'Videos', 'Cam*'))

        # 创建方案选择记录
        calibration_selection = {}

        for cam_dir in cam_dirs:
            cam_name = os.path.basename(cam_dir)

            # 确定使用哪个方案
            if cam_name in alternate_cams:
                # 使用方案1 (备选方案)
                source_file = os.path.join(cam_dir, 'InputMedia', 'calibration', 'cameraIntrinsicsExtrinsics_soln1.pickle')
                solution_num = 1
                logger.info(f"📷 {cam_name}: 选择方案1 (备选)")
            else:
                # 使用方案0 (默认方案)
                source_file = os.path.join(cam_dir, 'InputMedia', 'calibration', 'cameraIntrinsicsExtrinsics_soln0.pickle')
                solution_num = 0
                logger.info(f"📷 {cam_name}: 选择方案0 (默认)")

            # 记录选择
            calibration_selection[cam_name] = solution_num

            # 目标文件
            target_file = os.path.join(cam_dir, 'cameraIntrinsicsExtrinsics.pickle')

            # 检查源文件是否存在
            if os.path.exists(source_file):
                # 复制选择的方案到最终文件
                shutil.copy2(source_file, target_file)
                logger.info(f"✅ {cam_name}: 已应用选择的方案")

                # 验证复制是否成功
                if os.path.exists(target_file):
                    source_size = os.path.getsize(source_file)
                    target_size = os.path.getsize(target_file)
                    if source_size == target_size:
                        logger.info(f"   文件大小验证通过: {target_size} bytes")
                    else:
                        logger.warning(f"   文件大小不匹配: 源文件{source_size}, 目标文件{target_size}")
                else:
                    logger.error(f"❌ {cam_name}: 复制失败，目标文件不存在")
            else:
                logger.error(f"❌ {cam_name}: 源文件不存在: {source_file}")

        # 保存选择记录到会话目录，供后续试验使用
        self._save_calibration_selection(calibration_selection)

    def _save_calibration_selection(self, calibration_selection):
        """
        保存标定方案选择记录
        
        Args:
            calibration_selection: 摄像头方案选择字典 {"Cam1": 0, "Cam2": 1}
        """
        selection_file = os.path.join(self.session_dir, 'calibration_selection.yaml')
        
        selection_data = {
            'selection_time': datetime.now().isoformat(),
            'camera_solutions': calibration_selection,
            'description': '用户选择的标定方案记录，用于确保后续试验使用一致的内外参'
        }
        
        with open(selection_file, 'w', encoding='utf-8') as f:
            yaml.dump(selection_data, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"📋 标定方案选择已保存: {selection_file}")
        logger.info(f"   选择记录: {calibration_selection}")

    def _load_calibration_selection(self):
        """
        读取之前保存的标定方案选择
        
        Returns:
            dict: 摄像头方案选择字典 {"Cam1": 0, "Cam2": 1}，如果没有则返回None
        """
        selection_file = os.path.join(self.session_dir, 'calibration_selection.yaml')
        
        if os.path.exists(selection_file):
            try:
                with open(selection_file, 'r', encoding='utf-8') as f:
                    selection_data = yaml.safe_load(f)
                
                camera_solutions = selection_data.get('camera_solutions', {})
                logger.info(f"📋 读取到之前的标定方案选择: {camera_solutions}")
                return camera_solutions
            except Exception as e:
                logger.warning(f"读取标定方案选择文件失败: {str(e)}")
                return None
        else:
            return None

    def _ensure_calibration_consistency(self):
        """
        确保使用一致的标定方案 - 在每次处理试验前调用
        """
        # 读取之前保存的选择
        saved_selection = self._load_calibration_selection()
        
        if saved_selection:
            logger.info("🔒 检测到之前的标定方案选择，确保一致性...")
            
            cam_dirs = glob.glob(os.path.join(self.session_dir, 'Videos', 'Cam*'))
            
            for cam_dir in cam_dirs:
                cam_name = os.path.basename(cam_dir)
                
                if cam_name in saved_selection:
                    solution_num = saved_selection[cam_name]
                    
                    # 源文件和目标文件
                    source_file = os.path.join(cam_dir, 'InputMedia', 'calibration', f'cameraIntrinsicsExtrinsics_soln{solution_num}.pickle')
                    target_file = os.path.join(cam_dir, 'cameraIntrinsicsExtrinsics.pickle')
                    
                    # 检查是否需要更新
                    if os.path.exists(source_file) and os.path.exists(target_file):
                        # 比较文件内容是否一致
                        import hashlib
                        with open(source_file, 'rb') as f:
                            source_hash = hashlib.md5(f.read()).hexdigest()
                        with open(target_file, 'rb') as f:
                            target_hash = hashlib.md5(f.read()).hexdigest()
                        
                        if source_hash != target_hash:
                            # 需要更新
                            shutil.copy2(source_file, target_file)
                            logger.info(f"🔄 {cam_name}: 已恢复为方案{solution_num}，确保一致性")
                        else:
                            logger.info(f"✅ {cam_name}: 方案{solution_num}一致性检查通过")
                    elif os.path.exists(source_file):
                        # 目标文件不存在，直接复制
                        shutil.copy2(source_file, target_file)
                        logger.info(f"🔄 {cam_name}: 恢复方案{solution_num}")
                    else:
                        logger.warning(f"⚠️ {cam_name}: 方案{solution_num}文件不存在")
                else:
                    logger.warning(f"⚠️ {cam_name}: 未在保存的选择中找到，将使用默认方案0")
                    # 为缺失的摄像头使用默认方案
                    source_file = os.path.join(cam_dir, 'InputMedia', 'calibration', 'cameraIntrinsicsExtrinsics_soln0.pickle')
                    target_file = os.path.join(cam_dir, 'cameraIntrinsicsExtrinsics.pickle')
                    if os.path.exists(source_file):
                        shutil.copy2(source_file, target_file)
                        logger.info(f"🔄 {cam_name}: 使用默认方案0")
        else:
            logger.info("ℹ️ 未找到之前的标定方案选择记录")

    @staticmethod
    def apply_calibration_selection_to_session(session_path, camera_solution_map=None):
        """
        对现有会话应用标定方案选择的独立工具函数

        Args:
            session_path: 会话目录路径 (如: "Data/session_20250917_140441")
            camera_solution_map: 摄像头方案映射 (如: {"Cam1": 0, "Cam2": 1})

        Returns:
            bool: 是否成功应用选择
        """
        import shutil
        import hashlib

        logger.info(f"🔧 对现有会话应用标定方案选择: {session_path}")

        if not os.path.exists(session_path):
            logger.error(f"❌ 会话目录不存在: {session_path}")
            return False

        # 交互式选择方案（如果未指定映射）
        if camera_solution_map is None:
            camera_solution_map = LocalOpenCapPipeline._interactive_select_for_existing_session(session_path)

        if not camera_solution_map:
            logger.info("用户取消选择或无需更改")
            return True

        success_count = 0
        cam_dirs = glob.glob(os.path.join(session_path, 'Videos', 'Cam*'))

        for cam_dir in cam_dirs:
            cam_name = os.path.basename(cam_dir)

            if cam_name not in camera_solution_map:
                logger.info(f"📷 {cam_name}: 保持当前设置")
                continue

            solution_num = camera_solution_map[cam_name]
            if solution_num not in [0, 1]:
                logger.error(f"❌ {cam_name}: 无效的方案编号 {solution_num}")
                continue

            # 源文件和目标文件路径
            source_file = os.path.join(cam_dir, 'InputMedia', 'calibration', f'cameraIntrinsicsExtrinsics_soln{solution_num}.pickle')
            target_file = os.path.join(cam_dir, 'cameraIntrinsicsExtrinsics.pickle')

            if not os.path.exists(source_file):
                logger.error(f"❌ {cam_name}: 源文件不存在: {source_file}")
                continue

            try:
                # 检查是否需要更新（避免不必要的复制）
                if os.path.exists(target_file):
                    # 计算文件哈希值
                    with open(source_file, 'rb') as f:
                        source_hash = hashlib.md5(f.read()).hexdigest()
                    with open(target_file, 'rb') as f:
                        target_hash = hashlib.md5(f.read()).hexdigest()

                    if source_hash == target_hash:
                        logger.info(f"📷 {cam_name}: 方案{solution_num} 已是当前使用的方案")
                        success_count += 1
                        continue

                # 备份当前文件
                backup_file = target_file + '.backup'
                if os.path.exists(target_file):
                    shutil.copy2(target_file, backup_file)
                    logger.info(f"📷 {cam_name}: 已备份当前文件")

                # 应用新的方案
                shutil.copy2(source_file, target_file)
                logger.info(f"✅ {cam_name}: 已切换到方案{solution_num}")

                # 验证复制结果
                if os.path.exists(target_file):
                    source_size = os.path.getsize(source_file)
                    target_size = os.path.getsize(target_file)
                    if source_size == target_size:
                        logger.info(f"   文件验证通过: {target_size} bytes")
                        success_count += 1

                        # 删除备份文件（成功后）
                        if os.path.exists(backup_file):
                            os.remove(backup_file)
                    else:
                        logger.error(f"❌ {cam_name}: 文件大小验证失败")
                        # 恢复备份
                        if os.path.exists(backup_file):
                            shutil.copy2(backup_file, target_file)
                            logger.info(f"已恢复备份文件")

            except Exception as e:
                logger.error(f"❌ {cam_name}: 应用方案{solution_num}时出错: {str(e)}")
                # 恢复备份
                backup_file = target_file + '.backup'
                if os.path.exists(backup_file):
                    try:
                        shutil.copy2(backup_file, target_file)
                        logger.info(f"已恢复备份文件")
                    except:
                        pass

        logger.info(f"🎯 标定方案应用完成: {success_count}/{len(camera_solution_map)} 个摄像头成功")
        return success_count == len(camera_solution_map)

    @staticmethod
    def _interactive_select_for_existing_session(session_path):
        """
        对现有会话进行交互式标定方案选择

        Args:
            session_path: 会话目录路径

        Returns:
            dict: 摄像头方案映射 (如: {"Cam1": 0, "Cam2": 1})
        """
        import subprocess
        import platform

        print("\n" + "="*60)
        print("🔧 现有会话标定方案调整")
        print("="*60)

        # 查找标定图像
        cal_image_dir = os.path.join(session_path, 'CalibrationImages')
        cam_dirs = glob.glob(os.path.join(session_path, 'Videos', 'Cam*'))
        camera_solution_map = {}

        for cam_dir in cam_dirs:
            cam_name = os.path.basename(cam_dir)

            # 检查方案文件是否存在
            soln0_file = os.path.join(cam_dir, 'InputMedia', 'calibration', 'cameraIntrinsicsExtrinsics_soln0.pickle')
            soln1_file = os.path.join(cam_dir, 'InputMedia', 'calibration', 'cameraIntrinsicsExtrinsics_soln1.pickle')
            current_file = os.path.join(cam_dir, 'cameraIntrinsicsExtrinsics.pickle')

            if not (os.path.exists(soln0_file) and os.path.exists(soln1_file)):
                logger.warning(f"{cam_name}: 未找到完整的方案文件，跳过")
                continue

            # 检查当前使用的方案
            import hashlib
            current_solution = None
            if os.path.exists(current_file):
                with open(current_file, 'rb') as f:
                    current_hash = hashlib.md5(f.read()).hexdigest()
                with open(soln0_file, 'rb') as f:
                    soln0_hash = hashlib.md5(f.read()).hexdigest()
                with open(soln1_file, 'rb') as f:
                    soln1_hash = hashlib.md5(f.read()).hexdigest()

                if current_hash == soln0_hash:
                    current_solution = 0
                elif current_hash == soln1_hash:
                    current_solution = 1

            # 查找标定方案图像
            solution1_img = os.path.join(cal_image_dir, f'extrinsicCalib_{cam_name}.jpg') if os.path.exists(cal_image_dir) else None
            solution2_img = os.path.join(cal_image_dir, f'extrinsicCalib_altSoln_{cam_name}.jpg') if os.path.exists(cal_image_dir) else None

            print(f"\n📷 {cam_name} 标定方案:")
            if current_solution is not None:
                print(f"   当前使用: 方案{current_solution}")
            else:
                print(f"   当前使用: 未知")

            if solution1_img and os.path.exists(solution1_img):
                print(f"   方案0图像: {solution1_img}")
            if solution2_img and os.path.exists(solution2_img):
                print(f"   方案1图像: {solution2_img}")

            # 尝试自动打开图像
            if solution1_img and solution2_img and os.path.exists(solution1_img) and os.path.exists(solution2_img):
                try:
                    if platform.system() == 'Windows':
                        subprocess.run(['start', solution1_img], shell=True, check=False)
                        subprocess.run(['start', solution2_img], shell=True, check=False)
                except:
                    pass

            print("\n选项:")
            print("  0 - 使用方案0 (默认)")
            print("  1 - 使用方案1 (备选)")
            print("  s - 跳过 (保持当前)")

            while True:
                choice = input(f"\n请选择 {cam_name} 的方案 (0/1/s): ").strip().lower()
                if choice == '0':
                    camera_solution_map[cam_name] = 0
                    print(f"✅ {cam_name} 将使用方案0")
                    break
                elif choice == '1':
                    camera_solution_map[cam_name] = 1
                    print(f"✅ {cam_name} 将使用方案1")
                    break
                elif choice == 's':
                    print(f"⏭️ {cam_name} 保持当前设置")
                    break
                else:
                    print("❌ 请输入 0, 1 或 s")

        print("\n" + "="*60)
        if camera_solution_map:
            print(f"📋 将要应用的更改: {camera_solution_map}")
        else:
            print("📋 无更改需要应用")
        print("="*60 + "\n")

        return camera_solution_map if camera_solution_map else None

    def setup_from_videos(self, videos, trial_name, trial_type='dynamic', extrinsicsTrial=False, **kwargs):
        """
        从视频设置试验数据 - 统一的试验设置方法
        
        Args:
            videos: 视频文件列表或目录路径
            trial_name: 试验名称
            trial_type: 试验类型 ('calibration', 'static', 'dynamic')
            extrinsicsTrial: 是否为外参标定试验
            **kwargs: 其他参数
            
        Returns:
            str: 试验名称（如果成功）
        """
        logger.info(f"设置 {trial_type} 试验: {trial_name}")
        
        # 处理视频输入
        if isinstance(videos, str):
            if os.path.isdir(videos):
                # 支持多种视频格式
                video_patterns = ["*.MOV", "*.mp4", "*.MP4", "*.mov", "*.avi", "*.AVI"]
                video_files = []
                for pattern in video_patterns:
                    video_files.extend(glob.glob(os.path.join(videos, pattern)))
            else:
                video_files = [videos]
        else:
            video_files = videos
        
        if not video_files:
            logger.error(f"未找到视频文件: {videos}")
            return None
        
        # 按摄像头组织视频文件
        cameras = self._organize_videos_by_camera(video_files)
        
        if not cameras:
            logger.error("无法识别摄像头")
            return None
        
        # 创建试验目录结构
        for camera_name, video_file in cameras.items():
            # 使用官方目录结构: Videos/CameraName/InputMedia/TrialName/
            camera_dir = os.path.join(self.session_dir, 'Videos', camera_name)
            trial_dir = os.path.join(camera_dir, 'InputMedia', trial_name)
            os.makedirs(trial_dir, exist_ok=True)
            
            # 复制视频文件，保持原始扩展名
            original_ext = os.path.splitext(video_file)[1]
            dest_file = os.path.join(trial_dir, f"{trial_name}{original_ext}")
            if not os.path.exists(dest_file):
                shutil.copy2(video_file, dest_file)
                logger.info(f"复制视频: {os.path.basename(video_file)} -> {camera_name}/{trial_name}/")
        
        return trial_name
    
    def _organize_videos_by_camera(self, video_files):
        """根据文件名将视频按摄像头分组"""
        cameras = {}
        
        for video_file in video_files:
            filename = os.path.basename(video_file)
            
            # 多种摄像头命名模式
            if 'cam1' in filename.lower() or 'camera1' in filename.lower():
                camera_name = 'Cam1'
            elif 'cam2' in filename.lower() or 'camera2' in filename.lower():
                camera_name = 'Cam2'
            elif 'cam3' in filename.lower() or 'camera3' in filename.lower():
                camera_name = 'Cam3'
            elif 'cam4' in filename.lower() or 'camera4' in filename.lower():
                camera_name = 'Cam4'
            elif len(cameras) == 0:
                camera_name = 'Cam1'
            elif len(cameras) == 1:
                camera_name = 'Cam2'
            elif len(cameras) == 2:
                camera_name = 'Cam3'
            elif len(cameras) == 3:
                camera_name = 'Cam4'
            else:
                logger.warning(f"无法识别摄像头: {filename}")
                continue
            
            cameras[camera_name] = video_file
        
        logger.info(f"检测到 {len(cameras)} 个摄像头: {list(cameras.keys())}")
        return cameras
    
    def process_trial(self, trial_name, camera_names=None, trial_type='dynamic'):
        """
        处理单个试验 - 基于官方processTrial逻辑
        
        Args:
            trial_name: 试验名称
            camera_names: 摄像头名称列表
            trial_type: 试验类型 ('calibration', 'static', 'dynamic')
        """
        logger.info(f"开始处理试验: {trial_name} ({trial_type})")
        
        # 确保session metadata存在
        metadata_path = os.path.join(self.session_dir, 'sessionMetadata.yaml')
        if not os.path.exists(metadata_path):
            self.create_session_metadata()
        
        # 关键：在处理非标定试验前，确保标定方案一致性
        if trial_type != 'calibration':
            logger.info("🔒 确保标定方案一致性...")
            self._ensure_calibration_consistency()
        
        # 清理之前的结果文件（只清理输出文件，不删除输入视频）
        self._cleanup_previous_outputs(trial_type, trial_name)
        
        # 准备main()函数参数 - 基于官方逻辑
        main_args = {
            'sessionName': self.session_name,
            'trialName': trial_name,
            'trial_id': trial_name,  # 使用trial_name作为trial_id
            'genericFolderNames': True,  # 重要：使用通用文件夹命名
            'imageUpsampleFactor': self.config['processing']['image_upsample_factor'],
            'cameras_to_use': camera_names or ['all'],
        }
        
        # 根据试验类型设置特定参数
        if trial_type == 'calibration':
            # 处理标定方案选择 - 在标定完成后立即应用
            alternate_extrinsics = None
            if 'alternate_extrinsics' in self.config.get('calibration', {}):
                alternate_extrinsics = self.config['calibration']['alternate_extrinsics']
                logger.info(f"使用配置文件指定的备选标定方案: {alternate_extrinsics}")

            main_args.update({
                'extrinsicsTrial': True,
                'alternateExtrinsics': alternate_extrinsics,  # 为标定试验也添加选择支持
            })
            
        elif trial_type == 'static':
            # 获取标定数据
            calibration_options = self._get_calibration_data('static')
            
            # 应用姿态检测器设置
            pose_params = self._apply_pose_detector_settings(self.config['processing']['pose_detector'])
            
            # 处理标定方案选择 - 静态试验不再需要交互式选择
            alternate_extrinsics = None
            if 'alternate_extrinsics' in self.config.get('calibration', {}):
                alternate_extrinsics = self.config['calibration']['alternate_extrinsics']
                logger.info(f"使用配置文件指定的备选标定方案: {alternate_extrinsics}")
            else:
                # 静态试验使用已保存的标定方案选择，不再进行交互式选择
                saved_selection = self._load_calibration_selection()
                if saved_selection:
                    # 根据保存的选择确定需要使用备选方案的摄像头
                    alternate_cams = [cam for cam, solution in saved_selection.items() if solution == 1]
                    if alternate_cams:
                        alternate_extrinsics = alternate_cams
                        logger.info(f"使用已保存的备选标定方案: {alternate_extrinsics}")
                    else:
                        logger.info("使用已保存的标定方案选择: 所有摄像头使用默认方案")
                else:
                    logger.warning("未找到保存的标定方案选择，将使用默认方案")

            main_args.update({
                'extrinsicsTrial': False,
                'poseDetector': self.config['processing']['pose_detector'],
                'scaleModel': True,  # 关键：模型缩放
                'calibrationOptions': calibration_options,
                'alternateExtrinsics': alternate_extrinsics,  # 添加备选标定方案
                **pose_params
            })
            
        elif trial_type == 'dynamic':
            # 获取标定数据和模型
            calibration_options = self._get_calibration_data('dynamic')
            model_available = self._get_model_and_metadata()
            
            # 应用姿态检测器设置
            pose_params = self._apply_pose_detector_settings(self.config['processing']['pose_detector'])
            
            main_args.update({
                'extrinsicsTrial': False,
                'poseDetector': self.config['processing']['pose_detector'],
                'scaleModel': False,  # 关键：不缩放模型（使用已有的缩放模型）
                'calibrationOptions': calibration_options,
                **pose_params
            })
            
            if not model_available:
                logger.warning("未找到缩放模型，动态试验可能使用默认模型")
        
        # 运行main()函数
        try:
            logger.info(f"调用main()函数，参数: {main_args}")
            success = opencap_main(**main_args)
            
            if success:
                logger.info(f"✅ {trial_type} 试验处理成功: {trial_name}")
                
                # 保存试验特定的输出
                if trial_type == 'static':
                    self._save_static_trial_outputs(trial_name)
                    self.static_trial_name = trial_name
                elif trial_type == 'calibration':
                    # 标定完成后，立即进行交互式选择并应用
                    if self.config.get('calibration', {}).get('interactive_selection', False):
                        logger.info("🎯 标定完成，开始交互式方案选择...")
                        alternate_cams = self._interactive_calibration_selection()
                        if alternate_cams:
                            self._apply_calibration_selection(alternate_cams)
                            logger.info(f"✅ 已应用用户选择的标定方案: {alternate_cams}")
                        else:
                            # 即使用户选择了全部默认方案，也要保存选择记录
                            logger.info("✅ 用户选择使用所有默认标定方案")
                            default_selection = {}
                            cam_dirs = glob.glob(os.path.join(self.session_dir, 'Videos', 'Cam*'))
                            for cam_dir in cam_dirs:
                                cam_name = os.path.basename(cam_dir)
                                default_selection[cam_name] = 0  # 默认方案
                            self._save_calibration_selection(default_selection)

                return True
            else:
                logger.error(f"❌ {trial_type} 试验处理失败: {trial_name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ {trial_type} 试验处理异常: {str(e)}")
            
            # 尝试保存部分结果 - 基于官方错误处理逻辑
            try:
                self._save_partial_results(trial_name, trial_type)
            except Exception as save_e:
                logger.warning(f"保存部分结果失败: {str(save_e)}")
            
            return False
    
    def _save_static_trial_outputs(self, static_trial_name):
        """保存静态试验产生的重要数据"""
        logger.info("保存静态试验输出数据...")
        
        static_outputs = {
            'trial_name': static_trial_name,
            'processing_time': datetime.now().isoformat(),
            'outputs': {}
        }
        
        try:
            # 查找并记录重要输出文件
            outputs_to_check = {
                'scaled_model': os.path.join(self.session_dir, 'OpenSimData', 'Model', '*_scaled.osim'),
                'neutral_images': os.path.join(self.session_dir, 'NeutralPoseImages'),
                'marker_data': os.path.join(self.session_dir, 'MarkerData*', f'*{static_trial_name}*'),
                'pose_data': os.path.join(self.session_dir, 'Videos', '*', 'OutputPkl', static_trial_name)
            }
            
            for output_type, pattern in outputs_to_check.items():
                files = glob.glob(pattern)
                if files:
                    static_outputs['outputs'][output_type] = files
                    logger.info(f"  {output_type}: {len(files)} 个文件")
            
            # 保存静态试验输出记录
            output_file = os.path.join(self.session_dir, f'static_trial_outputs_{static_trial_name}.yaml')
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(static_outputs, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"静态试验输出记录已保存: {output_file}")
            
        except Exception as e:
            logger.error(f"保存静态试验输出失败: {str(e)}")
    
    def _save_partial_results(self, trial_name, trial_type):
        """保存部分结果 - 基于官方错误处理逻辑"""
        logger.info(f"尝试保存 {trial_type} 试验的部分结果...")
        
        # 查找可能已经生成的文件
        partial_outputs = {}
        
        # 姿态检测结果
        pose_files = glob.glob(os.path.join(self.session_dir, 'Videos', '*', 'OutputPkl', trial_name, '*keypoints*.pkl'))
        if pose_files:
            partial_outputs['pose_detection'] = pose_files
            logger.info(f"保存姿态检测结果: {len(pose_files)} 个文件")
        
        # 标定结果
        if trial_type == 'calibration':
            calib_files = glob.glob(os.path.join(self.session_dir, 'Videos', '*', 'cameraIntrinsicsExtrinsics.pickle'))
            if calib_files:
                partial_outputs['calibration'] = calib_files
                logger.info(f"保存标定结果: {len(calib_files)} 个文件")
        
        # 保存部分结果记录
        if partial_outputs:
            partial_file = os.path.join(self.session_dir, f'partial_results_{trial_name}_{trial_type}.yaml')
            with open(partial_file, 'w', encoding='utf-8') as f:
                yaml.dump({
                    'trial_name': trial_name,
                    'trial_type': trial_type,
                    'partial_outputs': partial_outputs,
                    'save_time': datetime.now().isoformat()
                }, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"部分结果已保存: {partial_file}")
    
    def process_session(self, video_directory, calibration_directory=None, static_directory=None):
        """
        处理完整会话 - 基于官方逻辑的完整流程
        
        Args:
            video_directory: 运动视频目录
            calibration_directory: 标定视频目录（可选）
            static_directory: 静态姿态视频目录（可选）
        """
        logger.info("="*60)
        logger.info("开始本地OpenCap会话处理")
        logger.info("="*60)
        
        try:
            # 创建会话元数据
            self.create_session_metadata()
            
            # 确定静态目录 - 从配置或参数获取
            if static_directory is None:
                static_directory = self.config.get('static_videos')
                if static_directory:
                    logger.info(f"从配置获取静态目录: {static_directory}")
            
            # 获取摄像头列表
            video_patterns = ["*.MOV", "*.mp4", "*.MP4", "*.mov", "*.avi", "*.AVI"]
            video_files = []
            for pattern in video_patterns:
                video_files.extend(glob.glob(os.path.join(video_directory, pattern)))

            if not video_files:
                raise ValueError(f"未找到视频文件: {video_directory}")
            
            cameras = self._organize_videos_by_camera(video_files)
            camera_names = list(cameras.keys())
            
            logger.info(f"检测到摄像头: {camera_names}")
            
            # 1. 处理标定试验（如果提供）
            calib_trial_name = None
            if calibration_directory and os.path.exists(calibration_directory):
                logger.info("处理标定试验...")
                calib_trial_name = self.setup_from_videos(
                    videos=calibration_directory,
                    trial_name='calibration',
                    trial_type='calibration',
                    extrinsicsTrial=True
                )
                if calib_trial_name:
                    calib_success = self.process_trial(calib_trial_name, camera_names, 'calibration')
                    if calib_success:
                        logger.info("✅ 标定试验处理成功")
                    else:
                        logger.warning("⚠️ 标定试验处理失败，但继续处理其他试验")
            
            # 2. 处理静态试验（如果提供）
            static_trial_name = None
            static_success = True
            if static_directory and os.path.exists(static_directory):
                logger.info("处理静态试验...")
                static_trial_name = self.setup_from_videos(
                    videos=static_directory,
                    trial_name='neutral',  # 官方标准静态试验名称
                    trial_type='static'
                )
                if static_trial_name:
                    static_success = self.process_trial(static_trial_name, camera_names, 'static')
                    if static_success:
                        logger.info("✅ 静态试验处理成功，已生成缩放后的模型")
                        self.static_trial_name = static_trial_name
                    else:
                        logger.warning("⚠️ 静态试验处理失败，动态试验将使用默认模型")
            elif self.config.get('static', {}).get('required', False):
                logger.error("❌ 配置要求静态试验，但未找到静态视频")
                return False
            else:
                logger.info("ℹ️ 未提供静态试验，将使用默认OpenSim模型进行动态分析")
            
            # 3. 处理动态试验（主要的运动视频）
            logger.info("处理动态试验...")
            motion_trial_name = self.setup_from_videos(
                videos=video_directory,
                trial_name=f"motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                trial_type='dynamic'
            )
            
            if motion_trial_name:
                success = self.process_trial(motion_trial_name, camera_names, 'dynamic')
                
                if success:
                    logger.info("="*60)
                    logger.info("✅ 会话处理成功完成!")
                    logger.info(f"会话目录: {self.session_dir}")
                    logger.info("="*60)
                    
                    # 生成处理报告
                    self._generate_report(motion_trial_name, camera_names)
                    return True
                else:
                    logger.error("❌ 动态试验处理失败")
                    return False
            else:
                logger.error("❌ 动态试验设置失败")
                return False
                
        except Exception as e:
            logger.error(f"❌ 会话处理失败: {str(e)}")
            logger.error(f"详细错误: {traceback.format_exc()}")
            return False
    
    def create_session_metadata(self):
        """创建会话元数据文件 - 兼容官方格式"""
        logger.info("=" * 60)
        logger.info("📋 创建会话元数据")
        logger.info("=" * 60)

        # 生成默认的摄像头模型映射
        camera_models = {}
        for i in range(1, 5):  # 支持最多4个摄像头
            cam_name = f'Cam{i}'
            camera_models[cam_name] = f'GenericCamera{i}'

        # 从配置获取棋盘格放置方式，默认为backWall
        checkerboard_placement = self.config.get('calibration', {}).get('checkerboard', {}).get('placement', 'backWall')

        metadata = {
            'sessionWithCalibration': True,
            'mass_kg': self.config['session']['subject_mass'],
            'height_m': self.config['session']['subject_height'] / 100.0,
            'sessionDate': datetime.now().isoformat(),
            'sessionDescription': self.config['session']['description'],
            'checkerBoard': {
                'black2BlackCornersWidth_n': self.config['calibration']['checkerboard']['dimensions'][0],
                'black2BlackCornersHeight_n': self.config['calibration']['checkerboard']['dimensions'][1],
                'squareSideLength_mm': self.config['calibration']['checkerboard']['square_size'],
                'placement': checkerboard_placement
            },
            'posemodel': self.config['processing']['pose_detector'],  # 官方字段名
            'poseDetector': self.config['processing']['pose_detector'],
            'resolutionPoseDetection': self.config['processing']['resolution'],
            'augmenter_model': self.config['processing']['augmenter_model'],
            'imageUpsampleFactor': self.config['processing']['image_upsample_factor'],
            'cameraModel': camera_models,
            'openSimModel': 'LaiUhlrich2022',
            'markerAugmentationSettings': {
                'markerAugmenterModel': 'LSTM'
            },
            'localProcessing': True,
            'created_by': 'LocalOpenCapPipeline'
        }

        # 添加强制朝向配置（如果存在）
        if 'force_correct_orientation' in self.config.get('calibration', {}):
            force_orientation = self.config['calibration']['force_correct_orientation']
            metadata['calibration'] = {
                'force_correct_orientation': force_orientation
            }
            logger.info(f"   🔒 强制朝向配置: {force_orientation}")
            if force_orientation:
                logger.info("      ⚠️ 将忽略棋盘格倒置检测，强制使用正确朝向")

        # 详细记录元数据信息
        logger.info("   📊 会话基本信息:")
        logger.info(f"      会话名称: {self.session_name}")
        logger.info(f"      受试者体重: {metadata['mass_kg']} kg")
        logger.info(f"      受试者身高: {metadata['height_m']} m")
        logger.info(f"      处理模式: 本地处理")

        logger.info("   🎯 标定板配置:")
        logger.info(f"      尺寸: {metadata['checkerBoard']['black2BlackCornersWidth_n']} x {metadata['checkerBoard']['black2BlackCornersHeight_n']}")
        logger.info(f"      正方形边长: {metadata['checkerBoard']['squareSideLength_mm']} mm")
        logger.info(f"      放置方式: {metadata['checkerBoard']['placement']}")
        logger.info("      ⚠️  放置方式说明:")
        if checkerboard_placement == 'backWall':
            logger.info("         - backWall: 棋盘格垂直放置在背景墙上")
            logger.info("         - 这将触发棋盘格倒置检测")
            logger.info("         - 影响坐标系转换: Y轴±90°, Z轴可能180°")
        elif checkerboard_placement == 'ground':
            logger.info("         - ground: 棋盘格水平放置在地面")
            logger.info("         - 坐标系转换: X轴90°, Y轴90°")
        else:
            logger.info(f"         - {checkerboard_placement}: 自定义放置方式")

        logger.info("   🔧 处理配置:")
        logger.info(f"      姿态检测器: {metadata['poseDetector']}")
        logger.info(f"      检测分辨率: {metadata['resolutionPoseDetection']}")
        logger.info(f"      增强模型: {metadata['augmenter_model']}")
        logger.info(f"      图像上采样因子: {metadata['imageUpsampleFactor']}")
        logger.info(f"      OpenSim模型: {metadata['openSimModel']}")

        logger.info("   📷 摄像头配置:")
        for cam_name, model in camera_models.items():
            logger.info(f"      {cam_name}: {model}")

        metadata_path = os.path.join(self.session_dir, 'sessionMetadata.yaml')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"   📁 元数据已保存: {metadata_path}")
        logger.info("=" * 60)

        self.session_metadata = metadata
        return metadata_path
    
    def _diagnose_coordinate_system_issues(self):
        """诊断可能的坐标系问题"""
        logger.info("🔍 开始坐标系问题诊断...")

        # 检查TRC文件
        trc_files = glob.glob(os.path.join(self.session_dir, 'MarkerData', '**', '*.trc'), recursive=True)

        if not trc_files:
            logger.warning("   未找到TRC文件，无法进行诊断")
            return

        for trc_file in trc_files:
            logger.info(f"   📊 分析TRC文件: {os.path.basename(trc_file)}")

            try:
                # 简单读取TRC文件前几行来获取数据
                with open(trc_file, 'r') as f:
                    lines = f.readlines()

                # 跳过头部，找到数据行
                data_start = -1
                for i, line in enumerate(lines):
                    if 'Frame#' in line or 'Time' in line:
                        data_start = i + 1
                        break

                if data_start > 0 and data_start < len(lines):
                    # 读取第一帧数据
                    data_line = lines[data_start].strip().split('\t')
                    if len(data_line) > 10:  # 确保有足够的数据
                        # 提取坐标数据 (跳过Frame#和Time列)
                        coords = []
                        for i in range(2, len(data_line), 3):  # X, Y, Z坐标
                            if i+2 < len(data_line):
                                try:
                                    x = float(data_line[i])
                                    y = float(data_line[i+1])
                                    z = float(data_line[i+2])
                                    if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                                        coords.append([x, y, z])
                                except (ValueError, IndexError):
                                    continue

                        if coords:
                            coords = np.array(coords)

                            # 分析坐标分布
                            x_range = [np.min(coords[:, 0]), np.max(coords[:, 0])]
                            y_range = [np.min(coords[:, 1]), np.max(coords[:, 1])]
                            z_range = [np.min(coords[:, 2]), np.max(coords[:, 2])]

                            centroid = np.mean(coords, axis=0)

                            logger.info(f"      坐标范围分析:")
                            logger.info(f"        X: [{x_range[0]:.1f}, {x_range[1]:.1f}] mm")
                            logger.info(f"        Y: [{y_range[0]:.1f}, {y_range[1]:.1f}] mm")
                            logger.info(f"        Z: [{z_range[0]:.1f}, {z_range[1]:.1f}] mm")
                            logger.info(f"        重心: [{centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}] mm")

                            # 诊断问题
                            issues = []

                            # 检查Y轴是否为垂直轴
                            y_span = y_range[1] - y_range[0]
                            x_span = x_range[1] - x_range[0]
                            z_span = z_range[1] - z_range[0]

                            if y_span < max(x_span, z_span) * 0.5:
                                issues.append("Y轴分布范围过小，可能不是垂直轴")

                            # 检查重心位置
                            if abs(centroid[1]) > 2000:
                                issues.append(f"Y轴重心异常: {centroid[1]:.1f}mm")

                            # 检查人体尺度
                            max_distance = 0
                            for i in range(len(coords)):
                                for j in range(i+1, len(coords)):
                                    dist = np.linalg.norm(coords[i] - coords[j])
                                    max_distance = max(max_distance, dist)

                            if max_distance < 800:  # 人体最大距离应该大于80cm
                                issues.append(f"人体尺度过小: 最大距离仅{max_distance:.1f}mm")
                            elif max_distance > 5000:  # 人体最大距离不应该超过5m
                                issues.append(f"人体尺度过大: 最大距离达{max_distance:.1f}mm")

                            # 报告问题
                            if issues:
                                logger.warning(f"      ⚠️ 发现潜在问题:")
                                for issue in issues:
                                    logger.warning(f"        - {issue}")

                                logger.info(f"      💡 建议检查:")
                                logger.info(f"        - 棋盘格放置方式是否正确设置")
                                logger.info(f"        - 标定方案选择是否合适")
                                logger.info(f"        - 摄像头标定质量")
                            else:
                                logger.info(f"      ✅ 坐标系看起来正常")

            except Exception as e:
                logger.warning(f"   分析TRC文件时出错: {str(e)}")

    def _generate_report(self, trial_name, camera_names):
        """生成处理报告"""
        # 先进行坐标系诊断
        self._diagnose_coordinate_system_issues()

        report = {
            'session_name': self.session_name,
            'processing_date': datetime.now().isoformat(),
            'trial_name': trial_name,
            'camera_names': camera_names,
            'static_trial': self.static_trial_name,
            'configuration': self.config,
            'output_files': self._list_output_files(),
            'pipeline_version': 'LocalOpenCapPipeline_v2.0'
        }

        report_path = os.path.join(self.session_dir, 'processing_report.yaml')
        with open(report_path, 'w', encoding='utf-8') as f:
            yaml.dump(report, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"处理报告已保存: {report_path}")
    
    def _list_output_files(self):
        """列出输出文件"""
        output_files = {}
        
        file_patterns = {
            'trc_files': '**/*.trc',
            'opensim_files': '**/OpenSimData/**/*',
            'videos': '**/VisualizerVideos/**/*',
            'pose_data': '**/*keypoints*.pkl',
            'calibration': '**/cameraIntrinsicsExtrinsics.pickle',
            'marker_data': '**/MarkerData*/**/*'
        }
        
        for file_type, pattern in file_patterns.items():
            files = glob.glob(os.path.join(self.session_dir, pattern), recursive=True)
            output_files[file_type] = files
        
        return output_files


# 便捷函数
def create_config_template(output_path):
    """创建配置文件模板"""
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'local_config_template.yaml')
    
    if os.path.exists(template_path):
        # 检查是否是同一个文件
        if os.path.abspath(template_path) != os.path.abspath(output_path):
            shutil.copy2(template_path, output_path)
            logger.info(f"配置模板已复制到: {output_path}")
        
        # 更新会话名称为当前时间
        with open(output_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if 'session' in config:
            config['session']['name'] = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    else:
        # 如果模板不存在，创建一个最基本的模板
        pipeline = LocalOpenCapPipeline()
        config = pipeline._load_template_config()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"配置文件模板已创建: {output_path}")

def apply_calibration_selection(session_path, camera_solution_map=None):
    """
    便捷函数：对现有会话应用标定方案选择

    Args:
        session_path: 会话目录路径 (如: "E:/path/to/Data/session_20250917_140441")
        camera_solution_map: 摄像头方案映射 (如: {"Cam1": 0, "Cam2": 1})
                            如果不指定，将进行交互式选择

    Returns:
        bool: 是否成功应用选择

    Example:
        # 交互式选择
        apply_calibration_selection("E:/guge/opencap-core-local/Data/session_20250917_140441")

        # 指定映射
        apply_calibration_selection(
            "E:/guge/opencap-core-local/Data/session_20250917_140441",
            {"Cam1": 0, "Cam2": 1}
        )
    """
    return LocalOpenCapPipeline.apply_calibration_selection_to_session(session_path, camera_solution_map)

def run_local_opencap(video_dir, calibration_dir=None, static_dir=None, config_file=None, **kwargs):
    """
    便捷函数：运行本地OpenCap处理
    
    Args:
        video_dir: 运动视频目录
        calibration_dir: 标定视频目录（可选）
        static_dir: 静态姿态视频目录（可选）
        config_file: 配置文件路径（可选）
        **kwargs: 其他配置参数
    """
    
    # 如果没有提供static_dir，尝试查找默认位置
    if static_dir is None:
        potential_static_dirs = [
            './LocalData/Static',
            os.path.join(os.path.dirname(video_dir), 'Static'),
            os.path.join(video_dir, '../Static'),
            video_dir + '_static',
            video_dir.replace('Videos', 'Static').replace('dynamic', 'static')
        ]
        
        for potential_dir in potential_static_dirs:
            if os.path.exists(potential_dir):
                # 检查是否包含视频文件
                video_patterns = ["*.MOV", "*.mp4", "*.MP4", "*.mov", "*.avi", "*.AVI"]
                found_videos = []
                for pattern in video_patterns:
                    found_videos.extend(glob.glob(os.path.join(potential_dir, pattern)))
                if found_videos:
                    static_dir = potential_dir
                    logger.info(f"自动找到静态视频目录: {static_dir}")
                    break
    
    # 合并配置
    config = {}
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        # 如果没有配置文件，直接加载模板配置
        template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'local_config_template.yaml')
        if os.path.exists(template_path):
            with open(template_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                # 更新会话名称为当前时间
                config['session']['name'] = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            # 如果模板不存在，创建最基本的默认配置
            config = {
                'session': {
                    'name': f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'description': '本地OpenCap会话',
                    'subject_mass': 67.0,
                    'subject_height': 170.0
                },
                'calibration': {
                    'checkerboard': {
                        'dimensions': [5, 4],
                        'square_size': 35
                    }
                },
                'processing': {
                    'pose_detector': 'OpenPose',
                    'resolution': '1x736',
                    'image_upsample_factor': 4,
                    'augmenter_model': 'v0.3'
                },
                'directories': {
                    'input_videos': './LocalData/Videos',
                    'static_videos': './LocalData/Static'
                }
            }
    
    # 应用kwargs覆盖
    for key, value in kwargs.items():
        if '.' in key:
            # 处理嵌套键，如 'processing.pose_detector'
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        else:
            # 简化的键映射
            if key == 'pose_detector':
                config.setdefault('processing', {})['pose_detector'] = value
            elif key == 'resolution':
                config.setdefault('processing', {})['resolution'] = value
            else:
                config[key] = value
    
    # 创建并运行管道
    pipeline = LocalOpenCapPipeline(config_dict=config)
    return pipeline.process_session(video_dir, calibration_dir, static_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='本地OpenCap处理管道')
    parser.add_argument('video_dir', help='运动视频目录')
    parser.add_argument('--calibration-dir', '-c', help='标定视频目录')
    parser.add_argument('--static-dir', '-s', help='静态姿态视频目录')
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
        static_dir=args.static_dir,
        config_file=args.config,
        pose_detector=args.pose_detector,
        resolution=args.resolution
    )
    
    sys.exit(0 if success else 1)