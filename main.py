"""
    @authors: Scott Uhlrich, Antoine Falisse, Łukasz Kidziński
    
    This function calibrates the cameras, runs the pose detection algorithm, 
    reconstructs the 3D marker positions, augments the marker set,
    and runs the OpenSim pipeline.

"""

import os 
import glob
import numpy as np
import yaml
import json
import traceback

import logging
logging.basicConfig(level=logging.INFO)

from utils import importMetadata, loadCameraParameters, getVideoExtension
from utils import getDataDirectory, getOpenPoseDirectory, getMMposeDirectory
from utilsChecker import saveCameraParameters
from utilsChecker import calcExtrinsicsFromVideo
from utilsChecker import isCheckerboardUpsideDown
from utilsChecker import autoSelectExtrinsicSolution
from utilsChecker import synchronizeVideos
from utilsChecker import triangulateMultiviewVideo
from utilsChecker import writeTRCfrom3DKeypoints
from utilsChecker import popNeutralPoseImages
from utilsChecker import rotateIntrinsics
from utilsDetector  import runPoseDetector
from utilsAugmenter import augmentTRC
from utilsOpenSim import runScaleTool, getScaleTimeRange, runIKTool, generateVisualizerJson

def main(sessionName, trialName, trial_id, cameras_to_use=['all'],
         intrinsicsFinalFolder='Deployed', isDocker=False,
         extrinsicsTrial=False, alternateExtrinsics=None, 
         calibrationOptions=None,
         markerDataFolderNameSuffix=None, imageUpsampleFactor=4,
         poseDetector='OpenPose', resolutionPoseDetection='default', 
         scaleModel=False, bbox_thr=0.8, augmenter_model='v0.3',
         genericFolderNames=False, offset=True, benchmark=False,
         dataDir=None, overwriteAugmenterModel=False,
         filter_frequency='default', overwriteFilterFrequency=False,
         scaling_setup='upright_standing_pose', overwriteScalingSetup=False,
         overwriteCamerasToUse=False):

    # %% High-level settings.
    # Camera calibration.
    runCameraCalibration = True
    # Pose detection.
    runPoseDetection = True
    # Video Synchronization.
    runSynchronization = True
    # Triangulation.
    runTriangulation = True
    # Marker augmentation.
    runMarkerAugmentation = True
    # OpenSim pipeline.
    runOpenSimPipeline = True    
    # High-resolution for OpenPose.
    resolutionPoseDetection = resolutionPoseDetection
    # Set to False to only generate the json files (default is True).
    # This speeds things up and saves storage space.
    generateVideo = True
    # This is a hack to handle a mismatch between the use of mmpose and hrnet,
    # and between the use of OpenPose and openpose.
    if poseDetector == 'hrnet':
        poseDetector = 'mmpose'        
    elif poseDetector == 'openpose':
        poseDetector = 'OpenPose'
    if poseDetector == 'mmpose':
        outputMediaFolder = 'OutputMedia_mmpose' + str(bbox_thr)
    elif poseDetector == 'OpenPose':
        outputMediaFolder = 'OutputMedia_' + resolutionPoseDetection
    
    # %% Special case: extrinsics trial.
    # For that trial, we only calibrate the cameras.
    if extrinsicsTrial:
        runCameraCalibration = True
        runPoseDetection = False
        runSynchronization = False
        runTriangulation =  False
        runMarkerAugmentation = False
        runOpenSimPipeline = False
        
    # %% Paths and metadata. This gets defined through web app.
    baseDir = os.path.dirname(os.path.abspath(__file__))
    if dataDir is None:
        dataDir = getDataDirectory(isDocker)
    if 'dataDir' not in locals():
        sessionDir = os.path.join(baseDir, 'Data', sessionName)
    else:
        sessionDir = os.path.join(dataDir, 'Data', sessionName)
    sessionMetadata = importMetadata(os.path.join(sessionDir,
                                                  'sessionMetadata.yaml'))
    
    # If augmenter model defined through web app.
    # If overwriteAugmenterModel is True, the augmenter model is the one
    # passed as an argument to main(). This is useful for local testing.
    if 'augmentermodel' in sessionMetadata and not overwriteAugmenterModel:
        augmenterModel = sessionMetadata['augmentermodel']
    else:
        augmenterModel = augmenter_model
        
    # Lowpass filter frequency of 2D keypoints for gait and everything else.
    # If overwriteFilterFrequency is True, the filter frequency is the one
    # passed as an argument to main(). This is useful for local testing.
    if 'filterfrequency' in sessionMetadata and not overwriteFilterFrequency:
        filterfrequency = sessionMetadata['filterfrequency']
    else:
        filterfrequency = filter_frequency
    if filterfrequency == 'default':
        filtFreqs = {'gait':12, 'default':500} # defaults to framerate/2
    else:
        filtFreqs = {'gait':filterfrequency, 'default':filterfrequency}

    # If scaling setup defined through web app.
    # If overwriteScalingSetup is True, the scaling setup is the one
    # passed as an argument to main(). This is useful for local testing.
    if 'scalingsetup' in sessionMetadata and not overwriteScalingSetup:
        scalingSetup = sessionMetadata['scalingsetup']
    else:
        scalingSetup = scaling_setup

    # If camerastouse is in sessionMetadata, reprocess with specified cameras.
    # This allows reprocessing trials with missing videos. If
    # overwriteCamerasToUse is True, the camera selection is the one
    # passed as an argument to main(). This is useful for local testing.
    if 'camerastouse' in sessionMetadata and not overwriteCamerasToUse:
        camerasToUse = sessionMetadata['camerastouse']
    else:
        camerasToUse = cameras_to_use

    # %% Paths to pose detector folder for local testing.
    if poseDetector == 'OpenPose':
        poseDetectorDirectory = getOpenPoseDirectory(isDocker)
    elif poseDetector == 'mmpose':
        poseDetectorDirectory = getMMposeDirectory(isDocker)    
        
    # %% Create marker folders
    # Create output folder.
    if genericFolderNames:
        markerDataFolderName = os.path.join('MarkerData') 
    else:
        if poseDetector == 'mmpose':
            suff_pd = '_' + str(bbox_thr)
        elif poseDetector == 'OpenPose':
            suff_pd = '_' + resolutionPoseDetection
                
        markerDataFolderName = os.path.join('MarkerData', 
                                            poseDetector + suff_pd) 
        if not markerDataFolderNameSuffix is None:
            markerDataFolderName = os.path.join(markerDataFolderName,
                                                markerDataFolderNameSuffix)
    preAugmentationDir = os.path.join(sessionDir, markerDataFolderName,
                                      'PreAugmentation')
    os.makedirs(preAugmentationDir, exist_ok=True)
    
    # Create augmented marker folders as well
    if genericFolderNames:
        postAugmentationDir = os.path.join(sessionDir, markerDataFolderName, 
                                           'PostAugmentation')
    else:
        postAugmentationDir = os.path.join(
            sessionDir, markerDataFolderName, 
            'PostAugmentation_{}'.format(augmenterModel))
    os.makedirs(postAugmentationDir, exist_ok=True)
        
    # %% Dump settings in yaml.
    if not extrinsicsTrial:
        pathSettings = os.path.join(postAugmentationDir,
                                    'Settings_' + trial_id + '.yaml')
        settings = {
            'poseDetector': poseDetector, 
            'augmenter_model': augmenterModel, 
            'imageUpsampleFactor': imageUpsampleFactor,
            'openSimModel': sessionMetadata['openSimModel'],
            'scalingSetup': scalingSetup,
            'filterFrequency': filterfrequency,
            }
        if poseDetector == 'OpenPose':
            settings['resolutionPoseDetection'] = resolutionPoseDetection
        elif poseDetector == 'mmpose':
            settings['bbox_thr'] = bbox_thr
        with open(pathSettings, 'w', encoding='utf-8') as file:
            yaml.dump(settings, file)

    # %% Camera calibration.
    if runCameraCalibration:    
        # Get checkerboard parameters from metadata.
        CheckerBoardParams = {
            'dimensions': (
                sessionMetadata['checkerBoard']['black2BlackCornersWidth_n'],
                sessionMetadata['checkerBoard']['black2BlackCornersHeight_n']),
            'squareSize': 
                sessionMetadata['checkerBoard']['squareSideLength_mm']}       
        # Camera directories and models.
        cameraDirectories = {}
        cameraModels = {}
        for pathCam in glob.glob(os.path.join(sessionDir, 'Videos', 'Cam*')):
            if os.name == 'nt': # windows
                camName = pathCam.split('\\')[-1]
            elif os.name == 'posix': # ubuntu
                camName = pathCam.split('/')[-1]
            cameraDirectories[camName] = os.path.join(sessionDir, 'Videos',
                                                      pathCam)
            # 支持通用摄像头，不仅仅是iPhone
            if 'cameraModel' in sessionMetadata:
                cameraModels[camName] = sessionMetadata['cameraModel'][camName]
            elif 'iphoneModel' in sessionMetadata:
                # 向后兼容旧的字段名
                cameraModels[camName] = sessionMetadata['iphoneModel'][camName]
            else:
                # 如果没有摄像头型号信息，使用通用命名
                cameraModels[camName] = f'GenericCamera_{camName}'        
        
        # Get cameras' intrinsics and extrinsics.     
        # Load parameters if saved, compute and save them if not.
        CamParamDict = {}
        loadedCamParams = {}
        for camName in cameraDirectories:
            camDir = cameraDirectories[camName]
            # Intrinsics ######################################################
            # Intrinsics and extrinsics already exist for this session.
            if os.path.exists(
                    os.path.join(camDir,"cameraIntrinsicsExtrinsics.pickle")):
                logging.info("Load extrinsics for {} - already existing".format(
                    camName))
                CamParams = loadCameraParameters(
                    os.path.join(camDir, "cameraIntrinsicsExtrinsics.pickle"))
                loadedCamParams[camName] = True
                
            # Extrinsics do not exist for this session.
            else:
                logging.info("Compute extrinsics for {} - not yet existing".format(camName))
                # Intrinsics ##################################################
                # Intrinsics directories.
                intrinsicDir = os.path.join(baseDir, 'CameraIntrinsics',
                                            cameraModels[camName])
                permIntrinsicDir = os.path.join(intrinsicDir, 
                                                intrinsicsFinalFolder)            
                # Intrinsics exist.
                if os.path.exists(permIntrinsicDir):
                    CamParams = loadCameraParameters(
                        os.path.join(permIntrinsicDir,
                                      'cameraIntrinsics.pickle'))                    
                # Intrinsics do not exist throw an error. Eventually the
                # webapp will give you the opportunity to compute them.
                
                else:
                    # 对于本地处理，如果是通用摄像头且没有预计算的内参，
                    # 尝试从当前标定视频计算内参
                    if 'GenericCamera' in cameraModels[camName] and extrinsicsTrial:
                        logging.info(f"为通用摄像头 {camName} 计算内参...")
                        try:
                            # 导入本地内参计算函数
                            from main_calcIntrinsics_local import calibrateCameraFromVideo
                            
                            # 从当前标定视频计算内参
                            pathVideoWithoutExtension = os.path.join(
                                camDir, 'InputMedia', trialName, trial_id)
                            extension = getVideoExtension(pathVideoWithoutExtension)
                            calibrationVideoPath = pathVideoWithoutExtension + extension
                            
                            if os.path.exists(calibrationVideoPath):
                                # 使用默认图像数量25进行标定
                                intrinsic_data = calibrateCameraFromVideo(
                                    calibrationVideoPath, CheckerBoardParams, 25)
                                if intrinsic_data is None:
                                    raise Exception(f"从视频计算内参失败: {calibrationVideoPath}")
                                
                                # 创建兼容的内参字典格式（与现有函数兼容）
                                CamParams = {
                                    'intrinsicMat': intrinsic_data['intrinsicMat'],
                                    'distortion': intrinsic_data['distortion'],
                                    'imageSize': intrinsic_data['imageSize']
                                }
                                logging.info(f"成功为 {camName} 计算内参")
                            else:
                                raise Exception(f"标定视频不存在: {calibrationVideoPath}")
                                
                        except ImportError:
                            exception = "无法导入本地内参计算模块。请确保 main_calcIntrinsics_local.py 存在。"
                            raise Exception(exception, exception)
                        except Exception as e:
                            exception = f"为通用摄像头计算内参失败: {str(e)}"
                            raise Exception(exception, exception)
                    else:
                        # 对于非标定试验，如果是通用摄像头且没有内参，提供更明确的错误信息
                        if 'GenericCamera' in cameraModels[camName]:
                            exception = f"通用摄像头 {camName} 缺少内参数据。请先运行标定试验以计算内参，或手动提供内参文件。"
                            raise Exception(exception, exception)
                        else:
                            exception = "Intrinsics don't exist for your camera model. OpenCap supports all iOS devices released in 2018 or later: https://www.opencap.ai/get-started."
                            raise Exception(exception, exception)
                        
                # Extrinsics ##################################################
                # Compute extrinsics from images popped out of this trial.
                # Hopefully you get a clean shot of the checkerboard in at
                # least one frame of each camera.
                useSecondExtrinsicsSolution = (
                    alternateExtrinsics is not None and
                    camName in alternateExtrinsics)
                logging.info(f"   🎯 {camName} 外参计算设置:")
                logging.info(f"      使用备选解决方案: {useSecondExtrinsicsSolution}")

                pathVideoWithoutExtension = os.path.join(
                    camDir, 'InputMedia', trialName, trial_id)
                extension = getVideoExtension(pathVideoWithoutExtension)
                extrinsicPath = os.path.join(camDir, 'InputMedia', trialName,
                                             trial_id + extension)
                logging.info(f"      标定视频路径: {extrinsicPath}")

                # Modify intrinsics if camera view is rotated
                logging.info(f"      图像上采样因子: {imageUpsampleFactor}")
                CamParams = rotateIntrinsics(CamParams,extrinsicPath)

                # for 720p, imageUpsampleFactor=4 is best for small board
                try:
                    CamParams = calcExtrinsicsFromVideo(
                        extrinsicPath,CamParams, CheckerBoardParams,
                        visualize=False, imageUpsampleFactor=imageUpsampleFactor,
                        useSecondExtrinsicsSolution = useSecondExtrinsicsSolution)

                    # 记录外参计算结果
                    if 'rotation' in CamParams:
                        import numpy as np  # 确保numpy在本地作用域内可用
                        rotation = CamParams['rotation']
                        translation = CamParams['translation']
                        logging.info(f"   ✅ {camName} 外参计算成功:")
                        logging.info(f"      旋转矩阵行列式: {np.linalg.det(rotation):.6f} (应该接近1)")
                        logging.info(f"      平移向量模长: {np.linalg.norm(translation):.3f} mm")
                        logging.info(f"      相机位置: [{translation[0][0]:.1f}, {translation[1][0]:.1f}, {translation[2][0]:.1f}] mm")

                except Exception as e:
                    if len(e.args) == 2: # specific exception
                        raise Exception(e.args[0], e.args[1])
                    elif len(e.args) == 1: # generic exception
                        exception = "Camera calibration failed. Verify your setup and try again. Visit https://www.opencap.ai/best-pratices to learn more about camera calibration and https://www.opencap.ai/troubleshooting for potential causes for a failed calibration."
                        raise Exception(exception, traceback.format_exc())
                loadedCamParams[camName] = False
                
       
            # Append camera parameters.
            if CamParams is not None:
                CamParamDict[camName] = CamParams.copy()
            else:
                CamParamDict[camName] = None

        # Save parameters if not existing yet.
        if not all([loadedCamParams[i] for i in loadedCamParams]):
            for camName in CamParamDict:
                saveCameraParameters(
                    os.path.join(cameraDirectories[camName],
                                 "cameraIntrinsicsExtrinsics.pickle"), 
                    CamParamDict[camName])
            
    # %% 3D reconstruction
    
    # Set output file name.
    pathOutputFiles = {}
    if benchmark:
        pathOutputFiles[trialName] = os.path.join(preAugmentationDir,
                                                  trialName + ".trc")
    else:
        pathOutputFiles[trialName] = os.path.join(preAugmentationDir,
                                                  trial_id + ".trc")
    
    # Trial relative path
    trialRelativePath = os.path.join('InputMedia', trialName, trial_id)
    
    if runPoseDetection:
        # Get rotation angles from motion capture environment to OpenSim.
        # Space-fixed are lowercase, Body-fixed are uppercase.
        checkerBoardMount = sessionMetadata['checkerBoard']['placement']
        logging.info(f"🎯 棋盘格放置方式: {checkerBoardMount}")
        logging.info("=" * 80)
        logging.info("🔄 开始坐标系转换配置分析")
        logging.info("=" * 80)

        if checkerBoardMount == 'backWall' or checkerBoardMount == 'Perpendicular':
            # 改进的棋盘格倒置检测
            logging.info("🔍 背墙放置模式，开始检测棋盘格朝向...")
            logging.info("   📋 背墙模式说明:")
            logging.info("      - 棋盘格垂直放置在墙上")
            logging.info("      - 需要检测棋盘格是否倒置")
            logging.info("      - 根据检测结果选择不同的坐标系转换")

            # 记录摄像头参数用于调试
            logging.info(f"   📷 可用摄像头数量: {len(CamParamDict)}")
            for cam_name in CamParamDict.keys():
                logging.info(f"      - {cam_name}: 外参已加载")



            upsideDownChecker = isCheckerboardUpsideDown(CamParamDict)
            logging.info(f"   🧭 棋盘格倒置检测结果: {upsideDownChecker}")


            if upsideDownChecker:
                rotationAngles = {'y':-90}
                logging.info("🔄 检测到棋盘格倒置，应用倒置补偿旋转:")
                logging.info("   y轴旋转: -90°")
                logging.info("   📝 说明: opencv y轴垂直向上,x轴超左,z轴超外 转向 x轴超前，z超左和Opensim 坐标轴要求对齐")
            else:
                rotationAngles = {'y':90, 'z':180}
                logging.info("🔄 检测到棋盘格正向，应用标准背墙旋转:")
                logging.info("   Y轴旋转: +90°")
                logging.info("   Z轴旋转: +180°")
                logging.info("   📝 说明: 从背墙坐标系转换到OpenSim坐标系")

        elif checkerBoardMount == 'ground' or checkerBoardMount == 'Lying':
            rotationAngles = {'x':90, 'y':90}
            logging.info("🔄 地面放置模式，应用地面旋转:")
            logging.info("   X轴旋转: +90°")
            logging.info("   Y轴旋转: +90°")
            logging.info("   📝 说明: 从地面坐标系转换到OpenSim坐标系")
        else:
            error_msg = f'棋盘格放置方式 "{checkerBoardMount}" 不受支持'
            logging.error(f"❌ {error_msg}")
            raise Exception(f'checkerBoard placement value "{checkerBoardMount}" in sessionMetadata.yaml is not currently supported')

        # 总结旋转设置
        logging.info("=" * 80)
        logging.info("📐 坐标系转换设置完成:")
        logging.info(f"   🎯 棋盘格放置: {checkerBoardMount}")
        if checkerBoardMount in ['backWall', 'Perpendicular']:
            logging.info(f"   🧭 倒置检测: {upsideDownChecker}")
        logging.info(f"   🔄 最终旋转角度: {rotationAngles}")
        logging.info("   📚 坐标系说明:")
        logging.info("      - 运动捕获坐标系: 基于棋盘格建立的原始坐标系")
        logging.info("      - OpenSim坐标系: Y轴向上((positive Y指向垂直向上，重力方向为负Y），X轴向前(positive Y指向垂直向上，重力方向为负Y）)，Z轴向右(positive Z指向受试者的右侧)")
        logging.info("      - 这些角度将用于从运动捕获坐标系转换到OpenSim坐标系")
        logging.info("=" * 80)

        # Detect all available cameras (ie, cameras with existing videos).
        cameras_available = []
        for camName in cameraDirectories:
            camDir = cameraDirectories[camName]
            pathVideoWithoutExtension = os.path.join(camDir, 'InputMedia', trialName, trial_id)
            if len(glob.glob(pathVideoWithoutExtension + '*')) == 0:
                print(f"Camera {camName} does not have a video for trial {trial_id}")
            else:
                if os.path.exists(os.path.join(pathVideoWithoutExtension + getVideoExtension(pathVideoWithoutExtension))):
                    cameras_available.append(camName)
                else:
                    print(f"Camera {camName} does not have a video for trial {trial_id}")

        if camerasToUse[0] == 'all':
            cameras_all = list(cameraDirectories.keys())
            if not all([cam in cameras_available for cam in cameras_all]):
                exception = 'Not all cameras have uploaded videos; one or more cameras might have turned off or lost connection'
                raise Exception(exception, exception)
            else:
                camerasToUse_c = camerasToUse
        elif camerasToUse[0] == 'all_available':
            camerasToUse_c = cameras_available
            print(f"Using available cameras: {camerasToUse_c}")
        else:
            if not all([cam in cameras_available for cam in camerasToUse]):
                raise Exception('Not all specified cameras in camerasToUse have videos; verify the camera names or consider setting camerasToUse to ["all_available"]')
            else:
                camerasToUse_c = camerasToUse
                print(f"Using cameras: {camerasToUse_c}")
        settings['camerasToUse'] = camerasToUse_c
        if camerasToUse_c[0] != 'all' and len(camerasToUse_c) < 2:
            exception = 'At least two videos are required for 3D reconstruction, video upload likely failed for one or more cameras.'
            raise Exception(exception, exception)
            
        # For neutral, we do not allow reprocessing with not all cameras.
        # The reason is that it affects extrinsics selection, and then you can only process
        # dynamic trials with the same camera selection (ie, potentially not all cameras). 
        # This might be addressable, but I (Antoine) do not see an immediate need + this
        # would be a significant change in the code base. In practice, a data collection
        # will not go through neutral if not all cameras are available.
        if scaleModel:
            if camerasToUse_c[0] != 'all' and len(camerasToUse_c) < len(cameraDirectories):
                exception = 'All cameras are required for calibration and neutral pose.'
                raise Exception(exception, exception)
        
        # Run pose detection algorithm.
        try:        
            videoExtension = runPoseDetector(
                    cameraDirectories, trialRelativePath, poseDetectorDirectory,
                    trialName, CamParamDict=CamParamDict, 
                    resolutionPoseDetection=resolutionPoseDetection, 
                    generateVideo=generateVideo, cams2Use=camerasToUse_c,
                    poseDetector=poseDetector, bbox_thr=bbox_thr)
            trialRelativePath += videoExtension
        except Exception as e:
            if len(e.args) == 2: # specific exception
                raise Exception(e.args[0], e.args[1])
            elif len(e.args) == 1: # generic exception
                exception = """Pose detection failed. Verify your setup and try again. 
                    Visit https://www.opencap.ai/best-pratices to learn more about data collection
                    and https://www.opencap.ai/troubleshooting for potential causes for a failed trial."""
                raise Exception(exception, traceback.format_exc())
      
    if runSynchronization:
        # Synchronize videos.
        try:
            keypoints2D, confidence, keypointNames, frameRate, nansInOut, startEndFrames, cameras2Use = (
                synchronizeVideos( 
                    cameraDirectories, trialRelativePath, poseDetectorDirectory,
                    undistortPoints=True, CamParamDict=CamParamDict,
                    filtFreqs=filtFreqs, confidenceThreshold=0.4,
                    imageBasedTracker=False, cams2Use=camerasToUse_c, 
                    poseDetector=poseDetector, trialName=trialName,
                    resolutionPoseDetection=resolutionPoseDetection))
        except Exception as e:
            if len(e.args) == 2: # specific exception
                raise Exception(e.args[0], e.args[1])
            elif len(e.args) == 1: # generic exception
                exception = """Video synchronization failed. Verify your setup and try again. 
                    A fail-safe synchronization method is for the participant to
                    quickly raise one hand above their shoulders, then bring it back down. 
                    Visit https://www.opencap.ai/best-pratices to learn more about 
                    data collection and https://www.opencap.ai/troubleshooting for 
                    potential causes for a failed trial."""
                raise Exception(exception, traceback.format_exc())
                
    # Note: this should not be necessary, because we prevent reprocessing the neutral trial
    # with not all cameras, but keeping it in there in case we would want to.
    if calibrationOptions is not None:
        allCams = list(calibrationOptions.keys())
        for cam_t in allCams:
            if not cam_t in cameras2Use:
                calibrationOptions.pop(cam_t)
                
    if scaleModel and calibrationOptions is not None and alternateExtrinsics is None:
        logging.info("🧠 正在触发自动外参解选择 autoSelectExtrinsicSolution ...")
        logging.info(f"   条件: scaleModel={scaleModel}, has calibrationOptions={calibrationOptions is not None}, alternateExtrinsics={alternateExtrinsics}")
        # Automatically select the camera calibration to use
        CamParamDict = autoSelectExtrinsicSolution(sessionDir,keypoints2D,confidence,calibrationOptions)
        # Report the chosen solutions if the JSON file exists
        try:
            calibSelPath = os.path.join(sessionDir, 'Videos', 'calibOptionSelections.json')
            if os.path.exists(calibSelPath):
                with open(calibSelPath, 'r', encoding='utf-8') as f:
                    chosen = json.load(f)
                logging.info("✅ 自动外参选择完成，选择结果如下(相机: 解索引):")
                for cam, sol in chosen.items():
                    logging.info(f"   - {cam}: soln{sol}")
            else:
                logging.info("ℹ️ 未找到 calibOptionSelections.json；自动选择已执行，但未写出选择文件。")
        except Exception as e:
            logging.warning(f"⚠️ 读取自动选择结果时出错: {str(e)}")
    else:
        logging.info("ℹ️ 跳过自动外参解选择：")
        logging.info(f"   条件: scaleModel={scaleModel}, has calibrationOptions={calibrationOptions is not None}, alternateExtrinsics={alternateExtrinsics}")
     
    if runTriangulation:
        # Triangulate.
        logging.info("=" * 80)
        logging.info("🔺 开始3D三角化重建")
        logging.info("=" * 80)
        logging.info(f"   📷 使用摄像头: {cameras2Use}")
        logging.info(f"   🎞️  帧率: {frameRate} fps")

        # 详细记录摄像头参数
        logging.info("   📷 摄像头外参详情:")
        for cam_name in cameras2Use:
            if cam_name in CamParamDict:
                cam_params = CamParamDict[cam_name]
                if 'rotation' in cam_params:
                    import numpy as np  # 确保numpy在本地作用域内可用
                    rotation = cam_params['rotation']
                    translation = cam_params['translation']
                    logging.info(f"      {cam_name}:")
                    logging.info(f"        旋转矩阵: {np.array2string(rotation.flatten()[:6], precision=3)}...")
                    logging.info(f"        平移向量: {np.array2string(translation.flatten(), precision=3)}")
                else:
                    logging.info(f"      {cam_name}: 外参格式未知")

        # 安全地获取2D关键点数据信息
        try:
            if 'keypoints2D' in locals() and keypoints2D is not None:
                if hasattr(keypoints2D, 'shape'):
                    logging.info(f"   📊 2D关键点数据形状: {keypoints2D.shape}")
                elif isinstance(keypoints2D, dict):
                    logging.info(f"   📊 2D关键点数据: {len(keypoints2D)} 个摄像头")
                    for cam_name, data in keypoints2D.items():
                        if hasattr(data, 'shape'):
                            logging.info(f"      {cam_name}: {data.shape}")
                            # 分析2D关键点的分布
                            if data.size > 0:
                                valid_points = data[~np.isnan(data)]
                                if len(valid_points) > 0:
                                    logging.info(f"        有效点范围: X[{np.min(valid_points):.1f}, {np.max(valid_points):.1f}]")
                else:
                    logging.info(f"   📊 2D关键点数据类型: {type(keypoints2D)}")
            else:
                logging.info(f"   📊 2D关键点数据: 未初始化")
        except Exception as e:
            logging.info(f"   📊 2D关键点数据: 获取信息时出错 - {str(e)}")

        try:
            keypoints3D, confidence3D = triangulateMultiviewVideo(
                CamParamDict, keypoints2D, ignoreMissingMarkers=False,
                cams2Use=cameras2Use, confidenceDict=confidence,
                spline3dZeros = True, splineMaxFrames=int(frameRate/5),
                nansInOut=nansInOut,CameraDirectories=cameraDirectories,
                trialName=trialName,startEndFrames=startEndFrames,trialID=trial_id,
                outputMediaFolder=outputMediaFolder)

            logging.info("✅ 3D三角化重建成功完成")
            logging.info(f"   📐 3D关键点数据形状: {keypoints3D.shape}")
            logging.info(f"   📊 置信度数据形状: {confidence3D.shape}")

            # 详细分析3D重建结果
            logging.info("   🔍 3D重建质量分析:")
            if keypoints3D.size > 0:
                # 分析各个轴的数据分布
                x_data = keypoints3D[0, :, :].flatten()
                y_data = keypoints3D[1, :, :].flatten()
                z_data = keypoints3D[2, :, :].flatten()

                valid_x = x_data[~np.isnan(x_data)]
                valid_y = y_data[~np.isnan(y_data)]
                valid_z = z_data[~np.isnan(z_data)]

                if len(valid_x) > 0:
                    logging.info(f"      X轴范围: [{np.min(valid_x):.3f}, {np.max(valid_x):.3f}] mm, 标准差: {np.std(valid_x):.3f}")
                if len(valid_y) > 0:
                    logging.info(f"      Y轴范围: [{np.min(valid_y):.3f}, {np.max(valid_y):.3f}] mm, 标准差: {np.std(valid_y):.3f}")
                if len(valid_z) > 0:
                    logging.info(f"      Z轴范围: [{np.min(valid_z):.3f}, {np.max(valid_z):.3f}] mm, 标准差: {np.std(valid_z):.3f}")

                # 快速比例检查：Z 轴范围是否远大于 X/Y（可提示外参或坐标系问题）
                try:
                    x_span = float(np.nanmax(valid_x) - np.nanmin(valid_x)) if len(valid_x) else np.nan
                    y_span = float(np.nanmax(valid_y) - np.nanmin(valid_y)) if len(valid_y) else np.nan
                    z_span = float(np.nanmax(valid_z) - np.nanmin(valid_z)) if len(valid_z) else np.nan
                    if np.isfinite(x_span) and np.isfinite(y_span) and np.isfinite(z_span):
                        xy_span = max(x_span, y_span, 1e-6)
                        ratio = z_span / xy_span
                        if ratio > 5:
                            logging.warning(f"      ⚠️ Z轴范围({z_span:.1f})显著大于XY({xy_span:.1f}), 比例≈{ratio:.1f}，可能存在外参/坐标系问题")
                except Exception:
                    pass

                # 计算人体尺度特征
                if keypoints3D.shape[2] > 0:
                    # 选择第一帧进行分析
                    frame_data = keypoints3D[:, :, 0]
                    valid_frame = frame_data[:, ~np.isnan(frame_data).any(axis=0)]

                    if valid_frame.shape[1] > 1:
                        # 计算点之间的距离分布
                        distances = []
                        for i in range(valid_frame.shape[1]):
                            for j in range(i+1, valid_frame.shape[1]):
                                dist = np.linalg.norm(valid_frame[:, i] - valid_frame[:, j])
                                distances.append(dist)

                        if distances:
                            logging.info(f"      点间距离: 平均 {np.mean(distances):.3f}mm, 最大 {np.max(distances):.3f}mm")

                            # 判断尺度是否合理（人体高度大概1000-2000mm）
                            max_dist = np.max(distances)
                            if max_dist < 500:
                                logging.warning(f"      ⚠️ 人体尺度可能过小，最大距离仅 {max_dist:.3f}mm")
                            elif max_dist > 5000:
                                logging.warning(f"      ⚠️ 人体尺度可能过大，最大距离达 {max_dist:.3f}mm")
                            else:
                                logging.info(f"      ✅ 人体尺度看起来合理")

        except Exception as e:
            logging.error("❌ 3D三角化重建失败")
            if len(e.args) == 2: # specific exception
                logging.error(f"   具体错误: {e.args[0]}")
                logging.error(e.args[0], exc_info=True)
                raise Exception(e.args[0], e.args[1])
            elif len(e.args) == 1: # generic exception
                exception = "Triangulation failed. Verify your setup and try again. Visit https://www.opencap.ai/best-pratices to learn more about data collection and https://www.opencap.ai/troubleshooting for potential causes for a failed trial."
                logging.error(f"   通用错误: {exception}")
                logging.error(exception, exc_info=True)
                raise Exception(exception, traceback.format_exc())

        # Throw an error if not enough data
        valid_frames = keypoints3D.shape[2]
        logging.info(f"   ✅ 有效3D数据帧数: {valid_frames}")

        if valid_frames < 10:
            error_msg = f'错误 - 有效的3D数据帧数少于10帧 (当前: {valid_frames}帧)'
            logging.error(f"❌ {error_msg}")
            logging.error("   可能原因:")
            logging.error("   - 2D姿态检测质量差")
            logging.error("   - 摄像头标定不准确")
            logging.error("   - 视频同步失败")
            logging.error("   - 被试人员在摄像头视野范围外")
            raise Exception(error_msg, error_msg)

        # Write TRC.
        logging.info("=" * 80)
        logging.info("📝 开始写入TRC文件")
        logging.info("=" * 80)
        logging.info(f"   📁 输出文件: {pathOutputFiles[trialName]}")
        logging.info(f"   🏷️  关键点名称数量: {len(keypointNames)}")
        logging.info(f"   🔄 应用的旋转角度: {rotationAngles}")
        logging.info(f"   🎞️  帧率: {frameRate} fps")

        # 记录3D数据的统计信息
        import numpy as np
        logging.info("   📊 3D数据统计:")
        logging.info(f"      数据形状: {keypoints3D.shape}")
        logging.info(f"      最小值: {np.nanmin(keypoints3D):.3f}")
        logging.info(f"      最大值: {np.nanmax(keypoints3D):.3f}")
        logging.info(f"      平均值: {np.nanmean(keypoints3D):.3f}")
        logging.info(f"      NaN比例: {np.isnan(keypoints3D).sum() / keypoints3D.size * 100:.1f}%")

        # 分析坐标系转换前的数据特征
        logging.info("   🔄 坐标系转换前数据分析:")
        if keypoints3D.shape[2] > 0:
            first_frame = keypoints3D[:, :, 0]
            valid_points = first_frame[:, ~np.isnan(first_frame).any(axis=0)]

            if valid_points.shape[1] > 0:
                logging.info(f"      转换前坐标系特征:")
                logging.info(f"        X轴 (转换前): [{np.min(valid_points[0, :]):.1f}, {np.max(valid_points[0, :]):.1f}] mm")
                logging.info(f"        Y轴 (转换前): [{np.min(valid_points[1, :]):.1f}, {np.max(valid_points[1, :]):.1f}] mm")
                logging.info(f"        Z轴 (转换前): [{np.min(valid_points[2, :]):.1f}, {np.max(valid_points[2, :]):.1f}] mm")

                # 计算重心位置
                centroid = np.mean(valid_points, axis=1)
                logging.info(f"        重心位置: [{centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}] mm")

                # 分析坐标系方向性
                y_spread = np.max(valid_points[1, :]) - np.min(valid_points[1, :])
                x_spread = np.max(valid_points[0, :]) - np.min(valid_points[0, :])
                z_spread = np.max(valid_points[2, :]) - np.min(valid_points[2, :])
                logging.info(f"        各轴分布范围: X={x_spread:.1f}, Y={y_spread:.1f}, Z={z_spread:.1f} mm")

                # 推断可能的坐标系问题
                if abs(centroid[1]) > 1000:  # Y轴重心偏离过大
                    logging.warning(f"      ⚠️ Y轴重心位置异常: {centroid[1]:.1f}mm，可能存在坐标系转换问题")

                # 检查是否存在明显的方向性错误
                if y_spread < x_spread and y_spread < z_spread:
                    logging.warning(f"      ⚠️ Y轴分布范围最小，可能不是垂直轴，检查坐标系设置")

        writeTRCfrom3DKeypoints(keypoints3D, pathOutputFiles[trialName],
                                keypointNames, frameRate=frameRate,
                                rotationAngles=rotationAngles)

        # 额外导出调试用3D采样JSON，便于快速人工检查
        try:
            from utilsChecker import save3DPointsDebug
            debug_dir = os.path.join(preAugmentationDir, 'Debug3D')
            os.makedirs(debug_dir, exist_ok=True)
            debug_json_path = os.path.join(debug_dir, f"{trial_id}_3d_sample.json")
            save3DPointsDebug(keypoints3D, keypointNames, frameRate, debug_json_path,
                              sample_strategy='auto', max_frames=10, rotationAngles=rotationAngles)
            logging.info(f"   🧪 已导出3D调试JSON: {debug_json_path}")
        except Exception as e:
            logging.warning(f"   ⚠️ 导出3D调试JSON失败: {str(e)}")

        logging.info("✅ TRC文件写入完成")
        logging.info("   📝 说明: TRC文件包含了经过坐标系转换的3D标记点数据")
        logging.info("   🔄 坐标系: 已从运动捕获坐标系转换为OpenSim坐标系")
        logging.info("=" * 80)
    
    # %% Augmentation.
    
    # Get augmenter model.
    augmenterModelName = (
        sessionMetadata['markerAugmentationSettings']['markerAugmenterModel'])
    
    # Set output file name.
    pathAugmentedOutputFiles = {}
    if genericFolderNames:
        pathAugmentedOutputFiles[trialName] = os.path.join(
                postAugmentationDir, trial_id + ".trc")
    else:
        if benchmark:
            pathAugmentedOutputFiles[trialName] = os.path.join(
                    postAugmentationDir, trialName + "_" + augmenterModelName +".trc")
        else:
            pathAugmentedOutputFiles[trialName] = os.path.join(
                    postAugmentationDir, trial_id + "_" + augmenterModelName +".trc")
    
    if runMarkerAugmentation:
        os.makedirs(postAugmentationDir, exist_ok=True)    
        augmenterDir = os.path.join(baseDir, "MarkerAugmenter")
        logging.info('Augmenting marker set')
        try:
            vertical_offset = augmentTRC(
                pathOutputFiles[trialName],sessionMetadata['mass_kg'], 
                sessionMetadata['height_m'], pathAugmentedOutputFiles[trialName],
                augmenterDir, augmenterModelName=augmenterModelName,
                augmenter_model=augmenterModel, offset=offset)
        except Exception as e:
            if len(e.args) == 2: # specific exception
                raise Exception(e.args[0], e.args[1])
            elif len(e.args) == 1: # generic exception
                exception = "Marker augmentation failed. Verify your setup and try again. Visit https://www.opencap.ai/best-pratices to learn more about data collection and https://www.opencap.ai/troubleshooting for potential causes for a failed trial."
                raise Exception(exception, traceback.format_exc())
        if offset:
            # If offset, no need to offset again for the webapp visualization.
            # (0.01 so that there is no overall offset, see utilsOpenSim).
            vertical_offset_settings = float(np.copy(vertical_offset)-0.01)
            vertical_offset = 0.01   
        
    # %% OpenSim pipeline.
    if runOpenSimPipeline:
        logging.info("=" * 80)
        logging.info("🦴 开始OpenSim生物力学分析管道")
        logging.info("=" * 80)

        openSimPipelineDir = os.path.join(baseDir, "opensimPipeline")

        if genericFolderNames:
            openSimFolderName = 'OpenSimData'
        else:
            openSimFolderName = os.path.join('OpenSimData',
                                             poseDetector + suff_pd)
            if not markerDataFolderNameSuffix is None:
                openSimFolderName = os.path.join(openSimFolderName,
                                                 markerDataFolderNameSuffix)

        openSimDir = os.path.join(sessionDir, openSimFolderName)
        outputScaledModelDir = os.path.join(openSimDir, 'Model')

        # Check if shoulder model.
        if 'shoulder' in sessionMetadata['openSimModel']:
            suffix_model = '_shoulder'
        else:
            suffix_model = ''

        logging.info(f"   📁 OpenSim目录: {openSimDir}")
        logging.info(f"   🏗️  基础模型: {sessionMetadata['openSimModel']}{suffix_model}")
        logging.info(f"   👤 受试者信息: 体重{sessionMetadata['mass_kg']}kg, 身高{sessionMetadata['height_m']}m")
        logging.info(f"   🔧 是否缩放模型: {scaleModel}")

        # Scaling.
        if scaleModel:
            logging.info("=" * 60)
            logging.info("📏 开始模型缩放（静态试验）")
            logging.info("=" * 60)

            os.makedirs(outputScaledModelDir, exist_ok=True)
            # Path setup file.
            if scalingSetup == 'any_pose':
                genericSetupFile4ScalingName = 'Setup_scaling_LaiUhlrich2022_any_pose.xml'
                logging.info("   🧘 使用任意姿态缩放设置")
            else: # by default, use upright_standing_pose
                genericSetupFile4ScalingName = 'Setup_scaling_LaiUhlrich2022.xml'
                logging.info("   🧍 使用直立站姿缩放设置")

            pathGenericSetupFile4Scaling = os.path.join(
                openSimPipelineDir, 'Scaling', genericSetupFile4ScalingName)
            # Path model file.
            pathGenericModel4Scaling = os.path.join(
                openSimPipelineDir, 'Models',
                sessionMetadata['openSimModel'] + '.osim')
            # Path TRC file.
            pathTRCFile4Scaling = pathAugmentedOutputFiles[trialName]

            logging.info(f"   📋 缩放设置文件: {genericSetupFile4ScalingName}")
            logging.info(f"   🏗️  通用模型文件: {os.path.basename(pathGenericModel4Scaling)}")
            logging.info(f"   📊 TRC数据文件: {os.path.basename(pathTRCFile4Scaling)}")

            # Get time range.
            try:
                logging.info("   🎯 开始识别缩放时间范围...")
                thresholdPosition = 0.003
                maxThreshold = 0.015
                increment = 0.001
                success = False
                timeRange4Scaling = None  # 初始化变量
                attempt_count = 0

                while thresholdPosition <= maxThreshold and not success:
                    attempt_count += 1
                    try:
                        logging.info(f"      尝试 #{attempt_count}: 位置阈值 = {thresholdPosition:.3f}")
                        timeRange4Scaling = getScaleTimeRange(
                            pathTRCFile4Scaling,
                            thresholdPosition=thresholdPosition,
                            thresholdTime=0.1, removeRoot=True)
                        success = True
                        logging.info(f"   ✅ 成功识别缩放时间范围: {timeRange4Scaling}")
                    except Exception as e:
                        logging.info(f"      ❌ 失败: {str(e)}")
                        thresholdPosition += increment  # Increase the threshold for the next iteration

                # 检查是否成功找到时间范围
                if not success or timeRange4Scaling is None:
                    error_msg = f"无法在尝试{attempt_count}次后找到合适的缩放时间范围"
                    logging.error(f"   ❌ {error_msg}")
                    logging.error("   可能原因:")
                    logging.error("      - 静态姿态数据质量差")
                    logging.error("      - 受试者在静态试验中移动太多")
                    logging.error("      - TRC文件中缺少足够的稳定数据")
                    raise Exception(error_msg)

                # Run scale tool.
                logging.info("   🚀 开始运行模型缩放工具...")
                logging.info(f"      时间范围: {timeRange4Scaling[0]:.3f}s - {timeRange4Scaling[1]:.3f}s")
                logging.info(f"      持续时间: {timeRange4Scaling[1] - timeRange4Scaling[0]:.3f}s")

                pathScaledModel = runScaleTool(
                    pathGenericSetupFile4Scaling, pathGenericModel4Scaling,
                    sessionMetadata['mass_kg'], pathTRCFile4Scaling,
                    timeRange4Scaling, outputScaledModelDir,
                    subjectHeight=sessionMetadata['height_m'],
                    suffix_model=suffix_model)

                logging.info(f"   ✅ 模型缩放成功完成")
                logging.info(f"      缩放后模型: {os.path.basename(pathScaledModel)}")

            except Exception as e:
                logging.error("❌ 模型缩放失败")
                if len(e.args) == 2: # specific exception
                    logging.error(f"   具体错误: {e.args[0]}")
                    raise Exception(e.args[0], e.args[1])
                elif len(e.args) == 1: # generic exception
                    exception = "Musculoskeletal model scaling failed. Verify your setup and try again. Visit https://www.opencap.ai/best-pratices to learn more about data collection and https://www.opencap.ai/troubleshooting for potential causes for a failed neutral pose."
                    logging.error(f"   通用错误: {exception}")
                    raise Exception(exception, traceback.format_exc())

            # Extract one frame from videos to verify neutral pose.
            logging.info("   📸 提取静态姿态图像用于验证...")
            staticImagesFolderDir = os.path.join(sessionDir,
                                                 'NeutralPoseImages')
            os.makedirs(staticImagesFolderDir, exist_ok=True)
            popNeutralPoseImages(cameraDirectories, cameras2Use,
                                 timeRange4Scaling[0], staticImagesFolderDir,
                                 trial_id, writeVideo = True)
            logging.info(f"      验证图像保存到: {staticImagesFolderDir}")

            pathOutputIK = pathScaledModel[:-5] + '.mot'
            pathModelIK = pathScaledModel

            logging.info("   📝 注意: 缩放后的模型将用于后续的逆运动学分析")

        # Inverse kinematics.
        if not scaleModel:
            logging.info("=" * 60)
            logging.info("🏃 开始逆运动学分析（动态试验）")
            logging.info("=" * 60)

            outputIKDir = os.path.join(openSimDir, 'Kinematics')
            os.makedirs(outputIKDir, exist_ok=True)
            # Check if there is a scaled model.
            pathScaledModel = os.path.join(outputScaledModelDir,
                                            sessionMetadata['openSimModel'] +
                                            "_scaled.osim")

            logging.info(f"   📂 IK输出目录: {outputIKDir}")
            logging.info(f"   🔍 查找缩放后模型: {os.path.basename(pathScaledModel)}")

            if os.path.exists(pathScaledModel):
                logging.info("   ✅ 找到缩放后的模型")

                # Path setup file.
                genericSetupFile4IKName = 'Setup_IK{}.xml'.format(suffix_model)
                pathGenericSetupFile4IK = os.path.join(
                    openSimPipelineDir, 'IK', genericSetupFile4IKName)
                # Path TRC file.
                pathTRCFile4IK = pathAugmentedOutputFiles[trialName]

                logging.info(f"   📋 IK设置文件: {genericSetupFile4IKName}")
                logging.info(f"   📊 TRC数据文件: {os.path.basename(pathTRCFile4IK)}")

                # Run IK tool.
                logging.info('   🚀 开始运行逆运动学工具...')
                try:
                    pathOutputIK, pathModelIK = runIKTool(
                        pathGenericSetupFile4IK, pathScaledModel,
                        pathTRCFile4IK, outputIKDir)

                    logging.info("   ✅ 逆运动学分析成功完成")
                    logging.info(f"      输出MOT文件: {os.path.basename(pathOutputIK)}")
                    logging.info("      📝 说明: MOT文件包含关节角度随时间的变化")

                except Exception as e:
                    logging.error("❌ 逆运动学分析失败")
                    if len(e.args) == 2: # specific exception
                        logging.error(f"   具体错误: {e.args[0]}")
                        raise Exception(e.args[0], e.args[1])
                    elif len(e.args) == 1: # generic exception
                        exception = "Inverse kinematics failed. Verify your setup and try again. Visit https://www.opencap.ai/best-pratices to learn more about data collection and https://www.opencap.ai/troubleshooting for potential causes for a failed trial."
                        logging.error(f"   通用错误: {exception}")
                        raise Exception(exception, traceback.format_exc())
            else:
                error_msg = "未找到缩放后的模型，请先运行静态试验进行模型缩放"
                logging.error(f"   ❌ {error_msg}")
                logging.error(f"   期望路径: {pathScaledModel}")
                raise ValueError(error_msg)

        # Write body transforms to json for visualization.
        logging.info("=" * 60)
        logging.info("📊 生成可视化数据")
        logging.info("=" * 60)

        outputJsonVisDir = os.path.join(sessionDir,'VisualizerJsons',
                                        trialName)
        os.makedirs(outputJsonVisDir,exist_ok=True)
        outputJsonVisPath = os.path.join(outputJsonVisDir,
                                         trialName + '.json')

        logging.info(f"   📁 可视化目录: {outputJsonVisDir}")
        logging.info(f"   📄 JSON文件: {os.path.basename(outputJsonVisPath)}")
        logging.info(f"   📏 垂直偏移: {vertical_offset}")

        generateVisualizerJson(pathModelIK, pathOutputIK,
                               outputJsonVisPath,
                               vertical_offset=vertical_offset,
                               roundToRotations=4, roundToTranslations=4)

        logging.info("   ✅ 可视化数据生成完成")
        logging.info("=" * 80)
        logging.info("🎉 OpenSim生物力学分析管道完成")
        logging.info("=" * 80)
        
    # %% Rewrite settings, adding offset  
    if not extrinsicsTrial:
        if offset:
            settings['verticalOffset'] = vertical_offset_settings 
        with open(pathSettings, 'w', encoding='utf-8') as file:
            yaml.dump(settings, file)
    
    # 返回成功状态
    return True
