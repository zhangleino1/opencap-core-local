# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
OpenCap Core is a Python-based motion capture pipeline that processes videos to estimate 3D human movement kinematics using computer vision and biomechanical modeling. The system integrates OpenPose/MMPose for pose detection, camera calibration, 3D triangulation, marker augmentation using LSTM models, and OpenSim for biomechanical analysis.

## Core Architecture

### Main Entry Points
- `main.py` - Primary pipeline function that orchestrates the entire motion capture process
- `app.py` - Server application for cloud processing of trials (Docker-based worker)
- `main_calcIntrinsics.py` - Standalone camera intrinsics calibration

### Key Processing Modules
- `utils*.py` - Modular utilities organized by function:
  - `utilsChecker.py` - Camera calibration, synchronization, triangulation (largest module)
  - `utilsDetector.py` - Pose detection (OpenPose/MMPose integration)
  - `utilsAugmenter.py` - LSTM-based marker set augmentation
  - `utilsOpenSim.py` - OpenSim pipeline integration (scaling, IK)
  - `utilsCameraPy3.py` - Camera parameter handling
  - `utilsServer.py` - Server-side trial processing
  - `utilsAPI.py` - API configuration and helpers
  - `utilsAuth.py` - Authentication utilities

### Data Flow
1. **Camera Calibration** - Checkerboard detection and intrinsic/extrinsic parameter estimation
2. **Pose Detection** - 2D keypoint detection using OpenPose or MMPose
3. **3D Reconstruction** - Triangulation of synchronized multi-view 2D poses
4. **Marker Augmentation** - LSTM models expand limited keypoints to full marker set
5. **Biomechanical Analysis** - OpenSim scaling and inverse kinematics

### Model Architecture
- LSTM models in `MarkerAugmenter/LSTM/` with versioned models (v0.0 to v0.3)
- Separate upper/lower body models for different anatomical regions
- Models stored with metadata.json and model.json configurations

## Development Commands

### Environment Setup
```bash
conda create -n opencap python=3.9 pip
conda activate opencap
conda install -c opensim-org opensim=4.4=py39np120
python -m pip install -r requirements.txt
```

### Testing
```bash
pytest tests/
```

### Dependencies
- TensorFlow GPU 2.9.3 for LSTM inference
- OpenCV 4.5.1.48 for computer vision operations
- OpenSim 4.4 conda package for biomechanical modeling
- External dependencies: OpenPose (C:\openpose\bin), FFmpeg (C:\ffmpeg\bin)

### Processing Examples
- `Examples/reprocessSession.py` - Reprocess existing sessions with different settings
- `Examples/createAuthenticationEnvFile.py` - Set up authentication for cloud API
- `Examples/batchDownloadData.py` - Batch download multiple sessions

## Key Configuration Files
- `defaultOpenCapSettings.json` - Default processing parameters
- `defaultSessionMetadata.yaml` - Session metadata template
- `requirements.txt` - Python dependencies
- Docker configuration in `docker/` directory

## Important Processing Parameters
- `poseDetector` - 'OpenPose' or 'MMPose'
- `resolutionPoseDetection` - Controls pose detection accuracy vs speed
- `augmenter_model` - LSTM model version (v0.0 to v0.3)
- `imageUpsampleFactor` - Calibration image upsampling (default: 4)
- `cameras_to_use` - Subset of cameras for processing

## Data Directories
- Input videos and calibration images organized by session/trial structure
- Camera intrinsics stored in `CameraIntrinsics/` with model-specific subfolders
- Processed outputs include TRC files, OpenSim models, and visualization data

## Cloud Integration
The codebase supports both local processing and cloud deployment via app.opencap.ai, with authentication and API integration for seamless data synchronization.