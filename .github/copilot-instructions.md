# OpenCap Core AI Coding Instructions

## Overview
OpenCap Core is a computer vision and biomechanics pipeline that processes multi-camera videos to estimate 3D human movement kinematics. The system integrates pose detection, camera calibration, 3D triangulation, marker augmentation via LSTM models, and OpenSim biomechanical analysis.

## Architecture & Data Flow

### Pipeline Stages (Sequential)
1. **Camera Calibration** (`utilsChecker.py`) - Checkerboard detection → intrinsic/extrinsic parameters
2. **Pose Detection** (`utilsDetector.py`) - 2D keypoints via OpenPose/MMPose  
3. **3D Reconstruction** (`utilsChecker.py`) - Synchronized multi-view triangulation
4. **Marker Augmentation** (`utilsAugmenter.py`) - LSTM expansion to full marker set
5. **Biomechanical Analysis** (`utilsOpenSim.py`) - Model scaling and inverse kinematics

### Entry Points by Use Case
- **Cloud Processing**: `app.py` (Docker worker) + `utilsServer.py`
- **Local Processing**: `main.py` (core pipeline) or `local_opencap_pipeline.py` (offline wrapper)
- **Camera Calibration Only**: `main_calcIntrinsics.py` or `main_calcIntrinsics_local.py`

### Critical Configuration Pattern
```python
# Environment-based mode switching (set before imports)
os.environ['OPENCAP_LOCAL_MODE'] = 'true'  # Skip API authentication
os.environ['PYTHONIOENCODING'] = 'utf-8'   # Windows encoding fix
```

## Key Coding Patterns

### Trial Type Handling
```python
# main.py uses extrinsicsTrial flag to control processing stages
if extrinsicsTrial:
    runPoseDetection = False  # Calibration only
    runOpenSimPipeline = False
elif trial_type == 'static':
    scaleModel = True  # Generate scaled OpenSim model
else:  # dynamic
    scaleModel = False  # Use existing scaled model
```

### Data Organization (Critical Structure)
```
Data/SessionName/
├── Videos/CameraName/InputMedia/TrialName/trial_id.mp4
├── MarkerData[_variant]/[Pre|Post]Augmentation/
├── OpenSimData/Model/*_scaled.osim
└── sessionMetadata.yaml
```

### Utils Module Boundaries
- `utilsChecker.py` - Camera operations, synchronization, triangulation (3000+ lines)
- `utilsDetector.py` - Pose detection abstraction layer (OpenPose/MMPose)
- `utilsAugmenter.py` - LSTM model loading and inference
- `utilsOpenSim.py` - Biomechanical modeling integration
- `utilsServer.py` - Cloud processing workflow orchestration

### Error Handling Pattern
```python
# Graceful degradation for missing components
try:
    from main import main as opencap_main
except Exception as e:
    logger.error(f"Failed to import main: {str(e)}")
    return False
```

## Development Workflows

### Local Testing
```bash
# Setup (requires CUDA GPU)
conda create -n opencap python=3.9
conda activate opencap
conda install -c opensim-org opensim=4.4=py39np120
pip install -r requirements.txt

# Run local pipeline
python examples_local_usage.py
python local_opencap_pipeline.py /path/to/videos --calibration-dir /path/to/calib
```

### Camera Calibration Debugging
```python
# Test intrinsics loading
from tests.test_read_camera_intrinsics import read_camera_intrinsics
intrinsics = read_camera_intrinsics("CameraIntrinsics/iPhone17,5/Deployed_720_60fps/cameraIntrinsics.pickle")
```

### Model Versioning
- LSTM models in `MarkerAugmenter/LSTM/v0.X/` with `metadata.json` + `model.json`
- Camera intrinsics in `CameraIntrinsics/DeviceModel/DeployedFolder/cameraIntrinsics.pickle`
- OpenSim models auto-generated in `OpenSimData/Model/*_scaled.osim`

## Critical Dependencies & Integration

### External Tools (Must be in PATH)
- **OpenPose**: `C:\openpose\bin` - 2D pose detection
- **FFmpeg**: `C:\ffmpeg\bin` - Video processing  
- **CUDA/cuDNN**: Required for TensorFlow GPU acceleration

### Processing Parameters
```python
# Key settings that affect output quality vs performance
poseDetector = 'OpenPose' | 'mmpose'  # Detection algorithm
resolutionPoseDetection = 'default' | '1x736' | '2x736'  # Accuracy vs speed
augmenter_model = 'v0.0' | 'v0.1' | 'v0.2' | 'v0.3'  # LSTM model version
imageUpsampleFactor = 4  # Calibration image quality
```

### File Format Conventions
- **Videos**: `.mp4` (rotated/unrotated variants as `_rotated.avi`)
- **Pose Data**: `.pkl` (pickled numpy arrays) → `.trc` (OpenSim format)
- **Camera Params**: `.pickle` (OpenCV calibration results)
- **Config**: `.yaml` (session metadata) + `.json` (processing defaults)

## Local vs Cloud Patterns

### Authentication Bypass
```python
# Local mode skips all API calls
if os.environ.get('OPENCAP_LOCAL_MODE') != 'true':
    token = getToken()  # Only call in cloud mode
```

### Data Flow Differences
- **Cloud**: Downloads from API → processes → uploads results
- **Local**: Reads local files → processes → saves locally (no uploads)

Use `local_opencap_pipeline.py` for fully offline processing, `main.py` for cloud-integrated workflows.