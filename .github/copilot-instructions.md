# OpenCap Core — AI Coding Instructions

## Overview
OpenCap Core is a computer vision and biomechanics pipeline for estimating 3D human movement kinematics from multi-camera videos. The system integrates pose detection (OpenPose/MMPose), camera calibration, 3D triangulation, LSTM-based marker augmentation, and OpenSim biomechanical analysis.

## Architecture & Pipeline Flow

### Sequential Processing Stages
1. **Camera Calibration** - Checkerboard detection → intrinsic/extrinsic parameters (`utilsChecker.py`)
2. **Pose Detection** - 2D keypoints via OpenPose/MMPose (`utilsDetector.py`, `utilsMMpose.py`)
3. **Synchronization & 3D Reconstruction** - Multi-view triangulation (`utilsChecker.py`)
4. **Marker Augmentation** - LSTM expansion to full marker set (`utilsAugmenter.py`)
5. **Biomechanical Analysis** - OpenSim model scaling and inverse kinematics (`utilsOpenSim.py`)

### Entry Points by Use Case
- **Cloud Processing**: `app.py` (Docker worker pulling from queue) + `utilsServer.processTrial()`
- **Local Processing**: `main.py` (orchestrator) or `local_opencap_pipeline.py` (fully offline wrapper)
- **Camera Calibration Only**: `main_calcIntrinsics.py` or `main_calcIntrinsics_local.py`

### Critical Configuration Pattern
```python
# Set BEFORE any imports to bypass API authentication in local mode
os.environ['OPENCAP_LOCAL_MODE'] = 'true'
os.environ['PYTHONIOENCODING'] = 'utf-8'  # Windows encoding fix
```

## Data Organization (Strict Structure)

### Required Directory Layout
```
Data/<SessionName>/
├── Videos/<CameraName>/InputMedia/<TrialName>/<trial_id>.mp4
├── CalibrationImages/extrinsicCalib_*.jpg  # Generated during calibration
├── MarkerData[_variant]/[Pre|Post]Augmentation/*.trc
├── OpenSimData/Model/*_scaled.osim
└── sessionMetadata.yaml  # Controls processing behavior
```

**Critical**: Many utilities assume this exact structure. Session metadata YAML defines trial types, checkerboard config, subject mass/height.

### File Types & Conventions
- Videos: `.mp4` or `.avi`
- Pose data: `.pkl` per camera (2D keypoints + confidence)
- 3D markers: `.trc` files in meters (OpenSim format)
- Camera params: `.pickle` files with intrinsics/extrinsics
- Trial types: `calibration` (extrinsics only), `static`/`neutral` (model scaling), `dynamic` (IK)

## Key Coding Patterns

### Trial Type Gating in main.py
```python
if extrinsicsTrial:  # Calibration-only trial
    runPoseDetection = False
    runOpenSimPipeline = False
elif trial_type == 'static' or scaleModel:  # Generate scaled model
    scaleModel = True
else:  # dynamic trial - use existing scaled model
    scaleModel = False
```

### Coordinate System Transformations
**Critical for OpenSim alignment**: Checkerboard defines world coordinate system, which must be rotated to OpenSim's Y-up convention.

```python
# main.py determines rotation based on checkerboard placement
if checkerBoardMount == 'backWall':
    upsideDown = isCheckerboardUpsideDown(CamParamDict)
    rotationAngles = {'y': -90} if upsideDown else {'y': 90, 'z': 180}
elif checkerBoardMount == 'ground':
    rotationAngles = {'x': 90, 'y': 90}

# Applied in writeTRCfrom3DKeypoints() when writing TRC files
```

**Two-solution extrinsics**: `cv2.solvePnP` produces ambiguous solutions; both saved as `cameraIntrinsicsExtrinsics_soln0.pickle` and `soln1.pickle`. Selection via `autoSelectExtrinsicSolution()` or manual review of `extrinsicCalib_*.jpg` images.

### Pose Detector Switching
```python
poseDetector = 'OpenPose'  # or 'mmpose'
if poseDetector == 'OpenPose':
    resolutionPoseDetection = '1x736'  # default, 1x736_2scales, 1x1008_4scales
    outputMediaFolder = 'OutputMedia_' + resolutionPoseDetection
elif poseDetector == 'mmpose':
    bbox_thr = 0.8  # bounding box threshold
    outputMediaFolder = 'OutputMedia_mmpose' + str(bbox_thr)
```

**GPU Memory Requirements**:
- `default`: 4GB
- `1x736` or `1x736_2scales`: 4-8GB
- `1x1008_4scales`: 24GB

## Development Workflows

### Environment Setup (Windows/Conda)
```bash
conda create -n opencap python=3.9 pip
conda activate opencap
conda install -c opensim-org opensim=4.4=py39np120
python -m pip install -r requirements.txt
```

External dependencies (add to PATH):
- OpenPose: `C:\openpose\bin`
- FFmpeg: `C:\ffmpeg\bin`

### Testing & Examples
```bash
# Run test suite
pytest tests/

# Local processing examples
python examples_local_usage.py
python local_opencap_pipeline.py ./videos --calibration-dir ./calib

# Reprocess cloud session
python Examples/reprocessSession.py  # Set session_id from app.opencap.ai
```

### Docker Deployment
See `docker/README.md`. Worker pulls trials from queue at `{API_URL}/trials/dequeue/?workerType={all|calibration}`.

## Implementation Guidelines

### When Editing Calibration Logic
- Preserve dual-solution artifacts (`soln0/soln1.pickle` + images) - other tools depend on them
- Respect `isCheckerboardUpsideDown()` detection for coordinate consistency
- Keep rotation angle logic in sync with `sessionMetadata['checkerBoard']['placement']`

### When Changing Pose Detection
- Check GPU memory availability before increasing resolution
- OpenPose paths resolved via `utils.getOpenPoseDirectory()`
- MMPose config lives in `mmpose/` directory

### When Debugging Coordinate Issues
- Inspect TRC file headers and rotation angles in logs
- Expect Y-axis dominant post-rotation (OpenSim convention)
- Verify checkerboard placement setting matches physical setup (see `坐标系转换说明.md`)

### Cloud vs Local Mode
- Cloud API calls in `utils.py`: `postFileToTrial()`, `getCalibration*()`, `postCalibration*()`
- Set `OPENCAP_LOCAL_MODE='true'` to bypass all API calls during local dev
- Local mode uses `local_opencap_pipeline.py` with self-contained session management

## Module Responsibilities

| Module | Primary Functions |
|--------|------------------|
| `utilsChecker.py` | Calibration, sync, triangulation, video I/O (largest module ~3272 lines) |
| `utilsDetector.py`/`utilsMMpose.py` | Pose detector abstraction and configuration |
| `utilsAugmenter.py` | LSTM model loading/inference (models in `MarkerAugmenter/LSTM/v0.*`) |
| `utilsOpenSim.py` | Scaling (`runScaleTool`), IK (`runIKTool`), visualizer JSON |
| `utils.py` | TRC writers (`numpy2TRC`), API helpers, session metadata I/O |
| `utilsServer.py` | Cloud worker trial processing (`processTrial()`, `runTestSession()`) |
| `app.py` | Docker worker queue polling and orchestration |

## Common Pitfalls

1. **Modifying data layout**: Utilities hardcode paths like `Videos/CamX/InputMedia/Trial/` - changes break many functions
2. **Skipping OPENCAP_LOCAL_MODE**: Causes API token errors in offline environments
3. **Ignoring trial type logic**: Calibration trials need `extrinsicsTrial=True` to skip pose detection
4. **Wrong rotation angles**: Mismatched checkerboard placement causes inverted/lying figures in OpenSim
5. **Missing external dependencies**: OpenPose/FFmpeg must be in PATH or specified via `getOpenPoseDirectory()`
