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
````instructions
# OpenCap Core — AI Coding Guide

Purpose: Enable AI agents to work productively in this Python CV+biomech pipeline by codifying project-specific patterns, entry points, data layout, and workflows.

## Architecture & Flow
- Stages: calibration → 2D pose → sync/triangulate → augment → OpenSim.
- Entry points:
    - `main.py`: orchestrates end-to-end local/cloud-aware runs.
    - `local_opencap_pipeline.py`: fully offline wrapper and examples.
    - `app.py` + `utilsServer.py`: cloud worker orchestration.
    - Intrinsics-only: `main_calcIntrinsics.py` (+ `_local.py`).
- Major modules:
    - `utilsChecker.py`: calibration (intrinsics/extrinsics), sync, triangulation, pkl/video I/O.
    - `utilsDetector.py`/`utilsMMpose.py`: OpenPose/MMPose abstraction and settings.
    - `utilsAugmenter.py`: LSTM marker augmentation (models in `MarkerAugmenter/LSTM/v0.*`).
    - `utilsOpenSim.py`: scaling (neutral), IK (dynamic), visualizer JSON.
    - `utils.py`: API helpers, session I/O, TRC writers (`numpy2TRC`), downloads.

## Data & Conventions
- Session tree (critical): `Data/<Session>/Videos/CamX/InputMedia/<Trial>/<trial_id>.<ext>`; results in `MarkerData/`, `OpenSimData/`, `sessionMetadata.yaml`.
- File types: videos `.mp4/.avi`; pose `.pkl` per cam; 3D markers `.trc` (meters); camera params `.pickle`.
- Trial types: `calibration` (extrinsics), `neutral` (scale model), others dynamic (IK only).
- Pose detector switch: `poseDetector = 'OpenPose' | 'mmpose'`; OpenPose resolution via `resolutionPoseDetection`.

## Calibration & Triangulation
- Extrinsics ambiguity: two PnP solutions per cam are saved; selection can be automatic (reprojection) or manual. See images `extrinsicCalib_*.jpg` and pickles `cameraIntrinsicsExtrinsics_soln*.pickle` under each `CamX/InputMedia/calibration/`.
- Upside-down checker handling (back wall): `main.py` sets rotation angles for TRC export.
    - Upright: `{'y': 90, 'z': 180}`.
    - Upside-down: `{'y': -90, 'z': 180}`.
- Triangulation: `utilsChecker.triangulateMultiviewVideo(...)` with per-cam confidence weighting and sync refinement.

## TRC Writing & Coordinates
- Writer: `writeTRCfrom3DKeypoints` is invoked from `main.py` and uses rotation angles above to map board coordinates to OpenSim (Y-up). Raw TRC formatting helper is `utils.numpy2TRC`.
- Units: meters. Markers follow OpenPose naming (subset or augmented set post-LSTM).
- Optional local-mode behavior: set `OPENCAP_LOCAL_MODE='true'` to bypass API calls. Some environments use `PYTHONIOENCODING='utf-8'` on Windows.

## Workflows (Windows, conda)
- Environment:
    - OpenSim 4.4 (conda), TensorFlow GPU, OpenCV; external tools in PATH: `C:\openpose\bin`, `C:\ffmpeg\bin`.
- Quick run:
    - `python examples_local_usage.py` or `python local_opencap_pipeline.py <videos> --calibration-dir <calib>`.
- Tests:
    - `pytest tests/` (e.g., `tests/test_api.py`, `tests/test_read_camera_intrinsics.py`).
    - Utilities in `tests/checkerboard_test.py` for chessboard sizing sanity checks.

## Implementation Patterns (do like this)
- Trial gating in `main.py` controls stages: calibration-only vs neutral (scale) vs dynamic (IK).
- Rotation logging and angles are determined from `sessionMetadata['checkerBoard']['placement']` and `isCheckerboardUpsideDown(...)`.
- When editing calibration logic, preserve dual-solution artifacts and image outputs—other scripts use them for selection.
- Keep data layout invariant; many utilities assume exact folder/file patterns.

## Pointers for Agents
- Before changing pose detector or resolutions, ensure GPU memory fits; OpenPose paths come from `utils.getOpenPoseDirectory()`.
- For coordinate issues, inspect TRC head and rotation angles in `main.py` logs; expect Y to be dominant post-rotation.
- Cloud mode paths and API posting live in `utils.py` (`postFileToTrial`, `getCalibration*`, `postCalibration*`). Use `OPENCAP_LOCAL_MODE` to avoid server calls during local dev.
````