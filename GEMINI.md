# GEMINI.md: Project Overview and Guide

## Project Overview

This project, **OpenCap Core**, is a comprehensive Python-based system for estimating 3D human movement dynamics from standard 2D video. It uses computer vision and biomechanical modeling to produce kinematic data compatible with OpenSim.

The core functionality involves:
1.  **Camera Calibration:** Determining camera intrinsic and extrinsic parameters.
2.  **2D Pose Estimation:** Using either **OpenPose** or **MMPose** to detect keypoints on the human body in videos.
3.  **3D Reconstruction:** Triangulating 2D keypoints from multiple camera views into 3D marker positions.
4.  **Marker Augmentation:** Using an LSTM model to estimate a full-body marker set from the detected keypoints.
5.  **OpenSim Integration:** Scaling a generic musculoskeletal model and running Inverse Kinematics to compute joint angles.

The repository supports two main workflows:
- A **cloud-integrated workflow** that interacts with the `app.opencap.ai` web service.
- A **fully offline local pipeline** (`local_opencap_pipeline.py`) designed for ease of use without any external API or internet dependency. This appears to be the focus of recent development.

**Key Technologies:**
- **Language:** Python
- **Core Libraries:** TensorFlow, OpenCV, OpenSim, Scipy, Pandas
- **Pose Estimation:** OpenPose, MMPose
- **Containerization:** Docker, Docker Compose
- **GPU Acceleration:** NVIDIA CUDA

## Building and Running

There are two primary methods for running the project: using a local Conda environment or using Docker.

### 1. Local Conda Environment

This method is detailed in `README.md` and requires manual setup of the environment and dependencies.

**Setup:**
1.  Create an Anaconda environment: `conda create -n opencap python=3.9 pip spyder`
2.  Activate the environment: `conda activate opencap`
3.  Install OpenSim: `conda install -c opensim-org opensim=4.4=py39np120`
4.  Install CUDA/cuDNN: `conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0`
5.  Install Python packages: `python -m pip install -r requirements.txt`
6.  Manually download and place OpenPose and ffmpeg binaries in `C:\`.

**Running the Offline Pipeline:**
The `local_opencap_pipeline.py` script is the main entry point for the offline workflow.

```bash
# Create a configuration file from the template
python local_opencap_pipeline.py --create-config my_config.yaml

# Run the full pipeline with your videos
# - video_dir contains your motion videos (e.g., walking, running)
# - calibration_dir contains videos of a checkerboard for calibration
python local_opencap_pipeline.py <path_to_video_dir> --calibration-dir <path_to_calibration_dir> --config my_config.yaml
```

### 2. Docker Environment

The `docker` directory contains the setup for a containerized environment using Docker Compose. This is a more isolated and reproducible way to run the pipeline.

**Setup:**
- Ensure you have Docker, Docker Compose, and the NVIDIA Container Toolkit installed.
- Environment variables for image names and GPU settings are expected (likely in a `.env` file).

**Running:**
The `docker` directory contains shell scripts to manage the containers.

```bash
# Start all service containers (mobilecap, openpose, mmpose)
# (Assuming start-containers.sh exists and is configured)
bash docker/start-containers.sh

# Stop all containers
bash docker/stop-all-containers.sh
```

## Development Conventions

*   **Configuration:** Project and pipeline settings are managed via YAML files. A template is provided in `local_config_template.yaml`.
*   **Modularity:** The code is organized into `main` scripts for orchestration and numerous `utils` modules for specific tasks (e.g., `utilsChecker` for calibration, `utilsDetector` for pose estimation).
*   **Offline-First:** The `local_opencap_pipeline.py` represents a shift towards a fully self-contained, offline-first design, reducing reliance on external APIs. It is implemented as a class (`LocalOpenCapPipeline`) that encapsulates the entire process.
*   **Encoding:** There is an active effort to standardize all file I/O to use `encoding='utf-8'` to prevent issues on different operating systems.
*   **Error Handling:** Exception handling includes `logging.error(..., exc_info=True)` to provide full stack traces for easier debugging.
