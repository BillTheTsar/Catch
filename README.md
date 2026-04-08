# Catch: Real-Time Ball Tracking and Landing Prediction System

---

A real-time perception system that tracks a fast-moving ball, reconstructs its 3D trajectory using stereo depth, and 
predicts its landing position for robotic interception.

<table>
  <tr>
    <td align="center">
      <a href="assets/demo1.mp4">
        <img src="assets/demo1.gif" width="400">
      </a><br>
      <sub>Demo 1: Throw from the front</sub>
    </td>
    <td align="center">
      <a href="assets/demo2.mp4">
        <img src="assets/demo2.gif" width="400">
      </a><br>
      <sub>Demo 2: Throw from the back</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="assets/demo3.mp4">
        <img src="assets/demo3.gif" width="400">
      </a><br>
      <sub>Demo 3: Night-time lighting scenario</sub>
    </td>
    <td align="center">
      <a href="assets/demo4.mp4">
        <img src="assets/demo4.gif" width="400">
      </a><br>
      <sub>Demo 4: Strong exposure lighting scenario</sub>
    </td>
  </tr>
</table>

## Overview

This project implements a full pipeline for real-time ball tracking. Its stages balance accuracy, latency and compute 
to achieve real-time performance.


1. **2D Detection and Tracking**  
   - The ball is detected using HSV-based color masking and tracked across frames using a custom multi-object tracker.  
   - This stage prioritizes speed and robustness to efficiently distinguish balls from similarly colored background objects.
2. **Cropped Depth Estimation**  
   - Instead of processing the full frame, a cropped region around the detected ball is passed to the TensorRT-optimized depth estimation model S2M2.  
   - This significantly reduces compute with a small loss in image context.
3. **3D Reconstruction**  
   - The 2D position and depth are combined with camera calibration parameters to recover the ball’s observed position in 3D space.
   - We fuse observations with predictions from a lightweight regression-based motion model to produce a robust estimate.
4. **Trajectory Prediction**  
   - A physics-based model uses the robust estimates to predict the ball’s trajectory and its landing position.

5. **Kalman Filtering**  
   - Landing predictions are smoothed using a Kalman filter to reduce frame-to-frame noise and improve temporal consistency.

6. **State Output and Telemetry**  
   - The system maintains a structured internal state and outputs real-time telemetry, enabling integration with downstream systems such as robotic control.

## Project Structure

```text
Catch/
├── output/            # Output directory (ignored)
├── K.txt              # Camera calibration parameters
├── annotator.py       # 🔴Offline visualization on recorded footage
├── ball.py            # 2D + 3D ball classes
├── confidence.py      # Confidence estimation
├── config.py          # Central configuration
├── estimator.py       # Physics-based predictive models
├── internalState.py   # System state representation
├── kalman.py          # Landing smoothing
├── main.py            # 🔴Live inference from camera
├── run.py             # 🔴CLI entrypoint
├── tracker.py         # 2D + 3D tracking logic
└── vision.py          # 2D detection + 3D depth estimation 
```
Scripts marked 🔴 are intended entrypoints.

---

## Usage

The project provides a CLI interface via `run.py`.

```bash
python run.py --help

usage: run.py [-h] [--config CONFIG] [--device-index DEVICE_INDEX] [--video VIDEO] [--camera-height CAMERA_HEIGHT]
              [--telemetry TELEMETRY] [--write-csv WRITE_CSV] [--show-print SHOW_PRINT]
              {main,annotator}
              
Run Catch system

positional arguments:
  {main,annotator}
    main                Run live inference
    annotator           Run video annotator
...
```
Both `main` and `annotator` require a configuration file. The default config path is `Catch/config.py`. 
We can either modify `config.py` directly or override selected parameters via CLI flags.

### Example usage
```bash
python run.py --video my_video_path.mp4 --camera-height 0.88 --show-print true annotator
python run.py --device-index 1 --camera-height 0.75 --telemetry true --write-csv false main
python run.py --config my_config_path.py main
```

### Notes
This project requires:
- A **pre-generated TensorRT engine** for depth estimation.
- A **camera calibration file** `K.txt` to convert from disparity to depth in meters.

A default `K.txt` provides the format we expect the calibration file to follow, but should be replaced to be accurate to 
your camera module. The TensorRT engine is not included in the repository, as it is hardware-specific.

## Requirements
- Python 3.10+
- NumPy
- OpenCV
- PyTorch


## Limitations
- No automatic camera calibration (requires `K.txt`).
- TensorRT engine must be generated externally.
- The ball is assumed to be sufficiently spherical.

## Planned improvements
- Integrate real-time control for RC-car interception.
- Path-planning and onboard obstacle avoidance.
- Replace color-based detection with a learned model for improved robustness.

## Acknowledgements

- Stereo depth estimation model based on S2M2.
- ChatGPT for refactoring an operational but unstructured repository.