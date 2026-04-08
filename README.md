# Catch: Real-Time Ball Tracking and Landing Prediction System

---

A real-time computer vision system that tracks a fast-moving ball, reconstructs its 3D trajectory, and predicts its 
landing position for robotic interception.

<table>
  <tr>
    <td align="center">
      <a href="assets/demo1.mp4">
        <img src="assets/demo1.gif" width="360">
      </a><br>
      <sub>Demo 1: Throw from the front</sub>
    </td>
    <td align="center">
      <a href="assets/demo2.mp4">
        <img src="assets/demo2.gif" width="360">
      </a><br>
      <sub>Demo 2: Throw from the back</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="assets/demo3.mp4">
        <img src="assets/demo3.gif" width="360">
      </a><br>
      <sub>Demo 3: Night-time lighting scenario</sub>
    </td>
    <td align="center">
      <a href="assets/demo4.mp4">
        <img src="assets/demo4.gif" width="360">
      </a><br>
      <sub>Demo 4: Strong exposure lighting scenario</sub>
    </td>
  </tr>
</table>

## Overview

This project implements a full pipeline for real-time ball tracking. Its stages are designed to balance 
accuracy and real-time performance.


1. **2D Detection and Tracking**  
   The ball is detected using HSV-based color masking and tracked across frames using a custom multi-object tracker.  
   This stage prioritizes speed and robustness to efficiently distinguish balls from similarly colored background objects.
2. **Cropped Depth Estimation**  
   Instead of processing the full frame, a cropped region around the detected ball is passed to the depth estimation model S2M2.  
   This significantly reduces compute with a small loss in image context.
3. **3D Reconstruction**  
   The 2D position and depth are combined with camera calibration parameters to recover the ball’s observed position in 3D space. We
   fuse observations with predictions from an internal regression model to produce a robust estimate.
4. **Trajectory Prediction**  
   A physics-based model uses the robust estimates to predict the ball’s trajectory and predict its landing position.

5. **Kalman Filtering**  
   Landing predictions are smoothed using a Kalman filter to reduce frame-to-frame noise and improve temporal consistency.

6. **State Output and Telemetry**  
   The system maintains a structured internal state and outputs real-time telemetry, enabling integration with downstream systems such as robotic control.

## Project Structure

```text
Catch/
├── output/            # Output directory (ignored)
├── K.txt              # 
├── annotator.py       # Offline visualization on recorded footage
├── confidence.py      # Confidence estimation
├── config.py          # Central configuration
├── estimator.py       #
├── internalState.py   # System state representation
├── kalman.py          # Landing smoothing
├── main.py            # Live inference from camera
├── run.py             # CLI entrypoint
├── tracker.py         # 2D + 3D tracking logic
└── vision.py          # 2D detection + 3D depth estimation    
```