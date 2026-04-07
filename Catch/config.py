# config.py

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from typing import Callable

"""
To a fellow user, please read the following key to find out what entries you must change as you experiment, which ones
you can use as given, and which ones you shouldn't touch.

Key:
    - MUST CHANGE: This variable should change across setup changes, update according to your setup.
    - No key: Should serve most experiments and setups well, only change if you want to play around.
    - DO NOT CHANGE: Self-explanatory, only change if you know exactly what it does.
"""


@dataclass(frozen=True)
class CropConfig:
    crop_w_2d: int = 240
    crop_h_2d: int = 200
    crop_w_3d: int = 224
    crop_h_3d: int = 224


@dataclass(frozen=True)
class RuntimeConfig:
    fps: int = 60 # FPS of camera.
    scale: float = 0.5 # Downscaling factor for any vision processing. DO NOT CHANGE.
    camera_height: float = 0.90 # Height of camera above ground, assumed to be fixed. MUST CHANGE.
    device_index: int = 1 # Index of stereo camera module. MUST CHANGE.
    show_print: bool = True # Whether to show basic debugging text, not telemetry!
    write_csv: bool = False # Whether to write tracking data to a csv.
    save_video: bool = True # Whether to save our annotations to a video.


@dataclass(frozen=True)
class SmoothingConfig:
    x_alpha: float = 0.8
    y_alpha: float = 0.8
    z_alpha: float = 0.8


@dataclass(frozen=True)
class PathConfig:
    engine_path: Path = Path("D:/S2M2/s2m2/engines/S2M2_L_224_224_fp16.engine") # Path to accelerated TRT engine.
    video_path: Path = Path("C:/Users/HP/Pictures/Camera Roll/(-0.291, 3.080).mp4") # Source video for annotation. MUST CHANGE.
    out_video_path: Path = Path("track2D/overlay(-0.291, 3.080)Kalman.mp4") # Output video path annotator writes to. MUST CHANGE.
    out_csv_path: Path = Path("track2D/predictTrackMisc.csv") # Output csv path annotator and main write to.
    npz_path: Path = Path("D:/Catch/stereoExperiment/stereo_params.npz") # npz file annotator and main read from. MUST CHANGE.
    k_txt: Path = Path("K.txt") # The K file to convert disparity to meters. MUST CHANGE.
    save_dir: Path = Path("track2D") # The directory we save outputs to.


@dataclass(frozen=True)
class Tracker2DConfig:
    B: int = 3 # The number of balls tracker_2D tracks
    F: int = 7 # DO NOT CHANGE.
    disp_tolerance: float = 0.05 # The amount we allow 2D balls to move in one frame (in normalized units).
    radius_tolerance: int = 16 # The amount we allow the radii of 2D balls to change in one frame (in pixels w.r.t. the full scale video).
    dist_tolerance: int = 160 # The minimum distance between two balls for us to consider them distinct (in pixels w.r.t. the full scale video).
    n_ball: int = 5 # The number of past data points 2D balls use to make predictions.
    status_threshold: int = 16 # The number of consecutive frames we need to see a ball before we conclude it is a ball.
    prime: int = 2 # DO NOT CHANGE.


@dataclass(frozen=True)
class Tracker3DConfig:
    disp_tolerance: float = 0.15 # The distance in meters we allow 3D balls to move in one frame.
    angle_tolerance: float = np.pi/4 # The angle change in radians we allow 3D balls to make in one frame.
    prediction_threshold: int = 2 # The number of consecutive depth estimations to begin prediction.
    n_ball: int = 9  # The number of past data points 3D balls use to make predictions.
    min_pPrime_len: int = 3 # DO NOT CHANGE.
    velocity_bound: float = 1.44 # DO NOT CHANGE.


@dataclass(frozen=True)
class Vision2DStrictConfig:
    padding: int = 32 # With respect to the full-scale image. The padding for vision and depth estimation are separate
    LOWER: np.ndarray = field(default_factory=lambda: np.array([26, 75, 80])) # The strict lower bound to detect tennis balls. MUST CHANGE according to your tennis ball.
    UPPER: np.ndarray = field(default_factory=lambda: np.array([41, 255, 255])) # The HSV upper bound to detect tennis balls. MUST CHANGE according to your tennis ball.
    LOWERR: np.ndarray = field(default_factory=lambda: np.array([23,70,75])) # MUST CHANGE. Make this slightly lower than LOWER.
    UPPERR: np.ndarray = field(default_factory=lambda: np.array([43,255,255])) # MUST CHANGE. Make this slightly higher than UPPER.
    min_radius: int = 2 # The minimum radius we allow our tennis balls to be (in pixels w.r.t. the full scale video).
    max_radius: int = 96 # The maximum radius we allow our tennis balls to be (in pixels w.r.t. the full scale video).
    ksize: int = 3 # The size of the morphology kernel. DO NOT CHANGE.


@dataclass(frozen=True)
class Vision2DLaxConfig:
    LOWER: np.ndarray = field(default_factory=lambda: np.array([21, 70, 80])) # MUST CHANGE.
    UPPER: np.ndarray = field(default_factory=lambda: np.array([45, 255, 255])) # MUST CHANGE.
    LOWERR: np.ndarray = field(default_factory=lambda: np.array([23,70,75])) # MUST CHANGE. Should be identical to that of vision2d_strict.
    UPPERR: np.ndarray = field(default_factory=lambda: np.array([43,255,255])) # MUST CHANGE. Should be identical to that of vision2d_strict.
    min_radius: int = 2
    max_radius: int = 24
    ksize: int = 3


@dataclass(frozen=True)
class Vision2DFocusConfig:
    LOWER: np.ndarray = field(default_factory=lambda: np.array([26, 75, 80])) # MUST CHANGE.
    UPPER: np.ndarray = field(default_factory=lambda: np.array([41, 255, 255])) # MUST CHANGE.
    LOWERR: np.ndarray = field(default_factory=lambda: np.array([23,75,75])) # MUST CHANGE.
    UPPERR: np.ndarray = field(default_factory=lambda: np.array([43,255,255])) # MUST CHANGE.
    min_radius: int = 2
    max_radius: int = 96
    ksize: int = 3


@dataclass(frozen=True)
class AnnotatorConfig:
    max_buffer_size: int = 20
    ball_color: tuple = (0, 255, 255)
    landing_prediction_color: tuple = (255, 0, 0)
    font_position: tuple = (50, 50)
    font_color: tuple = (200, 200, 0)


@dataclass(frozen=True)
class TelemetryConfig:
    enabled: bool = True # Whether we output telemetry.
    every_n_frames: int = 20
    sleep_for: float = 0.05
    state_history_len: int = 60


@dataclass(frozen=True)
class KalmanConfig:
    process_var: float = 1e-4 # Small value means we believe that the landing prediction should be constant.
    measurement_var: float = 1e-2 # Small value means we trust the measurement a lot.


@dataclass(frozen=True)
class FilterConfig:
    @staticmethod
    def weightFunc(x): # DO NOT CHANGE.
        return max(2/(1 + np.exp(4*x)), 0.002)
    weight_func: Callable[[float], float] = weightFunc


@dataclass(frozen=True)
class Config:
    crop: CropConfig = CropConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    smoothing: SmoothingConfig = SmoothingConfig()
    paths: PathConfig = PathConfig()
    tracker2d: Tracker2DConfig = Tracker2DConfig()
    tracker3d: Tracker3DConfig = Tracker3DConfig()
    vision2d_strict: Vision2DStrictConfig = Vision2DStrictConfig()
    vision2d_lax: Vision2DLaxConfig = Vision2DLaxConfig()
    vision2d_focus: Vision2DFocusConfig = Vision2DFocusConfig()
    annotator: AnnotatorConfig = AnnotatorConfig()
    telemetry: TelemetryConfig = TelemetryConfig()
    kalman: KalmanConfig = KalmanConfig()
    filter: FilterConfig = FilterConfig()


CONFIG = Config()
