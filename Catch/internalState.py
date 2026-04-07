from dataclasses import dataclass
from typing import Optional
import numpy as np
import time
from enum import Enum

def array_to_list(x):
    if x is None:
        return None
    return x.tolist()

class DepthJobStatus(Enum):
    NONE = "none" # Doesn't exist yet
    PENDING = "pending" # Working on it
    DONE = "done" # Done but no new job submitted
    SUBMITTED = "submitted" # Submitted a new job

@dataclass
class InternalState:
    frame_id: int
    timestamp: float

    tracking_active: bool
    continuous_tracking: bool
    camera_ok: bool

    active_ball_index: Optional[int] = None

    ball_2d_center_hw: Optional[np.ndarray] = None
    ball_2d_radius: Optional[int] = None
    ball_2d_updated: bool = False

    ball_3d_position_tbd: Optional[np.ndarray] = None # The actual 3D position yet to be calculated
    ball_3d_position_atm: Optional[np.ndarray] = None # The most up-to-date position we have
    can_predict: bool = False

    landing_prediction_raw_tbd: Optional[np.ndarray] = None # The actual landing prediction yet to be calculated
    landing_prediction_raw_atm: Optional[np.ndarray] = None # The most up-to-date landing prediction we have
    landing_prediction_smoothed_tbd: Optional[np.ndarray] = None
    landing_prediction_smoothed_atm: Optional[np.ndarray] = None
    confidence: float = 0.0

    depth_job_status: DepthJobStatus = DepthJobStatus.NONE

    def to_serializable_dict(self):
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "tracking_active": self.tracking_active,
            "continuous_tracking": self.continuous_tracking,
            "camera_ok": self.camera_ok,
            "active_ball_index": self.active_ball_index,
            "ball_2d_center_hw": array_to_list(self.ball_2d_center_hw),
            "ball_2d_radius": self.ball_2d_radius,
            "ball_2d_updated": self.ball_2d_updated,
            "ball_3d_position_tbd": array_to_list(self.ball_3d_position_tbd),
            "ball_3d_position_atm": array_to_list(self.ball_3d_position_atm),
            "can_predict": self.can_predict,
            "landing_prediction_raw_tbd": array_to_list(self.landing_prediction_raw_tbd),
            "landing_prediction_raw_atm": array_to_list(self.landing_prediction_raw_atm),
            "landing_prediction_smoothed_tbd": array_to_list(self.landing_prediction_smoothed_tbd),
            "landing_prediction_smoothed_atm": array_to_list(self.landing_prediction_smoothed_atm),
            "confidence": self.confidence,
            "depth_job_status": self.depth_job_status.value,
        }