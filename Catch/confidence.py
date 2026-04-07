import numpy as np
from internalState import DepthJobStatus, InternalState


def compute_confidence(state: InternalState) -> float:
    """
    Return a confidence score in [0, 1] for the current landing prediction.

    Heuristic ingredients:
    - active tracking
    - recent 2D update
    - 3D state exists
    - prediction mode enabled
    - landing prediction exists
    """

    if not state.tracking_active:
        return 0.0

    score = 0.0

    # 1) 2D confidence
    if state.ball_2d_center_hw is not None:
        score += 0.15
    if state.ball_2d_updated:
        score += 0.10

    # 2) 3D confidence
    if state.ball_3d_position_atm is not None:
        score += 0.20

    # 3) Ballistics / predictability
    if state.can_predict:
        score += 0.30

    # 4) Landing prediction availability
    if state.landing_prediction_raw_atm is not None:
        score += 0.10
    if state.landing_prediction_smoothed_atm is not None:
        score += 0.10

    # 5) Freshness of the depth pipeline
    if state.depth_job_status in (DepthJobStatus.DONE, DepthJobStatus.SUBMITTED):
        score += 0.05

    return score