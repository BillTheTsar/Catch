import numpy as np
from config import CONFIG


class LandingKalmanFilter:
    """
    Minimal Kalman filter for smoothing 2D landing predictions.

    State:
        x = [landing_x, landing_y]^T

    Model:
        x_k = A x_{k-1} + w_k
        z_k = H x_k + v_k

    with:
        A = I
        H = I
    """

    def __init__(
        self,
        process_var: float = 4e-2,
        measurement_var: float = 1e-2,
        initial_var: float = 1.0,
    ) -> None:
        """
        Args:
            process_var: variance of process noise Q
            measurement_var: variance of measurement noise R
            initial_var: initial state uncertainty
        """
        self.A = np.eye(2, dtype=np.float64)
        self.H = np.eye(2, dtype=np.float64)

        self.Q = process_var * np.eye(2, dtype=np.float64)
        self.R = measurement_var * np.eye(2, dtype=np.float64)

        self.P = initial_var * np.eye(2, dtype=np.float64)
        self.x = np.zeros((2, 1), dtype=np.float64)

        self.initialized = False

    def reset(self) -> None:
        """Reset the filter to an uninitialized state."""
        self.P = np.eye(2, dtype=np.float64)
        self.x = np.zeros((2, 1), dtype=np.float64)
        self.initialized = False

    def initialize(self, measurement: np.ndarray) -> None:
        """
        Initialize the filter from the first measurement.

        Args:
            measurement: array-like of shape (2,), [landing_x, landing_y]
        """
        z = np.asarray(measurement, dtype=np.float64).reshape(2, 1)
        self.x = z.copy()
        self.P = self.P.copy()
        self.initialized = True

    def predict(self) -> np.ndarray:
        """
        Prediction step.

        Returns:
            Predicted state as shape (2,)
        """
        if not self.initialized:
            raise RuntimeError("Filter must be initialized before predict().")

        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x.ravel()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Update step using a new landing prediction.

        Args:
            measurement: array-like of shape (2,), [landing_x, landing_y]

        Returns:
            Updated filtered state as shape (2,)
        """
        z = np.asarray(measurement, dtype=np.float64).reshape(2, 1)

        if not self.initialized:
            self.initialize(z)
            return self.x.ravel()

        # Predict
        x_pred = self.A @ self.x
        P_pred = self.A @ self.P @ self.A.T + self.Q

        # Innovation
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R

        # Kalman gain
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        # Update
        self.x = x_pred + K @ y
        I = np.eye(2, dtype=np.float64)
        self.P = (I - K @ self.H) @ P_pred

        return self.x.ravel()

    def current_state(self) -> np.ndarray:
        """
        Return current filtered estimate.

        Returns:
            Array of shape (2,)
        """
        if not self.initialized:
            raise RuntimeError("Filter has not been initialized.")
        return self.x.ravel()

    def landing_prediction(self, posStar3D):
        if posStar3D is None or len(posStar3D) == 0:
            return np.array([], dtype=np.float64)
        smoothedPosStar_xz = self.update(np.array([posStar3D[0], posStar3D[2]]))
        smoothedPosStar3D = np.array([smoothedPosStar_xz[0], CONFIG.runtime.camera_height, smoothedPosStar_xz[1]])
        return smoothedPosStar3D