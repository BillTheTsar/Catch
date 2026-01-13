from ball import Ball2D, Ball3D
import numpy as np
import estimator
from configVariables import *
from collections import deque

"""
A Tracker object manages balls in its corresponding dimension: Tracker2D manages up to 3 balls in 2D and Tracker3D
manages up to 3 balls in 3D
"""
class Tracker:
    def __init__(self, B):
        self.B = B # B is the maximum number of balls we track per frame
        self.balls = [None] * self.B # An array containing the B balls Tracker manages


class Tracker2D(Tracker):
    def __init__(self, B, F, dispTolerance, radiusTolerance):
        Tracker.__init__(self, B)
        self.F = F  # F is the maximum number of frames a ball can be missing before we deem it gone
        self.dispTolerance = dispTolerance  # Maximum displacement tolerance allowed per frame
        self.radiusTolerance = radiusTolerance
        self.frame = 0

    def update(self, potential_balls):
        """
        Given potential_balls from Vision2D, we decide which of these become tracked and how.
        potential_balls is an array of 2D arrays of the form (center_h, center_w, radius).
        It is reverse-sorted by radius
        """
        # num_updates = 0
        updated_balls = set() # Contains indices of balls that were updated
        empty_balls = set() # Contains indices of balls = None

        for i, ball in enumerate(self.balls): # Cleaning balls that have not been seen for F frames
            if (ball is not None) and (ball.F == self.frame): # It's been too long since we last saw the ball
                self.balls[i] = None
                empty_balls.add(i)

        for (p1_h, p1_w, r1) in potential_balls:
            updated = False
            # if num_updates == self.B:
            #     break
            p1 = np.array([p1_h, p1_w])
            for i, ball in enumerate(self.balls):
                if ball is None: # We do not update as there is nothing to update
                    empty_balls.add(i)
                    continue
                p0, v0, r0, f0 = ball.position, ball.velocity, ball.radius, ball.F
                taylor_diff = p1 - (p0 + v0)
                if np.linalg.norm(taylor_diff) < self.dispTolerance and abs(r1 - r0) < self.radiusTolerance: # A match
                    ball.move(p1) # Update its position
                    ball.radius = r1
                    ball.F = self.frame
                    # num_updates += 1
                    updated_balls.add(i)
                    updated = True
                    ball.updated = True
                    ball.confirm_status(True)
                    break
            if not updated and empty_balls: # Our potential ball can become a new ball we track!
                first_empty_index = min(empty_balls)
                empty_balls.remove(first_empty_index) # That index will no longer be empty
                self.balls[first_empty_index] = Ball2D(p1, np.array([0, 0]), r1, 6, self.frame)
                # updated = True
                updated_balls.add(first_empty_index)
                # num_updates += 1
                continue

        if len(updated_balls) < self.B: # We know not every ball has been updated
            for i in set(range(self.B)).difference(updated_balls):
                if self.balls[i] is None:
                    continue
                self.balls[i].unseen_move() # We should update them anyway
                self.balls[i].updated = False
                self.balls[i].confirm_status(False) # We didn't see the ball this frame

        self.frame = (self.frame + 1)%self.F # Roll the frame over


class Tracker3D(Tracker):
    def __init__(self, B, dispTolerance, angleTolerance, minPPrimeLen=3):
        Tracker.__init__(self, B)
        self.dispTolerance = dispTolerance # Distance in meters
        self.angleTolerance = angleTolerance # In radians
        self.minPPrimeLen = minPPrimeLen # Must be strictly smaller than the N for balls

    @staticmethod
    def update_predictability(ball: Ball3D, seen) -> None:
        """Checks whether there have been self.predictedThreshold number of consecutive depth measurements"""
        if ball is None: return
        if ball.canPredict: return
        if seen:
            ball.consecutiveZ += 1
            if ball.consecutiveZ == ball.predictedThreshold: # Very important, we reset our pPast, vPast and pPrimePast
                ball.canPredict = True
                ball.rescalePast()
        else:
            ball.consecutiveZ = 1

    def update(self, i, ball, position, seen):
        """We update the 3D balls we track one ball at a time.
        If our prediction deviates too far from the observation or the angle change is too large, then we
            reset ball.pPrimePast. This should resolve impact events such as hitting the floor."""
        if not seen and ball is None: return
        if ball is None: # If there is no ball at the corresponding position
            self.balls[i] = Ball3D(position=position, velocity=np.array([0,0,0]), radius=None, N=9, F=None)
        else: # There is a 3D ball
            ball.move(position)

    def check_observation_prediction_compatible(self, ball: Ball3D, observedPosition):
        """Checks whether our observed position is compatible with our prediction. If it is, do nothing.
        Otherwise, we truncate ball.pPrimePast"""
        if np.linalg.norm(ball.predictedNextPosition - observedPosition) > self.dispTolerance:
            # The code below keeps at most self.minPPrimeLen elements at the end of self.pPrimePast
            while len(ball.pPrimePast) > self.minPPrimeLen:
                ball.pPrimePast.popleft()
            return
        # Angle check
        a = np.linalg.norm(ball.vPast[-1])
        b = np.linalg.norm(ball.vPast[-2])
        if a*b == 0: # Edge case
            return
        else:
            clippedProd = np.clip(np.dot(ball.vPast[-1], ball.vPast[-2])/(a*b), a_min=-1, a_max=1)
            angle = np.arccos(clippedProd)
            if angle <= self.angleTolerance:
                return
            # Angle difference too large
            while len(ball.pPrimePast) > self.minPPrimeLen:
                ball.pPrimePast.popleft()
            return

    def filter_observation_prediction(self, ball: Ball3D, observedPosition, func):
        """Gives a weighted sum of the prediction and the observation based on how far the observation is from the
        prediction. The greater the distance, the more we trust the prediction. func is used to compute the weight."""
        dist = np.linalg.norm(ball.predictedNextPosition - observedPosition)
        alpha = func(dist)
        return alpha*observedPosition + (1-alpha)*ball.predictedNextPosition

    @staticmethod
    def predict_next_position(ball: Ball3D, del_t: float):
        """Predicts the next position of the ball. If the ball is None, we do nothing. Otherwise,
        run estimator.bestVelocity3DGravity."""
        # if ball is None: return
        vStar, ball.predictedNextPosition = estimator.bestVelocity3DGravity(ball.pPrimePast, del_t)
        return ball.predictedNextPosition

    @staticmethod
    def predict_position_n_frames(ball: Ball3D, del_t: float, m):
        """Predicts the next n positions of the ball. If the ball is None, we do nothing. Otherwise,
                run estimator.bestVelocity3DGravity."""
        vStar, ball.predictedNextPosition = estimator.bestVelocity3DGravity(ball.pPrimePast, del_t)
        predictions = []
        n = len(ball.pPrimePast)
        for i in range(n, m+n):
            predictions.append(ball.pPrimePast[0] + vStar * (del_t * i) + 0.5 * np.array([0, 9.81, 0]) * (del_t * i) ** 2)
        return predictions

    @staticmethod
    def predict_position_until_landing(ball: Ball3D, del_t: float):
        """Predicts the positions of the ball until its predicted landing."""
        vStar, ball.predictedNextPosition = estimator.bestVelocity3DGravity(ball.pPrimePast, del_t)

        # pPrimePast = ball.pPrimePast.copy()
        # predictions = [ball.predictedNextPosition]
        # for i in range(20):
        # pPrimePast.append(predictions[-1])
        # vStar, prediction = estimator.bestVelocity3DGravity(pPrimePast, del_t)
        # predictions.append(prediction)

        predictions = []
        n = len(ball.pPrimePast)
        y = ball.predictedNextPosition[1]
        i = 0
        while y <= cameraHeight:
            predictions.append(ball.pPrimePast[0] + vStar * (del_t * (n+i)) + 0.5 * np.array([0, 9.81, 0]) * (del_t * (n+i)) ** 2)
            i += 1
            y = predictions[-1][1]
        return predictions

    @staticmethod
    def educated_guess_position(ball2D: Ball2D, ball3D: Ball3D, eyes3D):
        """Even if we don't observe depth, sometimes we can make predictions based on past depths.
        The difference between educated_guess_position and predict_next_position is that we have x and y information
        for the former and only infer the depth, while we predict all 3 coordinates for the latter."""
        # if ball3D is None: return
        zPred = estimator.geometricVelocitySum(ball3D.pPrimePast, ball3D.vPast, 2, z_alpha)  # We can now make a depth prediction
        xPred = eyes3D.normalized_to_meter_x(ball2D.position[1], zPred)
        yPred = eyes3D.normalized_to_meter_y(ball2D.position[0], zPred)
        pos3DPred = np.array([xPred, yPred, zPred])
        return pos3DPred