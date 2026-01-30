from collections import deque
import estimator
import numpy as np
from configVariables import *

class Ball:
    def __init__(self, position, velocity, radius, N, F):
        self.position = position # normalized numpy array
        self.velocity = velocity # numpy array
        self.radius = radius
        self.N = N # N is the number of past data points we keep for prediction
        self.F = F # The frame in which the ball was created or last updated
        self.pPast = deque(maxlen=N) # Make these queues
        self.vPast = deque(maxlen=N)

class Ball2D(Ball):
    def __init__(self, position, velocity, radius, N, F, status_threshold=10):
        Ball.__init__(self, position, velocity, radius, N, F)
        # Status variables
        self.status_threshold = status_threshold # The number of contiguous frames to consider a patch of yellow a ball
        self.contiguous_seen = 1
        self.confirmed_ball = False # Will be confirmed whenever self.contiguous_seen >= a certain threshold

        self.updated = True  # Whether the ball was updated this frame
        self.framesSinceUpdate = 1
        self.lastObservedPosition = self.position # self.position could contain predictions too, not just observations
        self.lastObservedVelocity = self.velocity

    def move(self, p):
        """A method that updates the position, and velocity of the ball"""
        self.velocity = p - self.position
        self.lastObservedVelocity = p - self.lastObservedPosition # Difference between two observations
        self.position = p
        self.lastObservedPosition = p
        self.pPast.append(self.position)
        self.vPast.append(self.velocity)

    def unseen_move(self):
        """A method that updates the position and velocity of the ball when it was not seen"""
        self.position = self.position + self.lastObservedVelocity / self.framesSinceUpdate
        self.velocity = self.lastObservedVelocity / self.framesSinceUpdate
        self.pPast.append(self.position)
        self.vPast.append(self.velocity)

    def confirm_status(self, seen):
        """A method that updates contiguous_seen and therefore determines whether a ball is confirmed to be visible"""
        if seen:
            self.contiguous_seen += 1
        # elif not seen and not self.confirmed_ball:
        #     self.contiguous_seen = 0
        else: # Not seen but confirmed_ball already
            self.contiguous_seen = max(0, self.contiguous_seen - 1)
        if self.contiguous_seen >= self.status_threshold:
            self.confirmed_ball = True
        else:
            self.confirmed_ball = False

    def prime(self, num):
        """For a ball in the strict regime, this reduces the number of observations until we meet the status_threshold.
        num is the number of observations we still need to make before meeting the status_threshold."""
        self.contiguous_seen = max(self.contiguous_seen, self.status_threshold - num)



class Ball3D(Ball):
    def __init__(self, position, velocity, radius, N, F, predictionThreshold=2):
        # Notice that radius doesn't hold any meaning here since it is fixed in 3D
        Ball.__init__(self, position, velocity, radius, N, F)
        self.pPast.append(self.position)
        self.pPrimePast = deque(maxlen=N) # The effective past positions we use for prediction
        self.pPrimePast.append(self.position) # Necessary initialization
        self.predictedNextPosition = None # This would be a 3D vector once running
        self.predictedLandingPosition = None # This would be a 3D vector once determined
        self.predictedThreshold = predictionThreshold # The # consecutive depth measurements before we start predicting
        self.consecutiveZ = 1
        self.canPredict = False


    def unseen_move(self):
        """A method that updates the position and velocity of the ball when it was not seen"""
        pass

    def rescalePast(self):
        """Resets self.pPast, self.vPast and self.pPrimePast to initialize prediction
        1. At the start, we have self.predictedThreshold number of data points spaced apart by depthEstimationPeriod
            Thus, we have (self.predictedThreshold - 1)*depthEstimationPeriod + 1 potential interpolates to work with.
        2. We sequentially densify our pPast, vPast and pPrimePast
        """
        tempPPast = deque(maxlen=self.N)
        tempVPast = deque(maxlen=self.N)
        # tempPPrimePast = deque(maxlen=self.N)
        while len(self.pPast) > self.predictedThreshold:
            self.pPast.popleft()
        for i in range(self.predictedThreshold - 1): # At this point, self.predictedThreshold = len(self.pPast)
            startP = self.pPast[i]
            # print(len(self.pPast), i, self.predictedThreshold-1)
            endP = self.pPast[i + 1]
            tempPPast.append(startP) # We add the starting point
            for j in range(1, depthEstimationPeriod):
                t = j/depthEstimationPeriod
                intermediate = (1.0 - t)*startP + t*endP
                tempPPast.append(intermediate)
                tempVPast.append((endP-startP)/depthEstimationPeriod) # Even velocity by design
            tempVPast.append((endP-startP)/depthEstimationPeriod)
        tempPPast.append(endP)# We add the end point here
        self.pPast = tempPPast.copy()
        self.pPrimePast = tempPPast.copy()
        self.vPast = tempVPast.copy()

    def move(self, p):
        """A method that updates the position, and velocity of the ball"""
        self.velocity = p - self.position
        self.position = p
        self.pPast.append(self.position)
        self.pPrimePast.append(self.position) # Remember that pPrimePast should be a subset of pPast
        self.vPast.append(self.velocity)
