import numpy as np
from configVariables import *
from collections import deque

def geometricVelocitySum(pPrimePast: deque, vPast: deque, i, alpha) -> float:
    """We predict the next coordinate by taking a geometric sum of past velocities and adding that quantity to
    the last entry of pPrimePast. The rate of decay is controlled by alpha. We strongly recommend that
    len(pPrimePast) >= 4"""
    lastPosCoord = pPrimePast[-1][i]
    n = len(pPrimePast) # We recommend that n >= 4
    while len(vPast) > (n-1):
        vPast.popleft()
    numerator = 0
    denominator = 0
    for j, velocity in enumerate(reversed(vPast)): # Each velocity is a vector from which we take the ith coordinate
        velocityCoord = velocity[i]
        numerator += alpha**j * velocityCoord
        denominator += alpha**j
    return lastPosCoord + numerator / denominator


def bestVelocity3DGravity(pPrimePast, del_t):
    """Finds the velocity vector that best interpolates pPrimePast with a constant gravity vector
    Returns the best velocity vector and the predicted next position"""
    g = np.array([0, 9.81, 0])
    n = len(pPrimePast)
    if n == 1: # Trivial
        vStar = np.zeros(3)
        pred = pPrimePast[0] + 0.5 * g * (del_t ** 2)
        return vStar, pred
    # if n == 2: # Still trivial
    #     vStar = (pPrimePast[1] - pPrimePast[0] - 0.5*g*del_t**2)/del_t
    #     return vStar, pPrimePast[0] + 2*vStar*del_t + 0.5*g*(2*del_t)**2

    numerator = np.zeros(3, dtype=float)
    denominator = 0
    for j in range(1, n):
        tj = j*del_t
        dj = pPrimePast[j]-pPrimePast[0]-0.5*g*tj**2
        numerator += dj*tj
        denominator += tj**2
    vStar = numerator/denominator
    return vStar, pPrimePast[0] + vStar*del_t*n + 0.5*g*(del_t*n)**2
