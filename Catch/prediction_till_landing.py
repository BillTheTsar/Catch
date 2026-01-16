import numpy as np
import tracker
import vision
from ball import Ball2D, Ball3D
import cv2
import os
import csv
import time
import estimator
from collections import deque
from configVariables import *

# ========= USER SETTINGS =========
# video_path    = "C:/Users/HP/Pictures/Camera Roll/WIN_20251128_20_00_11_Pro.mp4"
video_path    = "C:/Users/HP/Pictures/Camera Roll/WIN_20251128_19_59_42_Pro.mp4"
out_video     = "track2D/overlay5.mp4"
out_csv       = "track2D/predictTrackMisc.csv"
NPZ_PATH      = "D:/Catch/stereoExperiment/stereo_params.npz"
K_TXT_FULL    = "K.txt"   # for the FULL rectified frame
SCALE         = 0.5 # Pick 0.5 or 0.25, or at your own expense
SAVE_DIR      = "track2D"
ENGINE_PATH = "D:/S2M2/s2m2/engines/S2M2_L_224_224_fp16.engine"
padding = 20 # With respect to the full-scale image. The padding for vision and depth estimation are separate
predictionGap = 4 # The no. frames between when we made the prediction and when we check whether it was accurate
weightFunc = lambda x: max(2/(1 + np.exp(4*x)), 0.02)

del_t = 1/FPS

def find_best_circles(circles_info, H, W, dist_tolerance=20):
    # dist_tolerance is the number of pixels between centers and seen centroid centers for them to be considered the
    # centers for the same circle
    best_circles_info = []
    centers_seen = []
    while circles_info:
        (center_h, center_w, radius) = circles_info.pop(0)
        if not centers_seen:  # The first circle: we should add the circle unconditionally
            best_circles_info.append([center_h, center_w, radius])
            centers_seen.append(np.array([center_h * H, center_w * W]))
            continue
        center = np.array([center_h * H, center_w * W])
        min_dist = min([np.linalg.norm(center - centroid) for centroid in centers_seen])
        if min_dist >= dist_tolerance:
            best_circles_info.append([center_h, center_w, radius])
            centers_seen.append(center)
    return best_circles_info # Already sorted by design!

def split_sbs(frame):
    h, w = frame.shape[:2]
    mid = w // 2
    return frame[:, :mid], frame[:, mid:]

def ball_within_bounds(H_full, W_full, center=None, padding=padding):
    """Remember that center is (height, width)
    The function determines whether the center of the ball is within the padding of the image.
    This should be used for depth estimation only, vision has its own padding system.
    Returns both the boolean and the position of the ball center within the crop frame"""

    center_h, center_w = map(int, center)
    withinBound = (padding <= center_h <= H_full - padding) and (padding <= center_w <= W_full - padding)
    if not withinBound:
        return False, None, None

    (hTL, wTL) = center_h - CROP_H // 2, center_w - CROP_W // 2 # Actual pixel coordinates of the top-left corner
    (hBR, wBR) = hTL + CROP_H, wTL + CROP_W
    if hTL < 0: # We have the top-left corner too high
        hBR = hBR - hTL
        hTL = 0 # hTL = hTL - hTL
    elif hBR > H_full: # We have the bottom-left corner too low
        hTL = hTL - hBR + H_full
        hBR = H_full  # hBR = hBR - hBR + H_full
    if wTL < 0: # We have the top-left corner too much to the left
        wBR = wBR - wTL
        wTL = 0  # wTL = wTL - wTL
    elif wBR > W_full: # We have the bottom-left corner too much to the right
        wTL = wTL - wBR + W_full
        wBR = W_full  # wBR = wBR - wBR + W_full

    relCenterH, relCenterW = (center_h - hTL, center_w - wTL)
    return True, (hTL, wTL), (relCenterH, relCenterW)

"""
Steps we must take:
1. Initialize the appropriate objects
2. Load the images and rectify them!
"""

# =================================

# K file
with open("K.txt", "r") as f:
    l1 = f.readline().strip()
    l2 = f.readline().strip()
vals = list(map(float, l1.split()))
fx, _, cx, _, fy, cy, _, _, _ = vals

# NPZ
params   = np.load(NPZ_PATH, allow_pickle=True)
mapLx, mapLy = params["mapLx"], params["mapLy"]
mapRx, mapRy = params["mapRx"], params["mapRy"]

# Opening video
cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise IOError(f"Could not open video: {video_path}")
fps    = cap.get(cv2.CAP_PROP_FPS)
W_full = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))//2
H_full = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(out_video), fourcc, fps//4, (int(W_full*SCALE), int(H_full*SCALE)))
if not out.isOpened():
    cap.release()
    raise IOError(f"Could not open output VideoWriter: {out_video}")

B = 4
F = 7
# Instantiating all objects
tracker2DStrict = tracker.Tracker2D(B=B,F=F,dispTolerance=0.10,radiusTolerance=int(16*SCALE))
tracker2DLax = tracker.Tracker2D(B=B,F=F,dispTolerance=0.10,radiusTolerance=int(16*SCALE))
tracker_3D = tracker.Tracker3D(B=B, dispTolerance=0.15, angleTolerance=np.pi/4)
eyes2DStrict = vision.Vision2D(H=H_full, W=W_full, B=B, LOWER=[26, 75, 80], UPPER=[41, 255, 255],
                          LOWERR=[23,70,75], UPPERR=[43,255,255], minradius=2, maxradius=64)
eyes2DLax = vision.Vision2D(H=H_full, W=W_full, B=B, LOWER=[21, 70, 80], UPPER=[45, 255, 255],
                            LOWERR=[23,70,75], UPPERR=[43,255,255], minradius=2, maxradius=24)
eyes_3D = vision.Vision3D(H_full=H_full, W_full=W_full, K_path=K_TXT_FULL, engine_path=ENGINE_PATH)
meta2DTracker = tracker.Meta2DTracker(B, B)

# Some OS humdrum
os.makedirs(SAVE_DIR, exist_ok=True)
if os.path.exists(out_csv): # Overwrite the old csv file
    os.remove(out_csv)
csvfile = open(out_csv, "w", newline="")
writer = csv.writer(csvfile)
lengthPerEntry = 5
headerRow = ["frame"]
for i in range(1, B+1):
    headerRow += [f"ball{i}_x", f"ball{i}_y", f"ball{i}_r", f"ball{i}_u", f"ball{i}_c"]
writer.writerow(headerRow)


frame_id = 0
t0 = time.time()

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break

    left_full, right_full = split_sbs(frame_bgr) # Full scale images
    rectL = cv2.remap(left_full, mapLx, mapLy, cv2.INTER_LINEAR) # Rectification
    rectR = cv2.remap(right_full, mapRx, mapRy, cv2.INTER_LINEAR) # Rectification
    # Downscaling for eyes2DStrict
    rectL_half = cv2.resize(rectL, (int(W_full * SCALE), int(H_full * SCALE)), interpolation=cv2.INTER_AREA)

    # We use Vision2D to find the centroids and centers in rectL_half
    centroidsStrict, (H_centroids, W_centroids) = eyes2DStrict.find_centroids_hsv(rectL_half)
    # if found_centroids: # Non-empty centroids detected
    best_circles_strict = find_best_circles(centroidsStrict, H_centroids, W_centroids, dist_tolerance=int(SCALE*160))
    tracker2DStrict.update(best_circles_strict)
    # else:
    #     best_circles, (H_circles, W_circles) = eyes2DStrict.find_circles_hough(rectL_half)
    #     # We now update our system via tracker2DStrict
    #     tracker2DStrict.update(best_circles)
    centroidsLax, (H_centroids, W_centroids) = eyes2DLax.find_centroids_hsv(rectL_half) # Parallelize!
    best_circles_lax = find_best_circles(centroidsLax, H_centroids, W_centroids, dist_tolerance=int(SCALE*160))
    tracker2DLax.update(best_circles_lax)
    meta2DTracker.produceMatchMap(tracker2DStrict, tracker2DLax, 0.03)
    meta2DTracker.processMatchMap(tracker2DStrict, tracker2DLax)

    # We write to the csv file now
    line = [frame_id]
    for ball in tracker2DStrict.balls:
        if ball is None:
            line += [None] * lengthPerEntry
        else:
            line += [ball.position[1], ball.position[0], ball.radius, ball.updated, ball.confirmed_ball]
    writer.writerow(line)

    # ----------------------------------- The real stuff happens here -----------------------------------

    # Frame annotation
    for i, ball in enumerate(tracker2DStrict.balls):
        if ball is None or not ball.confirmed_ball: # We can't do anything
            tracker_3D.balls[i] = None
            continue
        ball3D = tracker_3D.balls[i]  # The corresponding 3D counterpart
        (center_h, center_w) = ball.position
        withinBound, TL, relCenter = ball_within_bounds(H_full, W_full, (center_h * H_full,center_w * W_full))
        if not withinBound: # Either out of bounds or not a confirmed ball. Ignore. Remember to batch!
            continue
        cv2.circle(rectL_half, center=(int(W_centroids * ball.position[1]), int(H_centroids * ball.position[0])),
                radius=ball.radius, color=(0, 255, 255), thickness=2) # Draw a yellow circle

        # Depth observation, we should only perform once every depthEstimationPeriod frames and if the ball was seen
        if frame_id % depthEstimationPeriod == 0:
            if ball.updated: # We saw the 2D ball this frame
                # left_full_rgb = cv2.cvtColor(rectL, cv2.COLOR_BGR2RGB)  # For feeding into s2m2
                # right_full_rgb = cv2.cvtColor(rectR, cv2.COLOR_BGR2RGB)  # For feeding into s2m2
                corner_h_crop, corner_w_crop = TL
                # Lc = left_full_rgb[corner_h_crop:corner_h_crop+CROP_H, corner_w_crop:corner_w_crop+CROP_W].copy()
                # Rc = right_full_rgb[corner_h_crop:corner_h_crop+CROP_H, corner_w_crop:corner_w_crop+CROP_W].copy()
                Lc = rectL[corner_h_crop:corner_h_crop + CROP_H, corner_w_crop:corner_w_crop + CROP_W].copy()
                Rc = rectR[corner_h_crop:corner_h_crop + CROP_H, corner_w_crop:corner_w_crop + CROP_W].copy()
                (X, Y, depth) = eyes_3D.estimate_position(ball, Lc, Rc, relCenter)  # We have the observed coordinates
                pos3D = np.array([X, Y, depth])  # Observation in meters
                if ball3D is None: # Handling an edge case
                    tracker_3D.update(i, ball3D, pos3D, ball.updated) # We create a ball and make no predictions
                    continue
                # We predict where the current ball should be, given past data
                if ball3D.canPredict: # ball3D would not be None in this case
                    posStar3D = tracker_3D.predict_position_until_landing(ball3D, del_t) # Prediction in meters
                    tracker_3D.check_observation_prediction_compatible(ball3D, pos3D) # We check that the prediction fits the observation well enough
                    pos3D = tracker_3D.filter_observation_prediction(ball3D, pos3D, weightFunc)
                    tracker_3D.update(i, ball3D, pos3D, ball.updated) # Remember, we predict, check and only then update
                else:
                    posStar3D = None # No prediction needed
                    tracker_3D.update(i, ball3D, pos3D, ball.updated) # Predict = None, check = None, update
            else: # We didn't see the 2D ball this frame
                if ball3D is None: # It is possible for ball to not be None and ball3D to be None
                    continue
                if ball3D.canPredict:
                    pos3D = tracker_3D.educated_guess_position(ball, ball3D, eyes_3D)
                    posStar3D = tracker_3D.predict_position_until_landing(ball3D, del_t)
                    tracker_3D.check_observation_prediction_compatible(ball3D, pos3D)
                    pos3D = tracker_3D.filter_observation_prediction(ball3D, pos3D, weightFunc)
                    tracker_3D.update(i, ball3D, pos3D, ball.updated)  # Remember, we predict, check and only then update
                else: # We can't even predict
                    tracker_3D.balls[i] = None # We delete the 3D ball
                    posStar3D = None
                    continue
            tracker_3D.update_predictability(i, ball3D, ball.updated)
        else: # Even if we don't observe depth, sometimes we can make predictions based on past depths
            if ball3D is None: # There is no existing 3D data, hence we continue
                continue
            if ball3D.canPredict:
                pos3D = tracker_3D.educated_guess_position(ball, ball3D, eyes_3D)
                posStar3D = tracker_3D.predict_position_until_landing(ball3D, del_t)
                tracker_3D.check_observation_prediction_compatible(ball3D, pos3D)
                pos3D = tracker_3D.filter_observation_prediction(ball3D, pos3D, weightFunc)
                tracker_3D.update(i, ball3D, pos3D, ball.updated)
            else:
                continue # Can't do anything at this stage

        # Fix from here!
        X, Y, depth = pos3D
        cv2.putText(rectL_half, f"({np.round(X, 2)}, {np.round(Y, 2)}, {np.round(depth, 2)})",
                    (int(W_centroids * ball.position[1]), int(H_centroids * ball.position[0])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 0),
                    1,
                    2)

        if posStar3D is not None and len(posStar3D)>0:
            # print(frame_id, pos3D[1], [posStar[1] for posStar in posStar3D])
            (XStar, YStar, depthStar) = posStar3D[-1]
            print(frame_id, [XStar, YStar, depthStar], [X, Y, depth])

            # Makeshift visualization code
            center_xStar = int((XStar * fx / depthStar + cx) / 2)
            center_yStar = int((YStar * fy / depthStar + cy) / 2)
            if ((center_xStar < 0 or center_xStar > SCALE * W_full) or
                    (center_yStar < 0 or center_yStar > SCALE * H_full) or depthStar < 0):
                continue
            cv2.circle(rectL_half, center=(center_xStar, center_yStar),
                       radius=4, color=(255, 0, 0),
                       thickness=4)  # Draw a blue circle
            # print()

    out.write(rectL_half)
    frame_id += 1

cap.release()
out.release()
csvfile.close()

t1 = time.time()
print(f"Time taken for video with {frame_id} frames: {t1 - t0} seconds")


# Things to do:
# 1. Simulate depthEstimationPeriod = 1 DONE
# 2. Show the prediction made predictionGap ago
# 3. Implement no gravity as well
# If observation is clearly wrong, use prediction instead