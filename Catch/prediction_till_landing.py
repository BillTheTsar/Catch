import numpy as np
import tracker
import vision
from ball import Ball2D, Ball3D
import cv2
import os
import csv
import time
from concurrent.futures import ThreadPoolExecutor
from configVariables import *

# ========= USER SETTINGS =========
video_path    = "C:/Users/HP/Pictures/Camera Roll/(0.536, 2.175).mp4"
out_video     = "track2D/overlay(0.536, 2.175).mp4"
out_crop      = "track2D/cropFootage.mp4"
out_csv       = "track2D/predictTrackMisc.csv"
NPZ_PATH      = "D:/Catch/stereoExperiment/stereo_params.npz"
K_TXT         = "K.txt"   # for the FULL rectified frame
SAVE_DIR      = "track2D"
padding = 32 # With respect to the full-scale image. The padding for vision and depth estimation are separate
weightFunc = lambda x: max(2/(1 + np.exp(4*x)), 0.02)

del_t = 1/FPS # Used for prediction only

def split_sbs(frame):
    h, w = frame.shape[:2]
    mid = w // 2
    return frame[:, :mid], frame[:, mid:]

def updateStrict(rectL_downscaled):
    """
    :param rectL_downscaled: Downscaled, rectified left image from the stereo pair
    :return: True
    A worker function handling the Strict tracker for concurrency.
    """
    # We use Vision2D to find the centroids and centers in rectL_half
    centroids = eyes2DStrict.find_centroids_hsv(rectL_downscaled)  # Bottleneck line
    best_circles = eyes2DStrict.find_best_circles(centroids, H_centroids, W_centroids,
                                                         dist_tolerance=int(SCALE * 160))
    tracker2DStrict.update(best_circles)
    return True

def updateLax(rectL_downscaled):
    """
        :param rectL_downscaled: Downscaled, rectified left image from the stereo pair
        :return: True
        A worker function handling the Strict tracker for concurrency.
        """
    # We use Vision2D to find the centroids and centers in rectL_half
    centroids = eyes2DLax.find_centroids_hsv(rectL_downscaled) # Parallelize!
    best_circles = eyes2DStrict.find_best_circles(centroids, H_centroids, W_centroids,
                                                      dist_tolerance=int(SCALE*160))
    tracker2DLax.update(best_circles)
    return True

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
mapLx_downscaled = mapLx[::2, ::2] * SCALE # These 4 downscaled maps are for ball detection in Vision2D
mapLy_downscaled = mapLy[::2, ::2] * SCALE
# mapRx_downscaled = mapRx[::2, ::2] * SCALE
# mapRy_downscaled = mapRy[::2, ::2] * SCALE
mapLx_crop_3D = np.empty((CROP_H_3D,CROP_W_3D), np.float32) # These 4 are for depth estimation
mapLy_crop_3D = np.empty((CROP_H_3D,CROP_W_3D), np.float32)
mapRx_crop_3D = np.empty((CROP_H_3D,CROP_W_3D), np.float32)
mapRy_crop_3D = np.empty((CROP_H_3D,CROP_W_3D), np.float32)
mapLx_crop_2D = np.empty((CROP_H_2D, CROP_W_2D), np.float32) # These 2 are for active tracking
mapLy_crop_2D = np.empty((CROP_H_2D, CROP_W_2D), np.float32)

# Opening video
cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise IOError(f"Could not open video: {video_path}")
fps    = cap.get(cv2.CAP_PROP_FPS)
W_full = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))//2
H_full = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
H_centroids, W_centroids = H_full * SCALE, W_full * SCALE

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(out_video), fourcc, fps//4, (int(W_full*SCALE), int(H_full*SCALE)))
if not out.isOpened():
    cap.release()
    raise IOError(f"Could not open output VideoWriter: {out_video}")
# outCrop = cv2.VideoWriter(str(out_crop), fourcc, fps//4, (CROP_W_2D, CROP_H_2D))
# if not outCrop.isOpened():
#     cap.release()
#     raise IOError(f"Could not open output VideoWriter: {out_crop}")

B = 3
F = 7
# Instantiating all objects
tracker2DStrict = tracker.Tracker2D(B=B,F=F,dispTolerance=0.05,radiusTolerance=int(16*SCALE))
tracker2DLax = tracker.Tracker2D(B=B,F=F,dispTolerance=0.05,radiusTolerance=int(16*SCALE))
strictLaxPool = ThreadPoolExecutor(max_workers=2)

tracker_3D = tracker.Tracker3D(B=B, dispTolerance=0.15, angleTolerance=np.pi/4)

eyes2DStrict = vision.Vision2D(H=H_full, W=W_full, B=B, LOWER=[26, 75, 80], UPPER=[41, 255, 255],
                          LOWERR=[23,70,75], UPPERR=[43,255,255], minradius=2, maxradius=64, ksize=3)
eyes2DLax = vision.Vision2D(H=H_full, W=W_full, B=B, LOWER=[21, 70, 80], UPPER=[45, 255, 255],
                            LOWERR=[23,70,75], UPPERR=[43,255,255], minradius=2, maxradius=24, ksize=3)
eyes2DFocus = vision.Vision2D(H=CROP_H_2D, W=CROP_W_2D, B=1, LOWER=[26, 75, 80], UPPER=[41, 255, 255],
                              LOWERR=[23,75,75], UPPERR=[43,255,255], minradius=2, maxradius=64, ksize=3)
eyes3D = vision.Vision3D(H_full=H_full, W_full=W_full, K_path=K_TXT, engine_path=ENGINE_PATH)
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
    # Delete these lines afterwards!
    left_downscaled = cv2.resize(left_full, (int(W_full * SCALE), int(H_full * SCALE)),
                                 interpolation=cv2.INTER_AREA)
    rectL_downscaled = cv2.remap(left_downscaled, mapLx_downscaled, mapLy_downscaled, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT)
    if tracker2DStrict.activelyTracking: # Tight crop but full resolution
        # First create a crop around where the ball was previously seen
        trackedBall = tracker2DStrict.balls[tracker2DStrict.activeBallIndex]
        trackedBall_h, trackedBall_w = trackedBall.position # In normalized coordinates
        withinBound, TL, relCenter = eyes2DFocus.ball_within_bounds(
            H_full, W_full, (trackedBall_h * H_full, trackedBall_w * W_full), CROP_H_2D, CROP_W_2D, padding=0)
        if withinBound:
            corner_h_crop_2D, corner_w_crop_2D = TL
            mapLx_crop_2D[:] = mapLx[
                corner_h_crop_2D:corner_h_crop_2D + CROP_H_2D, corner_w_crop_2D:corner_w_crop_2D + CROP_W_2D]
            mapLy_crop_2D[:] = mapLy[
                corner_h_crop_2D:corner_h_crop_2D + CROP_H_2D, corner_w_crop_2D:corner_w_crop_2D + CROP_W_2D]
            Lc2D = cv2.remap(
                left_full,
                mapLx_crop_2D,
                mapLy_crop_2D,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT
            )
            # outCrop.write(Lc2D)
            centroids = eyes2DFocus.find_centroids_hsv(Lc2D)  # Bottleneck line
            if centroids: # We found at least 1 ball
                centroid_h, centroid_w, radius = centroids[0] # Convert to global normalized coordinates
                global_centroid_h = (corner_h_crop_2D + centroid_h * CROP_H_2D)/H_full
                global_centroid_w = (corner_w_crop_2D + centroid_w * CROP_W_2D)/W_full
                radius = int(radius * SCALE)
                tracker2DStrict.update([[global_centroid_h, global_centroid_w, radius]])
                tracker2DLax.update([[global_centroid_h, global_centroid_w, radius]])
            else: # We didn't find any balls within the crop
                tracker2DStrict.update([])
                tracker2DLax.update([])
        else: # Tracked ball not even in the frame
            tracker2DStrict.update([])
            tracker2DLax.update([])
    else: # Full FOV but downscaled resolution
        # Uncomment these lines afterwards!
        # left_downscaled = cv2.resize(left_full, (int(W_full * SCALE), int(H_full * SCALE)),
        #                              interpolation=cv2.INTER_AREA)
        # rectL_downscaled = cv2.remap(left_downscaled, mapLx_downscaled, mapLy_downscaled, cv2.INTER_LINEAR,
        #                   borderMode=cv2.BORDER_CONSTANT)
        f_strict = strictLaxPool.submit(updateStrict, rectL_downscaled)
        f_lax = strictLaxPool.submit(updateLax, rectL_downscaled)
        f_strict.result()
        f_lax.result()
        meta2DTracker.produceMatchMap(tracker2DStrict, tracker2DLax, 0.03)
        meta2DTracker.processMatchMap(tracker2DStrict, tracker2DLax)
    tracker2DStrict.checkActive() # Activates or deactivates for the next frame

    # We write to the csv file now
    line = [frame_id]
    for ball in tracker2DStrict.balls:
        if ball is None:
            line += [None] * lengthPerEntry
        else:
            line += [ball.position[1], ball.position[0], ball.radius, ball.updated, ball.confirmed_ball]
    writer.writerow(line)

    # ----------------------------------- The 3D stuff happens here -----------------------------------

    # Frame annotation
    for i, ball in enumerate(tracker2DStrict.balls):
        if ball is None or not ball.confirmed_ball: # We can't do anything in 3D
            tracker_3D.balls[i] = None
            continue
        ball3D = tracker_3D.balls[i] # The corresponding 3D counterpart
        (center_h, center_w) = ball.position # Normalized 2D coordinates
        withinBound, TL, relCenter = eyes2DStrict.ball_within_bounds(
            H_full, W_full, (center_h * H_full,center_w * W_full), CROP_H_3D, CROP_W_3D, padding=padding)
        if not withinBound: # Either out of bounds or not a confirmed ball. Ignore.
            continue
        cv2.circle(rectL_downscaled, center=(int(W_centroids * ball.position[1]), int(H_centroids * ball.position[0])),
                radius=ball.radius, color=(0, 255, 255), thickness=2) # Draw a yellow circle

        # Depth observation, we should only perform once every depthEstimationPeriod frames and if the ball was seen
        if frame_id % depthEstimationPeriod == 0:
            if ball.updated: # We saw the 2D ball this frame
                corner_h_crop_3D, corner_w_crop_3D = TL
                mapLx_crop_3D[:] = mapLx[corner_h_crop_3D:corner_h_crop_3D + CROP_H_3D, corner_w_crop_3D:corner_w_crop_3D + CROP_W_3D]
                mapLy_crop_3D[:] = mapLy[corner_h_crop_3D:corner_h_crop_3D + CROP_H_3D, corner_w_crop_3D:corner_w_crop_3D + CROP_W_3D]
                mapRx_crop_3D[:] = mapRx[corner_h_crop_3D:corner_h_crop_3D + CROP_H_3D, corner_w_crop_3D:corner_w_crop_3D + CROP_W_3D]
                mapRy_crop_3D[:] = mapRy[corner_h_crop_3D:corner_h_crop_3D + CROP_H_3D, corner_w_crop_3D:corner_w_crop_3D + CROP_W_3D]

                Lc = cv2.remap(
                    left_full,
                    mapLx_crop_3D,
                    mapLy_crop_3D,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT
                )
                Rc = cv2.remap(
                    right_full,
                    mapRx_crop_3D,
                    mapRy_crop_3D,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT
                )
                # The following lines requires 1.8 frames of compute. We can upper bound it by 2.
                (X, Y, depth) = eyes3D.estimate_position(ball, Lc, Rc, relCenter)  # We have the observed coordinates
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
                    pos3D = tracker_3D.educated_guess_position(ball, ball3D, eyes3D)
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
                pos3D = tracker_3D.educated_guess_position(ball, ball3D, eyes3D)
                posStar3D = tracker_3D.predict_position_until_landing(ball3D, del_t)
                tracker_3D.check_observation_prediction_compatible(ball3D, pos3D)
                pos3D = tracker_3D.filter_observation_prediction(ball3D, pos3D, weightFunc)
                tracker_3D.update(i, ball3D, pos3D, ball.updated)
            else:
                continue # Can't do anything at this stage

        # Fix from here!
        X, Y, depth = pos3D
        # cv2.putText(rectL_downscaled, f"({np.round(X, 2)}, {np.round(Y, 2)}, {np.round(depth, 2)})",
        #             (int(W_centroids * ball.position[1]), int(H_centroids * ball.position[0])),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5,
        #             (200, 200, 0),
        #             1,
        #             2)

        if posStar3D is not None and len(posStar3D)>0:
            (XStar, YStar, depthStar) = posStar3D[-1]
            print(frame_id, [XStar, YStar, depthStar], [X, Y, depth])

            # Makeshift visualization code
            center_xStar = int((XStar * fx / depthStar + cx) / 2)
            center_yStar = int((YStar * fy / depthStar + cy) / 2)
            if ((center_xStar < 0 or center_xStar > SCALE * W_full) or
                    (center_yStar < 0 or center_yStar > SCALE * H_full) or depthStar < 0):
                continue
            cv2.circle(rectL_downscaled, center=(center_xStar, center_yStar),
                       radius=4, color=(255, 0, 0),
                       thickness=4)  # Draw a blue circle

    out.write(rectL_downscaled)
    frame_id += 1

cap.release()
out.release()
# outCrop.release()
csvfile.close()
strictLaxPool.shutdown(wait=True)

t1 = time.time()
print(f"Time taken for video with {frame_id} frames: {t1 - t0} seconds")
print(f"Could have saved {0.06*(t1 - t0)} seconds")