import numpy as np
import tracker
import vision
from ball import Ball2D, Ball3D
import cv2
import os
import csv
import time

ENGINE_PATH = "D:/S2M2/s2m2/engines/S2M2_L_224_224_fp16.engine"
CROP_W, CROP_H = 224, 224 # For feeding into the engine
padding = 30 # With respect to the full-scale image. The padding for vision and depth estimation are separate
FPS = 60
depthEstimationPeriod = 1 # We sample the depth once every 3 frames
del_t = depthEstimationPeriod/FPS

def find_best_circles(circles_info, H, W, dist_tolerance=80):
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

def ball_within_bounds(H_full, W_full, ch, cw, center=None, padding=padding):
    """Remember that center is (height, width)
    The function determines whether the center of the ball is within the padding of the image.
    This should be used for depth estimation only, vision has its own padding system."""
    center_h, center_w = map(int, center)
    h0 = center_h - ch // 2
    w0 = center_w - cw // 2

    return (padding <= h0 <= H_full - padding - ch) and (padding <= w0 <= W_full - padding - cw)

"""
Steps we must take:
1. Initialize the appropriate objects
2. Load the images and rectify them!
"""

# ========= USER SETTINGS =========
# video_path    = "C:/Users/HP/Pictures/Camera Roll/WIN_20251128_20_00_11_Pro.mp4"
video_path    = "C:/Users/HP/Pictures/Camera Roll/WIN_20251228_22_16_48_Pro.mp4"
out_video     = "track2D/overlay13.mp4"
out_csv       = "track2D/track13.csv"
NPZ_PATH      = "D:/Catch/stereoExperiment/stereo_params.npz"
K_TXT_FULL    = "K.txt"   # for the FULL rectified frame
SCALE         = 0.5 # Pick 0.5 or 0.25, or at your own expense
SAVE_DIR      = "track2D"

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

# Some OS humdrum
os.makedirs(SAVE_DIR, exist_ok=True)
if os.path.exists(out_csv): # Overwrite the old csv file
    os.remove(out_csv)
csvfile = open(out_csv, "w", newline="")
writer = csv.writer(csvfile)
writer.writerow(["frame", "ball1_x", "ball1_y", "ball1_r", "ball1_u", "ball2_x", "ball2_y", "ball2_r", "ball2_u",
                 "ball3_x", "ball3_y", "ball3_r", "ball3_u"])  # header

# Instantiating all objects
B = 3
F = 7
tracker_2D = tracker.Tracker2D(B=B,F=F,dispTolerance=0.1,radiusTolerance=int(16*SCALE))
tracker_3D = tracker.Tracker3D(B=B, dispTolerance=0.2, angleTolerance=np.pi/2)
eyes_2D = vision.Vision2D(H=H_full, W=W_full, B=B)
eyes_3D = vision.Vision3D(H_full=H_full, W_full=W_full, K_path=K_TXT_FULL, engine_path=ENGINE_PATH)

frame_id = 0
t0 = time.time()

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break

    left_full, right_full = split_sbs(frame_bgr) # Full scale images
    rectL = cv2.remap(left_full, mapLx, mapLy, cv2.INTER_LINEAR) # Rectification
    rectR = cv2.remap(right_full, mapRx, mapRy, cv2.INTER_LINEAR) # Rectification
    # Downscaling for eyes_2D
    rectL_half = cv2.resize(rectL, (int(W_full * SCALE), int(H_full * SCALE)), interpolation=cv2.INTER_AREA)

    # We use Vision2D to find the centroids and centers in rectL_half
    found_centroids, centroids, (H_centroids, W_centroids) = eyes_2D.find_centroids_hsv(rectL_half)
    if found_centroids: # Non-empty centroids detected
        best_circles = find_best_circles(centroids, H_centroids, W_centroids, dist_tolerance=int(SCALE*160))
        tracker_2D.update(best_circles)
    else:
        best_circles, (H_circles, W_circles) = eyes_2D.find_circles_hough(rectL_half)
        # We now update our system via tracker_2D
        tracker_2D.update(best_circles)

    # We write to the csv file now
    line = [frame_id]
    for ball in tracker_2D.balls:
        if ball is None:
            line += [None, None, None, None]
        else:
            line += [ball.position[1], ball.position[0], ball.radius, ball.updated]
    writer.writerow(line)

    # Frame annotation
    for i, ball in enumerate(tracker_2D.balls):
        if ball is None:
            tracker_3D.balls[i] = None # We delete whatever 3D ball that was there
            continue
        (center_h, center_w) = ball.position
        if not (ball_within_bounds(H_full, W_full, CROP_H, CROP_W,(center_h * H_full,center_w * W_full))
            and ball.confirmed_ball and ball.updated):  # Remember to batch!
            continue
        cv2.circle(rectL_half, center=(int(W_centroids * ball.position[1]), int(H_centroids * ball.position[0])),
                radius=ball.radius, color=(0, 255, 255), thickness=2) # Draw a yellow circle

        # Depth estimation, we should only perform once every three frames
        if frame_id % depthEstimationPeriod != 0:
            continue
        # left_full_rgb = cv2.cvtColor(rectL, cv2.COLOR_BGR2RGB)  # For feeding into s2m2
        # right_full_rgb = cv2.cvtColor(rectR, cv2.COLOR_BGR2RGB)  # For feeding into s2m2
        corner_h_crop, corner_w_crop = int(center_h*H_full - CROP_H//2), int(center_w*W_full - CROP_W//2)
        # Lc = left_full_rgb[corner_h_crop:corner_h_crop+CROP_H, corner_w_crop:corner_w_crop+CROP_W].copy()
        # Rc = right_full_rgb[corner_h_crop:corner_h_crop+CROP_H, corner_w_crop:corner_w_crop+CROP_W].copy()
        Lc = rectL[corner_h_crop:corner_h_crop + CROP_H, corner_w_crop:corner_w_crop + CROP_W].copy()
        Rc = rectR[corner_h_crop:corner_h_crop + CROP_H, corner_w_crop:corner_w_crop + CROP_W].copy()
        (X, Y, depth) = eyes_3D.estimate_position(ball, Lc, Rc) # We have the observed coordinates
        pos3D = np.array([X, Y, depth])
        # We predict where the ball will be, then update given the new observation
        posStar3D = tracker_3D.predict_next_position(tracker_3D.balls[i], del_t)
        tracker_3D.update(i, tracker_3D.balls[i], pos3D, del_t)
        if posStar3D is not None:
            for j, (XStar, YStar, depthStar) in enumerate(posStar3D):
                # (xStar, yStar, depthStar) = posStar3D
                # Makeshift visualization code
                center_xStar = int((XStar*fx/depthStar + cx)/2)
                center_yStar = int((YStar*fy/depthStar + cy)/2)
                cv2.circle(rectL_half, center=(center_xStar, center_yStar),
                           radius=ball.radius, color=(min(20*j,255), 0, min(20*j, 255)), thickness=2)  # Draw a purple circle
        cv2.putText(rectL_half, f"({X}, {Y}, {depth})",
                    (int(W_centroids * ball.position[1]), int(H_centroids * ball.position[0])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,0,0),
                    1,
                    2)

    out.write(rectL_half)
    frame_id += 1

cap.release()
out.release()
csvfile.close()

t1 = time.time()
print(f"Time taken for video with {frame_id} frames: {t1 - t0} seconds")


# Things to do:
# 1. Simulate depthEstimationPeriod = 1
# 2. Use alpha, beta and gamma instead of 1
# 3. Show the prediction made depthEstimationPeriod ago