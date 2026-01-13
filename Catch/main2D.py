import numpy as np
import tracker
import vision
import cv2
import os
import csv
import time

def find_best_circles(circles_info, H, W, dist_tolerance=80):
    # dist_tolerance is the number of pixels between centers and seen centroid centers for them to be considered the
    # centers for the same circle
    best_circles_info = []
    centers_seen = []
    while circles_info:
        (center_h, center_w, radius) = circles_info.pop(0)
        if not centers_seen:  # We should add the circle unconditionally
            best_circles_info.append([center_h, center_w, radius])
            centers_seen.append(np.array([center_h * H, center_w * W]))
            continue
        center = np.array([center_h * H, center_w * W])
        min_dist = min([np.linalg.norm(center - centroid) for centroid in centers_seen])
        if min_dist >= dist_tolerance:
            best_circles_info.append([center_h, center_w, radius])
            centers_seen.append(center)
    return best_circles_info # Already sorted by design!

# def find_best_circles(centroids_info, circles_info, N, H, W, dist_tolerance=80):
#     """Given the centroids and centers found by color masking and the Hough transform respectively,
#     we want to find the finalized list of at most N potential circles in the following way:
#     1. We loop through the centroids with the largest radii, we accept these automatically
#     2. We loop through the centers with the largest radii that are sufficiently far from centers we've already seen
#         and accept these.
#     dist_tolerance is the number of pixels between centers and seen centroid centers for them to be considered the
#         centers for the same circle
#     """
#     best_circles_info = []
#     centers_seen = []
#     while len(best_circles_info) < N:
#         if centroids_info:
#             (centroid_h, centroid_w, radius) = centroids_info.pop(0)
#             if not centers_seen:  # We should add the circle unconditionally
#                 best_circles_info.append([centroid_h, centroid_w, radius])
#                 centers_seen.append(np.array([centroid_h * H, centroid_w * W]))
#                 continue
#             center = np.array([centroid_h * H, centroid_w * W])
#             min_dist = min([np.linalg.norm(center - centroid) for centroid in centers_seen])
#             if min_dist >= dist_tolerance:
#                 best_circles_info.append([centroid_h, centroid_w, radius])
#             continue
#         if circles_info:
#             (center_h, center_w, radius) = circles_info.pop(0)
#             if not centers_seen: # We should add the circle unconditionally
#                 best_circles_info.append([center_h, center_w, radius])
#                 continue
#             center = np.array([center_h * H, center_w * W])
#             min_dist = min([np.linalg.norm(center - centroid) for centroid in centers_seen])
#             if min_dist >= dist_tolerance:
#                 best_circles_info.append([center_h, center_w, radius])
#             continue
#         break
#
#     best_circles_info = sorted(best_circles_info, key=lambda x: x[-1], reverse=True) # Reverse sort by radius
#     return best_circles_info

def split_sbs(frame):
    h, w = frame.shape[:2]
    mid = w // 2
    return frame[:, :mid], frame[:, mid:]


"""
Steps we must take:
1. Initialize the appropriate objects
2. Load the images and rectify them!
"""

# ========= USER SETTINGS =========
video_path    = "C:/Users/HP/Pictures/Camera Roll/WIN_20251128_10_15_25_Pro.mp4"
out_video     = "track2D/overlay4.mp4"
out_csv       = "track2D/track4.csv"
NPZ_PATH      = "D:/Catch/stereoExperiment/stereo_params.npz"
SCALE         = 0.5 # Pick 0.5 or 0.25, or at your own expense
SAVE_DIR      = "track2D"

# =================================

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
tracker_2D = tracker.Tracker2D(B=B,F=F,dispTolerance=0.08,radiusTolerance=int(16*SCALE))
eyes_2D = vision.Vision2D(H=H_full, W=W_full, B=B)

frame_id = 0
t0 = time.time()

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break

    left_full, right_full = split_sbs(frame_bgr) # Full scale images
    rectL = cv2.remap(left_full, mapLx, mapLy, cv2.INTER_LINEAR) # Rectification
    rectR = cv2.remap(right_full, mapRx, mapRy, cv2.INTER_LINEAR) # Rectification
    H, W = rectL.shape[:2]
    # Downscaling for eyes_2D
    rectL_half = cv2.resize(rectL, (int(W * SCALE), int(H * SCALE)), interpolation=cv2.INTER_AREA)

    # We use Vision2D to find the centroids and centers in rectL_half
    found_centroids, centroids, (H_centroids, W_centroids) = eyes_2D.find_centroids_hsv(rectL_half)
    if found_centroids: # Non-empty centroids detected
        best_circles = find_best_circles(centroids, H_centroids, W_centroids, dist_tolerance=int(SCALE*160))
        tracker_2D.update(best_circles)
    else:
        circles, (H_circles, W_circles) = eyes_2D.find_circles_hough(rectL_half)
        # We now update our system via tracker_2D
        tracker_2D.update(circles)

    # We write to the csv file now
    line = [frame_id]
    for ball in tracker_2D.balls:
        if ball is None:
            line += [None, None, None, None]
            continue
        line += [ball.position[1], ball.position[0], ball.radius, ball.updated]
        # Drawing the circle onto our output video frame
        if ball.confirmed_ball:
            cv2.circle(rectL_half, center=(int(W_centroids*ball.position[1]), int(H_centroids*ball.position[0])),
                       radius=ball.radius, color=(0, 255, 255), thickness=2)
    writer.writerow(line)
    out.write(rectL_half)
    frame_id += 1

cap.release()
out.release()
csvfile.close()

t1 = time.time()
print(f"Time taken for video with {frame_id} frames: {t1 - t0} seconds")