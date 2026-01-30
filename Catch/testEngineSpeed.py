"""Given the path to an engine and a video, we will see the average number of frames the engine can process per second"""
import vision
from ball import Ball2D
import cv2
import time
import numpy as np
from configVariables import *
video_path    = "C:/Users/HP/Pictures/Camera Roll/WIN_20251128_19_59_42_Pro.mp4"
NPZ_PATH      = "D:/Catch/stereoExperiment/stereo_params.npz"
K_TXT         = "K.txt"   # for the FULL rectified frame

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

def split_sbs(frame):
    h, w = frame.shape[:2]
    mid = w // 2
    return frame[:, :mid], frame[:, mid:]

t0 = time.time()
eyes_3D = vision.Vision3D(H_full=H_full, W_full=W_full, K_path=K_TXT, engine_path=ENGINE_PATH)
ball = Ball2D([0.5,0.5], np.array([0, 0]), 6, 6, 0)

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break

    left_full, right_full = split_sbs(frame_bgr) # Full scale images
    rectL = cv2.remap(left_full, mapLx, mapLy, cv2.INTER_LINEAR) # Rectification
    rectR = cv2.remap(right_full, mapRx, mapRy, cv2.INTER_LINEAR) # Rectification

    Lc = rectL[600:600 + CROP_H_3D, 600:600 + CROP_W_3D].copy()
    Rc = rectR[600:600 + CROP_H_3D, 600:600 + CROP_W_3D].copy()
    (X, Y, depth) = eyes_3D.estimate_position(ball, Lc, Rc, [112,112])  # We have the observed coordinates
    t1 = time.time()
    print(1/(t1 - t0), (X, Y, depth))
    t0 = t1