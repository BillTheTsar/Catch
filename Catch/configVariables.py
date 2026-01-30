CROP_W_2D, CROP_H_2D = 400, 300 # For focusing on the ball
CROP_W_3D, CROP_H_3D = 224, 224 # For feeding into the engine
FPS = 60
depthEstimationPeriod = 3 # We sample the depth once every 3 frames
z_alpha, x_alpha, y_alpha = 0.8, 0.8, 0.8
cameraHeight = 0.86
SCALE = 0.5 # Pick 0.5 or 0.25, or at your own expense
ENGINE_PATH = "D:/S2M2/s2m2/engines/S2M2_L_224_224_fp16.engine"
