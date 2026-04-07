# vision.py

from config import CONFIG
import os
import cv2
import numpy as np
import torch
import tensorrt as trt

"""
A Vision2D object takes in an image, then performs the following:
1. Runs HSV masking
2. Morphologies
3. Connected components and centroids
5. Output all centroids, radii of tennis balls
"""

class Vision2D:
    def __init__(self, H, W, B, LOWER, UPPER, LOWERR, UPPERR, minradius, maxradius, ksize):
        self.H = H # Height of the full-scale image
        self.W = W # Width of the full-scale image
        self.B = B
        self.minradius = minradius # With respect to the full-scale image
        self.maxradius = maxradius

        # These are specific to our project
        # These are for color masking
        self.LOWER = np.array(LOWER, dtype=np.uint8)
        self.UPPER = np.array(UPPER, dtype=np.uint8)
        self.LOWERR = np.array(LOWERR, dtype=np.uint8)
        self.UPPERR = np.array(UPPERR, dtype=np.uint8)
        self.morphologyKernel = np.ones((ksize, ksize), dtype=np.uint8)

    def find_centroids_hsv(self, left_img):
        """
        :param left_img: The rectified left image of the stereo pair in BGR format.
                Ideally, left_img has scale 0.5 or even 0.25 to reduce computation time.
        :return: A list of centroids, each centroid is a tuple of (centroid_h, centroid_w, radius) and (H, W)
        """
        H, W = left_img.shape[:2]
        currentScale = H/self.H
        hsv = cv2.cvtColor(left_img, cv2.COLOR_BGR2HSV) # Color conversion to HSV
        mask_raw = cv2.inRange(hsv, self.LOWER, self.UPPER) # Mask production using the HSV

        # MorphologyEx
        mask = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, self.morphologyKernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morphologyKernel)

        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num <= 1: # We only have the background, no need to proceed further
            return []

        # We try to vectorize earlier logic for maximum efficiency
        stats_foreground = stats[1:]  # Index 0 is always the background, which is never a ball
        centroids_foreground = centroids[1:]
        widths = stats_foreground[:, cv2.CC_STAT_WIDTH]
        heights = stats_foreground[:, cv2.CC_STAT_HEIGHT]
        areas = stats_foreground[:, cv2.CC_STAT_AREA]
        radii = np.maximum(widths, heights) // 2 # Vectorized radii according to our definition

        minRadiusScale, maxRadiusScale = self.minradius * currentScale, self.maxradius * currentScale
        valid_radius = (radii >= minRadiusScale) & (radii <= maxRadiusScale)
        theoretical_area = np.pi * radii ** 2
        valid_area = areas > (theoretical_area / 3)
        valid_mask = valid_area & valid_radius
        idxs = np.flatnonzero(valid_mask)

        if idxs.size == 0:
            return []

        centroids_info = []
        for i in idxs:
            radius = int(radii[i])
            centroid_w_pixel = int(centroids_foreground[i, 0])
            centroid_h_pixel = int(centroids_foreground[i, 1])

            # We now check whether the centroid is valid or likely a yellow wall
            step = int(min(self.maxradius*currentScale, 2*radius))
            rows = np.array([centroid_h_pixel - step, centroid_h_pixel, centroid_h_pixel + step], dtype=np.int32)
            cols = np.array([centroid_w_pixel - step, centroid_w_pixel, centroid_w_pixel + step], dtype=np.int32)
            valid_rows = (rows >= 0) & (rows < H)
            valid_cols = (cols >= 0) & (cols < W)
            rows_v = rows[valid_rows]
            cols_v = cols[valid_cols]
            patch = hsv[np.ix_(rows_v, cols_v)]
            in_range = np.all((patch >= self.LOWERR) & (patch <= self.UPPERR), axis=-1)
            count_in_range = np.count_nonzero(in_range)
            if count_in_range == 1:
                centroid_w, centroid_h = centroid_w_pixel / W, centroid_h_pixel / H
                centroids_info.append([centroid_h, centroid_w, radius])
        centroids_info = sorted(centroids_info, key=lambda x: x[-1], reverse=True)  # We reverse-sort by centroids by radius
        return centroids_info[:self.B]  # We only return up to self.B number of centroids

    @staticmethod
    def find_best_circles(circles_info, H, W, dist_tolerance=40 * CONFIG.runtime.scale):
        """
        :param circles_info: A list where each element has the form (circle_h, circle_w, radius) and is sorted in
                decreasing radius. circle_h and circle_w are normalized in [0, 1].
        :param H: The height of the image that the circle finder in vision operates under.
        :param W: The width of the image that the circle finder in vision operates under.
        :param dist_tolerance: The minimum distance between circles.
        :return: We return a cleaned version of circles_info in descending radius.
        It is worth noting that the function operates under scale=SCALE.
        """
        best_circles_info = []  # Returns elements of type (center_h, center_w, radius) in descending radius
        centers_seen = []
        if circles_info:  # Handling the base case where centers_seen is still empty
            (center_h, center_w, radius) = circles_info.pop(0)
            best_circles_info.append([center_h, center_w, radius])
            centers_seen.append(np.array([center_h * H, center_w * W]))

        while circles_info:
            (center_h, center_w, radius) = circles_info.pop(0)
            center = np.array([center_h * H, center_w * W])
            min_dist_squared = min([np.inner(center - centroid, center - centroid) for centroid in centers_seen])
            if min_dist_squared >= dist_tolerance ** 2:
                best_circles_info.append([center_h, center_w, radius])
                centers_seen.append(center)
        return best_circles_info  # Already sorted by design!

    @staticmethod
    def ball_within_bounds(H_full, W_full, center, crop_h, crop_w, padding):
        """
        :param H_full: The height of the full-resolution image
        :param W_full: The width of the full-resolution image
        :param center: (height, width) of the ball center in pixels
        :param crop_h: The height of the cropped image
        :param crop_w: The width of the cropped image
        :param padding: This prevents the ball centers from being too close to the edges and corners
        :return: (bool of whether the ball is within bounds, the pixel coordinates of the top left corner of crop box,
                and the pixel coordinates of the ball center within the crop)
        """

        center_h, center_w = map(int, center)
        withinBound = (padding <= center_h <= H_full - padding) and (padding <= center_w <= W_full - padding)
        if not withinBound:
            return False, None, None

        (hTL, wTL) = center_h - crop_h // 2, center_w - crop_w // 2  # Actual pixel coordinates of the top-left corner
        (hBR, wBR) = hTL + crop_h, wTL + crop_w
        if hTL < 0:  # We have the top-left corner too high
            hBR = hBR - hTL
            hTL = 0  # hTL = hTL - hTL
        elif hBR > H_full:  # We have the bottom-left corner too low
            hTL = hTL - hBR + H_full
            hBR = H_full  # hBR = hBR - hBR + H_full
        if wTL < 0:  # We have the top-left corner too much to the left
            wBR = wBR - wTL
            wTL = 0  # wTL = wTL - wTL
        elif wBR > W_full:  # We have the bottom-right corner too much to the right
            wTL = wTL - wBR + W_full
            wBR = W_full  # wBR = wBR - wBR + W_full

        relCenterH, relCenterW = (center_h - hTL, center_w - wTL)  # For the depth estimation in Vision3D
        return True, (hTL, wTL), (relCenterH, relCenterW)


"""Vision3D is responsible for the mechanics of depth estimation"""

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision('high')

def load_engine(path: str) -> trt.ICudaEngine:
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    with open(path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

def trt_dtype_to_torch(np_dtype):
    """Map TensorRT data type (via trt.nptype) to a torch.dtype."""
    if np_dtype == np.float16:
        return torch.float16
    if np_dtype == np.float32:
        return torch.float32
    if np_dtype == np.int32:
        return torch.int32
    if np_dtype == np.uint8:
        return torch.uint8
    raise TypeError(f"Unsupported dtype from TensorRT: {np_dtype}")

def allocate_tensors(engine, max_batch=1):
    tensors = {} # A dictionary of name_of_tensor: actual_tensor_memory
    bindings = [0] * engine.num_io_tensors # A list of pointers to the actual_tensor_memory

    for idx in range(engine.num_io_tensors):
        name = engine[idx]
        mode = engine.get_tensor_mode(name)
        np_dtype = trt.nptype(engine.get_tensor_dtype(name))
        torch_dtype = trt_dtype_to_torch(np_dtype)

        if mode == trt.TensorIOMode.INPUT:
            shape = (max_batch, 3, CONFIG.crop.crop_h_3d, CONFIG.crop.crop_w_3d)
        else:
            # Ask engine for static output shape ignoring batch, then prepend max_batch
            out_shape = list(engine.get_tensor_shape(name))
            out_shape[0] = max_batch
            shape = tuple(out_shape)

        t = torch.empty(*shape, device="cuda", dtype=torch_dtype)
        tensors[name] = t
        bindings[idx] = int(t.data_ptr())

    return bindings, tensors


class Vision3D:
    def __init__(self, H_full, W_full, K_path, engine_path):
        """Reads the intrinsics file"""
        self.H_full = H_full
        self.W_full = W_full

        with open(K_path, "r") as f:
            l1 = f.readline().strip()
            l2 = f.readline().strip()
        vals = list(map(float, l1.split()))
        fx, _, cx, _, fy, cy, _, _, _ = vals
        B = float(l2)
        self.fx = fx
        self.cx = cx
        self.fy = fy
        self.cy = cy
        self.B = B
        # TRT engine
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.engine = load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.bindings, self.tensors = allocate_tensors(self.engine, 1)
        self.context.set_input_shape("input_left", (1, 3, CONFIG.crop.crop_h_3d, CONFIG.crop.crop_w_3d))
        self.context.set_input_shape("input_right", (1, 3, CONFIG.crop.crop_h_3d, CONFIG.crop.crop_w_3d))

    def normalized_to_meter_x(self, normalized_x, depth):
        """Converts normalized x coordinates to pixel coordinates"""
        return (self.W_full * normalized_x - self.cx) / self.fx * depth

    def normalized_to_meter_y(self, normalized_y, depth):
        """Converts normalized y coordinates to pixel coordinates"""
        return (self.H_full * normalized_y - self.cy) / self.fy * depth


    def estimate_position(self, ball_h, ball_w, ball_r, Lc, Rc, relCenter):
        """Estimates the depth of the ball given its relative position in the frame"""
        # Lc = cv2.cvtColor(Lc, cv2.COLOR_BGR2RGB)
        # Rc = cv2.cvtColor(Rc, cv2.COLOR_BGR2RGB)

        left_torch = torch.from_numpy(Lc).permute(2, 0, 1).unsqueeze(0).to(device=self.device, dtype=torch.uint8)
        right_torch = torch.from_numpy(Rc).permute(2, 0, 1).unsqueeze(0).to(device=self.device, dtype=torch.uint8)

        self.tensors["input_left"][:1].copy_(left_torch, non_blocking=True) # We fill the space in tensors with our real inputs
        self.tensors["input_right"][:1].copy_(right_torch, non_blocking=True)

        self.context.execute_v2(bindings=self.bindings)

        # Debugging
        # print("row abs diff mean:", np.mean(np.abs(Lc - Rc)))
        #
        # print("output_disp shape:", tuple(disp.shape),
        #       "nan count:", int(torch.isnan(disp).sum().item()),
        #       "min:", float(torch.nan_to_num(disp, nan=0.0).min().item()),
        #       "max:", float(torch.nan_to_num(disp, nan=0.0).max().item()))

        # End of debugging

        disp_map = self.tensors["output_disp"][0, 0]
        relCenterH, relCenterW = relCenter
        i0 = max(0, relCenterH - ball_r)
        i1 = min(CONFIG.crop.crop_h_3d, relCenterH + ball_r + 1)
        j0 = max(0, relCenterW - ball_r)
        j1 = min(CONFIG.crop.crop_w_3d, relCenterW + ball_r + 1)
        patch = disp_map[i0:i1, j0:j1]
        valid = torch.isfinite(patch) & (patch > 1e-6)

        if not torch.any(valid):
            return (np.nan, np.nan, np.nan)

        depth_patch = (self.fx * self.B) / patch[valid]
        depth = torch.median(depth_patch).item()

        # disp_map_np = disp_map.cpu().numpy()[0, 0].astype(float)
        #
        # patch = disp_map_np[i0:i1, j0:j1]
        #
        # valid = np.isfinite(patch) & (patch > 1e-6)
        # if not np.any(valid):
        #     return (np.nan, np.nan, np.nan)
        #
        # depth_patch = (self.fx * self.B) / patch[valid]  # 1D
        # depth = np.median(depth_patch)  # no NaNs now

        X = (self.W_full * ball_w - self.cx) / self.fx * depth
        Y = (self.H_full * ball_h - self.cy) / self.fy * depth
        return (X, Y, depth)

    def warmup(self, num_iters: int = 3):
        """
        Run a few dummy inferences so that the first real inference is fast.
        """
        dummy_left = torch.zeros((1, 3, CONFIG.crop.crop_h_3d, CONFIG.crop.crop_w_3d), device=self.device, dtype=torch.uint8)
        dummy_right = torch.zeros((1, 3, CONFIG.crop.crop_h_3d, CONFIG.crop.crop_w_3d), device=self.device, dtype=torch.uint8)

        for _ in range(num_iters):
            self.tensors["input_left"][:1].copy_(dummy_left, non_blocking=True)
            self.tensors["input_right"][:1].copy_(dummy_right, non_blocking=True)
            self.context.execute_v2(bindings=self.bindings)

            # Touch output so work is definitely realized
            _ = self.tensors["output_disp"][0, 0, 0, 0].item()
