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
4. Hough transform
5. Output all centroids, radii of tennis balls
"""

class Vision2D:
    def __init__(self, H, W, B, minradius=2, maxradius=64, padding=0):
        self.H = H # Height of the full-scale image
        self.W = W # Width of the full-scale image
        self.B = B
        self.minradius = minradius # With respect to the full-scale image
        self.maxradius = maxradius
        self.padding = padding # With respect to the full-scale image

        # These are specific to our project
        # These are for color masking
        hl, hh, sl, sh, vl, vh = (25, 41, 85, 255, 80, 255) # Color of the tennis ball
        self.LOWER = np.array([hl, sl, vl], dtype=np.uint8)
        self.UPPER = np.array([hh, sh, vh], dtype=np.uint8)
        self.LOWERR = np.array([hl-3, sl, vl], dtype=np.uint8)
        self.UPPERR = np.array([hh+3, sh, vh], dtype=np.uint8)

        # These are parameters for the Hough transform
        self.DP = 1.2  # inverse accumulator ratio
        self.MIN_DIST = 160  # minimum distance between detected centers (in pixels)
        self.PARAM1 = 100  # Canny high threshold
        self.PARAM2 = 50  # accumulator threshold for center detection

    @staticmethod
    def generate_web(centroid_h, centroid_w, step, H, W):
        """Checks with neighboring pixels whether the current centroid is a tennis ball or a yellow wall"""
        offsets = [-step, 0, step]
        points = []

        for di in offsets:
            for dj in offsets:
                ni = centroid_h + di
                nj = centroid_w + dj
                if 0 <= ni < H and 0 <= nj < W:
                    points.append((ni, nj))
        return points

    def find_centroids_hsv(self, left_img):
        """left_img must be the rectified left image of the stereo pair in BGR order.
        Ideally, left_img has scale 0.5 to reduce computation time.
        We return a list of centroids, each centroid is a tuple of (centroid_h, centroid_w, radius) and (H, W)"""
        H, W = left_img.shape[:2]
        scale = H/self.H # Scale of the downscaled left_img
        found_centroids = False
        hsv = cv2.cvtColor(left_img, cv2.COLOR_BGR2HSV) # Color conversion to HSV
        mask_raw = cv2.inRange(hsv, self.LOWER, self.UPPER) # Mask production using the HSV

        # MorphologyEx
        if (H//180)%2 == 0:
            ksize = H//180 + 1 # This must be an odd number
        else:
            ksize = H//180 # This works in practice we found
        kernel = np.ones((ksize, ksize), np.uint8)
        mask = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        centroids_info = []
        for i in range(num):
            # area = stats[i, cv2.CC_STAT_AREA]
            # We estimate and check the radius
            radius = max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) // 2 # w.r.t. scaled down image
            if radius > self.maxradius*scale or radius < self.minradius*scale:
                continue
            # Area check
            theoreticalArea = np.pi*radius**2
            # print(radius, stats[i, cv2.CC_STAT_AREA])
            if stats[i, cv2.CC_STAT_AREA] <= theoreticalArea/2:
                continue
            found_centroids = True  # There exist centroids in the frame that are reasonably sized
            centroid_w_pixel = centroids[i, 0]
            centroid_h_pixel = centroids[i, 1]
            # if not ((self.padding*scale <= centroid_w_pixel <= W - self.padding*scale) and
            #         (self.padding*scale <= centroid_h_pixel <= H - self.padding*scale)): # Vision padding
            #     continue

            # We now check whether the centroid is valid or likely a yellow wall
            color_web = [hsv[pi, pj] for (pi, pj) in self.generate_web(int(centroid_h_pixel), int(centroid_w_pixel),
                                                                        int(min(self.maxradius*scale, 1.8*radius)), H, W)]
            count_in_range = sum(
                np.all(self.LOWERR <= color) and np.all(color <= self.UPPERR) for color in color_web
            )
            if count_in_range == 1:
                centroid_w, centroid_h = centroid_w_pixel / W, centroid_h_pixel / H
                centroids_info.append([centroid_h, centroid_w, radius])
        centroids_info = sorted(centroids_info, key=lambda x: x[-1], reverse=True) # We reverse-sort by centroids by radius
        return found_centroids, centroids_info[:self.B], (H, W) # We only return up to self.B number of centroids


    def find_circles_hough(self, left_img):
        H, W = left_img.shape[:2]
        return [], (H, W)
    # def find_circles_hough(self, left_img):
    #     """left_img must be the rectified left image of the stereo pair in BGR order.
    #     Ideally, left_img has scale 0.5 to reduce computation time.
    #     We return a list of centers, each centers is a tuple of (center_h, center_w, radius) and (H, W)"""
    #     H, W = left_img.shape[:2]
    #     scale = H/self.H
    #     # green = cv2.split(left_img)[1] # Green channel
    #     gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY) # Color conversion to gray
    #     # hsv = cv2.cvtColor(left_img, cv2.COLOR_BGR2HSV)
    #     # hue = hsv[:,:,0]
    #     if (H//120)%2 == 0:
    #         ksize = H//120 + 1 # Must be an odd number
    #     else:
    #         ksize = H//120 # This works in practice we found
    #     gray_blur = cv2.medianBlur(gray, ksize) # A Gaussian blur to prepare for the Hough transform
    #
    #     # The Hough transform
    #     circles = cv2.HoughCircles(
    #         gray_blur,
    #         cv2.HOUGH_GRADIENT,
    #         dp=self.DP,
    #         minDist=int(self.MIN_DIST*scale),
    #         param1=self.PARAM1,
    #         param2=self.PARAM2,
    #         minRadius=int(self.minradius*scale),
    #         maxRadius=int(self.maxradius*scale)
    #     )
    #
    #     circles_info = []
    #     if circles is not None:
    #         circles = circles[0, :]
    #         for (x, y, r) in circles:
    #             # if not ((self.padding * scale <= x <= W - self.padding * scale) and
    #             #         (self.padding * scale <= y <= H - self.padding * scale)):  # Vision padding
    #             #     continue
    #             # if self.LOWERR <= hsv[y, x] <= self.UPPERR:
    #             circles_info.append([y/H, x/W, int(r)])
    #     circles_info = sorted(circles_info, key=lambda x: x[-1], reverse=True)
    #     return circles_info[:self.B], (H, W) # We only return up to self.B number of circles



"""Vision3D is responsible for the mechanics of depth estimation"""

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision('high')
CROP_W, CROP_H = 224, 224 # For feeding into the engine

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

def allocate_tensors(engine, max_batch=3):
    tensors = {} # A dictionary of name_of_tensor: actual_tensor_memory
    bindings = [0] * engine.num_io_tensors # A list of pointers to the actual_tensor_memory

    for idx in range(engine.num_io_tensors):
        name = engine[idx]
        mode = engine.get_tensor_mode(name)
        np_dtype = trt.nptype(engine.get_tensor_dtype(name))
        torch_dtype = trt_dtype_to_torch(np_dtype)

        if mode == trt.TensorIOMode.INPUT:
            shape = (max_batch, 3, CROP_H, CROP_W)
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
        self.bindings, self.tensors = allocate_tensors(self.engine, 3)

    @staticmethod
    def depth_adjust(depth, x, y):
        """Adjusts the depth of the ball given its relative position in the frame"""
        dx = abs(x - 0.5)
        dy = abs(y - 0.5)
        denominator = 1.5825 - 0.5241 * dx - 0.2993 * dy - 2.2004 * dx ** 2 - 1.4422 * dy ** 2
        return depth * (1.5825 / denominator)

    def normalized_to_meter_x(self, normalized_x, depth):
        """Converts normalized x coordinates to pixel coordinates"""
        return (self.W_full * normalized_x - self.cx) / self.fx * depth

    def normalized_to_meter_y(self, normalized_y, depth):
        """Converts normalized y coordinates to pixel coordinates"""
        return (self.H_full * normalized_y - self.cy) / self.fy * depth


    def estimate_position(self, ball, Lc, Rc, relCenter):
        """Estimates the depth of the ball given its relative position in the frame"""
        left_torch = (torch.from_numpy(Lc).permute(-1, 0, 1).unsqueeze(0)).half().to(self.device)
        right_torch = (torch.from_numpy(Rc).permute(-1, 0, 1).unsqueeze(0)).half().to(self.device)

        self.tensors["input_left"][:1].copy_(left_torch)  # We fill the space in tensors with our real inputs
        self.tensors["input_right"][:1].copy_(right_torch)

        self.context.set_input_shape("input_left",
                                (1, 3, CROP_H, CROP_W))  # We don't need to specify for every input, but why not!
        self.context.set_input_shape("input_right", (1, 3, CROP_H, CROP_W))

        self.context.execute_v2(bindings=self.bindings)
        disp_map = self.tensors["output_disp"]
        torch.cuda.synchronize()

        disp_map_np = disp_map.cpu().numpy()[0, 0].astype(float)
        depths = []
        relCenterH, relCenterW = relCenter
        for i in range(relCenterH - ball.radius // 2, relCenterH + ball.radius // 2):
            for j in range(relCenterW - ball.radius // 2, relCenterW + ball.radius // 2):
                if disp_map_np[i, j] == 0:
                    continue
                depths.append(self.fx * self.B / disp_map_np[i, j])

        depth = np.median(depths)
        X = (self.W_full * ball.position[1] - self.cx) / self.fx * depth
        Y = (self.H_full * ball.position[0] - self.cy) / self.fy * depth
        return (X, Y, depth)
