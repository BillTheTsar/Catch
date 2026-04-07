# annotator.py

import numpy as np
import tracker
import vision
import kalman
from ball import Ball2D, Ball3D
import cv2
import os
import csv
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from config import CONFIG
from internalState import InternalState, DepthJobStatus
from confidence import compute_confidence

def run(CONFIG):
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
                                                             dist_tolerance=int(CONFIG.runtime.scale * CONFIG.tracker2d.dist_tolerance))
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
                                                          dist_tolerance=int(CONFIG.runtime.scale*CONFIG.tracker2d.dist_tolerance))
        tracker2DLax.update(best_circles)
        return True

    def depthJob(frame_id, ballIndex, ball_h, ball_w, ball_r, Lc, Rc, relCenter):
        """
        :param frame_id: The frame at which we submit the depth estimation task
        :param ballIndex: The index of the ball in tracker_3D
        :param ball_h: The normalized height of the snapshot ball
        :param ball_w: The normalized width of the snapshot ball
        :param ball_r: The radius of the snapshot ball w.r.t. SCALE
        :param Lc: The rectified left crop
        :param Rc: The rectified right crop
        :param relCenter: The relative position of the ball center within the crop
        :return: frame_id (reiterated), ballIndex, pos3D=(X, Y, Z)
        """
        (X, Y, Z) = eyes3D.estimate_position(ball_h, ball_w, ball_r, Lc, Rc, relCenter)
        return frame_id, ballIndex, np.array([X, Y, Z], dtype=np.float64)

    def flush_ready_frames(force=False):
        nonlocal nextFrameToWrite, depthFuture, depthMeta

        while nextFrameToWrite in frameBuffer:
            if not force:
                # Do not flush the outstanding request frame (or anything after it)
                if depthFuture is not None and depthMeta is not None:
                    protected_frame = depthMeta["frameProtectStart"]
                    if nextFrameToWrite >= protected_frame:
                        break

                newestBuffered = max(frameBuffer.keys())
                if newestBuffered - nextFrameToWrite < CONFIG.annotator.max_buffer_size:
                    break

            out.write(frameBuffer.pop(nextFrameToWrite))
            nextFrameToWrite += 1

    def annotate_landing_circle(frame_idx, posStar3D, radius):
        if posStar3D is None or len(posStar3D) == 0:
            return
        if frame_idx not in frameBuffer:
            return

        XStar, YStar, depthStar = posStar3D
        if depthStar <= 0:
            return

        center_xStar = int((XStar * fx / depthStar + cx) / 2)
        center_yStar = int((YStar * fy / depthStar + cy) / 2)

        if (0 <= center_xStar <= CONFIG.runtime.scale * W_full) and (0 <= center_yStar <= CONFIG.runtime.scale * H_full):
            cv2.circle(
                frameBuffer[frame_idx],
                center=(center_xStar, center_yStar),
                radius=radius,
                color=CONFIG.annotator.landing_prediction_color,
                thickness=2
            )

    def add_state(state: InternalState):
        if len(state_history) == state_history.maxlen:
            oldest = state_history[0]
            state_by_frame.pop(oldest.frame_id, None)
        state_history.append(state)
        state_by_frame[state.frame_id] = state

    def cleanup(cap=None, out=None, csvfile=None, strictLaxPool=None, depthPool=None):
        if strictLaxPool is not None:
            strictLaxPool.shutdown(wait=True, cancel_futures=True)

        if depthPool is not None:
            depthPool.shutdown(wait=True, cancel_futures=True)

        if out is not None:
            out.release()

        if cap is not None:
            cap.release()

        if csvfile is not None:
            csvfile.close()

        cv2.destroyAllWindows()

    # =================================

    # The following lines guarantee cleanup safety and should be ignored otherwise.
    cap = None
    csvfile = None
    out = None
    strictLaxPool = None
    depthPool = None

    try:
        # K file
        with open(CONFIG.paths.k_txt, "r") as f:
            l1 = f.readline().strip()
            l2 = f.readline().strip()
        vals = list(map(float, l1.split()))
        fx, _, cx, _, fy, cy, _, _, _ = vals

        # NPZ
        params   = np.load(CONFIG.paths.npz_path, allow_pickle=True)
        mapLx, mapLy = params["mapLx"], params["mapLy"]
        mapRx, mapRy = params["mapRx"], params["mapRy"]
        mapLx_downscaled = mapLx[::2, ::2] * CONFIG.runtime.scale # These 4 downscaled maps are for ball detection in Vision2D
        mapLy_downscaled = mapLy[::2, ::2] * CONFIG.runtime.scale
        # mapRx_downscaled = mapRx[::2, ::2] * CONFIG.runtime.scale
        # mapRy_downscaled = mapRy[::2, ::2] * CONFIG.runtime.scale
        mapLx_crop_3D = np.empty((CONFIG.crop.crop_h_3d, CONFIG.crop.crop_w_3d), np.float32) # These 4 are for depth estimation
        mapLy_crop_3D = np.empty((CONFIG.crop.crop_h_3d, CONFIG.crop.crop_w_3d), np.float32)
        mapRx_crop_3D = np.empty((CONFIG.crop.crop_h_3d, CONFIG.crop.crop_w_3d), np.float32)
        mapRy_crop_3D = np.empty((CONFIG.crop.crop_h_3d, CONFIG.crop.crop_w_3d), np.float32)
        mapLx_crop_2D = np.empty((CONFIG.crop.crop_h_2d, CONFIG.crop.crop_w_2d), np.float32) # These 2 are for active tracking
        mapLy_crop_2D = np.empty((CONFIG.crop.crop_h_2d, CONFIG.crop.crop_w_2d), np.float32)

        # Opening video
        cap = cv2.VideoCapture(str(CONFIG.paths.video_path), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {str(CONFIG.paths.video_path)}")
        fps    = cap.get(cv2.CAP_PROP_FPS)
        W_full = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))//2
        H_full = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        H_centroids, W_centroids = H_full * CONFIG.runtime.scale, W_full * CONFIG.runtime.scale

        if CONFIG.runtime.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(CONFIG.paths.out_video_path), fourcc, fps//4, (int(W_full*CONFIG.runtime.scale), int(H_full*CONFIG.runtime.scale)))
            if not out.isOpened():
                cap.release()
                raise IOError(f"Could not open output VideoWriter: {str(CONFIG.paths.out_video_path)}")

        # State
        state_history = deque(maxlen=CONFIG.telemetry.state_history_len)
        state_by_frame = dict()

        # Instantiating all objects
        tracker2DStrict = tracker.Tracker2D(
            B=CONFIG.tracker2d.B,
            F=CONFIG.tracker2d.F,
            dispTolerance=CONFIG.tracker2d.disp_tolerance,
            radiusTolerance=int(CONFIG.tracker2d.radius_tolerance * CONFIG.runtime.scale))

        tracker2DLax = tracker.Tracker2D(
            B=CONFIG.tracker2d.B,
            F=CONFIG.tracker2d.F,
            dispTolerance=CONFIG.tracker2d.disp_tolerance,
            radiusTolerance=int(CONFIG.tracker2d.radius_tolerance * CONFIG.runtime.scale))

        strictLaxPool = ThreadPoolExecutor(max_workers=2)
        meta2DTracker = tracker.Meta2DTracker(CONFIG.tracker2d.B, CONFIG.tracker2d.B)

        tracker_3D = tracker.Tracker3D(
            B=1,
            dispTolerance=CONFIG.tracker3d.disp_tolerance,
            angleTolerance=CONFIG.tracker3d.angle_tolerance,
            minPPrimeLen=CONFIG.tracker3d.min_pPrime_len, )

        eyes2DStrict = vision.Vision2D(
            H=H_full,
            W=W_full,
            B=CONFIG.tracker2d.B,
            LOWER=CONFIG.vision2d_strict.LOWER,
            UPPER=CONFIG.vision2d_strict.UPPER,
            LOWERR=CONFIG.vision2d_strict.LOWERR,
            UPPERR=CONFIG.vision2d_strict.UPPERR,
            minradius=CONFIG.vision2d_strict.min_radius,
            maxradius=CONFIG.vision2d_strict.max_radius,
            ksize=CONFIG.vision2d_strict.ksize)

        eyes2DLax = vision.Vision2D(
            H=H_full,
            W=W_full,
            B=CONFIG.tracker2d.B,
            LOWER=CONFIG.vision2d_lax.LOWER,
            UPPER=CONFIG.vision2d_lax.UPPER,
            LOWERR=CONFIG.vision2d_lax.LOWERR,
            UPPERR=CONFIG.vision2d_lax.UPPERR,
            minradius=CONFIG.vision2d_lax.min_radius,
            maxradius=CONFIG.vision2d_lax.max_radius,
            ksize=CONFIG.vision2d_lax.ksize)

        eyes2DFocus = vision.Vision2D(
            H=CONFIG.crop.crop_h_2d,
            W=CONFIG.crop.crop_w_2d,
            B=1,
            LOWER=CONFIG.vision2d_focus.LOWER,
            UPPER=CONFIG.vision2d_focus.UPPER,
            LOWERR=CONFIG.vision2d_focus.LOWERR,
            UPPERR=CONFIG.vision2d_focus.UPPERR,
            minradius=CONFIG.vision2d_focus.min_radius,
            maxradius=CONFIG.vision2d_focus.max_radius,
            ksize=CONFIG.vision2d_focus.ksize)

        eyes3D = vision.Vision3D(
                H_full=H_full,
                W_full=W_full,
                K_path=CONFIG.paths.k_txt,
                engine_path=CONFIG.paths.engine_path)
        eyes3D.warmup(num_iters=3) # So actual inference is fast off the bat

        landingFilter = kalman.LandingKalmanFilter(
                process_var=CONFIG.kalman.process_var,
                measurement_var=CONFIG.kalman.measurement_var,)

        depthPool = ThreadPoolExecutor(max_workers=1) # 3D estimation
        depthFuture = None # Whether we have a job currently running or pending release
        continuousTracking = True # Whether there is a break in eyes2DFocus
        frameBuffer = {}
        nextFrameToWrite = 0


        # Some CSV humdrum
        os.makedirs(CONFIG.paths.save_dir, exist_ok=True)
        if os.path.exists(CONFIG.paths.out_csv_path): # Overwrite the old csv file
            os.remove(CONFIG.paths.out_csv_path)
        csvfile = open(CONFIG.paths.out_csv_path, "w", newline="")
        writer = csv.writer(csvfile)
        lengthPerEntry = 5
        headerRow = ["frame"]
        for i in range(1, CONFIG.tracker2d.B+1):
            headerRow += [f"ball{i}_x", f"ball{i}_y", f"ball{i}_r", f"ball{i}_u", f"ball{i}_c"]
        writer.writerow(headerRow)


        frame_id = 0
        frameLastReceived = 0 # The frame in which the last future was received.
        frameLastRequest = 0
        gapsForRescale = deque(maxlen=CONFIG.tracker3d.prediction_threshold)
        t0 = time.time()
        tst = time.time()

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            left_full, right_full = split_sbs(frame_bgr) # Full scale images
            left_downscaled = cv2.resize(left_full, (int(W_full * CONFIG.runtime.scale), int(H_full * CONFIG.runtime.scale)),
                                         interpolation=cv2.INTER_AREA)
            rectL_downscaled = cv2.remap(left_downscaled, mapLx_downscaled, mapLy_downscaled, cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT)
            cv2.putText(rectL_downscaled, f"{frame_id}",
                        CONFIG.annotator.font_position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        CONFIG.annotator.font_color,
                        1,
                        2)
            if tracker2DStrict.activelyTracking: # Tight crop but full resolution
                # First create a crop around where the ball was previously seen
                trackedBall = tracker2DStrict.balls[tracker2DStrict.activeBallIndex] # We can only have up to one at a time
                trackedBall_h, trackedBall_w = trackedBall.position # In normalized coordinates
                withinBound, TL, relCenter = eyes2DFocus.ball_within_bounds(
                    H_full, W_full, (trackedBall_h * H_full, trackedBall_w * W_full), CONFIG.crop.crop_h_2d, CONFIG.crop.crop_w_2d, padding=0)
                if withinBound: # Take a crop
                    corner_h_crop_2D, corner_w_crop_2D = TL
                    mapLx_crop_2D[:] = mapLx[
                        corner_h_crop_2D:corner_h_crop_2D + CONFIG.crop.crop_h_2d, corner_w_crop_2D:corner_w_crop_2D + CONFIG.crop.crop_w_2d]
                    mapLy_crop_2D[:] = mapLy[
                        corner_h_crop_2D:corner_h_crop_2D + CONFIG.crop.crop_h_2d, corner_w_crop_2D:corner_w_crop_2D + CONFIG.crop.crop_w_2d]
                    Lc2D = cv2.remap(
                        left_full,
                        mapLx_crop_2D,
                        mapLy_crop_2D,
                        cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT
                    )
                    centroids = eyes2DFocus.find_centroids_hsv(Lc2D)  # Bottleneck line
                    if centroids: # We found at least 1 ball
                        centroid_h, centroid_w, radius = centroids[0] # Convert to global normalized coordinates
                        global_centroid_h = (corner_h_crop_2D + centroid_h * CONFIG.crop.crop_h_2d)/H_full
                        global_centroid_w = (corner_w_crop_2D + centroid_w * CONFIG.crop.crop_w_2d)/W_full
                        radius = int(radius * CONFIG.runtime.scale) # Remember, we keep radius w.r.t. SCALE
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
            activeBefore = tracker2DStrict.activelyTracking
            tracker2DStrict.checkActive() # Activates or deactivates for the next frame
            activeAfter = tracker2DStrict.activelyTracking
            if activeBefore and not activeAfter:
                continuousTracking = False

            # We write to the csv file now
            line = [frame_id]
            for ball in tracker2DStrict.balls:
                if ball is None:
                    line += [None] * lengthPerEntry
                else:
                    line += [ball.position[1], ball.position[0], ball.radius, ball.updated, ball.confirmed_ball]
            writer.writerow(line)

            # We can fill in the mandatory fields in the state
            state = InternalState(
                frame_id=frame_id,
                timestamp=time.time(),
                tracking_active=tracker2DStrict.activelyTracking,
                continuous_tracking=continuousTracking,
                camera_ok=True,
                active_ball_index=tracker2DStrict.activeBallIndex
            )

            # ----------------------------------- The 3D stuff happens here -----------------------------------

            # Frame annotation
            if not tracker2DStrict.activelyTracking: # No confirmed balls
                tracker_3D.balls[0] = None # Delete whatever was there
                landingFilter.reset()
                if CONFIG.runtime.save_video:
                    frameBuffer[frame_id] = rectL_downscaled.copy()
                    flush_ready_frames()

                # Finishing the current state
                if depthFuture is not None:
                    if not depthFuture.done():  # Pending
                        state.depth_job_status = DepthJobStatus.PENDING
                    else:  # Done
                        state.depth_job_status = DepthJobStatus.DONE
                add_state(state)

                frame_id += 1
                if frame_id % CONFIG.runtime.fps == 0 and CONFIG.runtime.show_print:
                    print(time.time() - tst)
                    tst = time.time()
                continue

            ballIndex, ball3D = tracker2DStrict.activeBallIndex, tracker_3D.balls[0] # Only one ball tracked by tracker_3D
            ball2D = tracker2DStrict.balls[ballIndex]
            (center_h, center_w) = ball2D.position.copy()  # Normalized 2D coordinates

            # Can fill in 2D ball state information
            state.ball_2d_center_hw = ball2D.position.copy()
            state.ball_2d_radius = ball2D.radius
            state.ball_2d_updated = ball2D.updated

            withinBound, TL, relCenter = eyes2DStrict.ball_within_bounds(
                H_full, W_full, (center_h * H_full,center_w * W_full), CONFIG.crop.crop_h_3d, CONFIG.crop.crop_w_3d, padding=CONFIG.vision2d_strict.padding)
            if not withinBound: # Out of frame. Ignore.
                tracker_3D.balls[0] = None
                landingFilter.reset()
                if CONFIG.runtime.save_video:
                    frameBuffer[frame_id] = rectL_downscaled.copy()
                    flush_ready_frames()

                # Finishing the current state
                if depthFuture is not None:
                    if not depthFuture.done():  # Pending
                        state.depth_job_status = DepthJobStatus.PENDING
                    else:  # Done
                        state.depth_job_status = DepthJobStatus.DONE
                add_state(state)

                frame_id += 1
                if frame_id % CONFIG.runtime.fps == 0 and CONFIG.runtime.show_print:
                    print(time.time() - tst)
                    tst = time.time()
                continue
            # The code from here ONLY runs if we have a confirmed ball within bounds.
            if CONFIG.runtime.save_video:
                cv2.circle(rectL_downscaled, center=(int(W_centroids * ball2D.position[1]), int(H_centroids * ball2D.position[0])),
                        radius=ball2D.radius, color=CONFIG.annotator.ball_color, thickness=2) # Draw a yellow circle
                frameBuffer[frame_id] = rectL_downscaled.copy()
                flush_ready_frames()

            # 1) We poll to check whether our depth estimation task is completed
            if depthFuture is not None and depthFuture.done() and continuousTracking: # Job done + Can receive
                state.depth_job_status = DepthJobStatus.DONE
                frameSubmitted, ballIndexSubmitted, pos3D = depthFuture.result()
                r = frameLastReceived
                s = frameSubmitted
                f1 = frame_id

                framesSinceLastReceive = frame_id - frameLastReceived
                # if CONFIG.runtime.show_print: print("Received: ", framesSinceLastReceive)

                # Three cases, we cannot predict, we can begin prediction, and we can continue predicting
                if ball3D is None: # We can create a new ball
                    tracker_3D.update(0, ball3D, pos3D, depthMeta["ballUpdated"]) # We create a ball and make no predictions
                    landingFilter.reset()
                # We predict where the current ball should be, given past data
                elif ball3D.canPredict:
                    # posStar3D = tracker_3D.predict_position_until_landing(ball3D, 1/CONFIG.runtime.fps)  # Prediction in meters UNCOMMENT FOR REAL INFERENCE

                    # -----------------------
                    # Part 1: r ... s-1
                    # -----------------------
                    for k in range(r, s):
                        j = len(ball2D.pPast) - (frame_id - k)
                        if j < 0 or j >= len(ball2D.pPast):
                            continue
                        ball2D_h, ball2D_w = ball2D.pPast[j]
                        pos3D_gap = tracker_3D.educated_guess_position(ball2D_h, ball2D_w, ball3D, eyes3D)
                        tracker_3D.predict_next_position(ball3D, 1/CONFIG.runtime.fps)
                        tracker_3D.check_observation_prediction_compatible(ball3D, pos3D_gap)
                        pos3D_gap = tracker_3D.filter_observation_prediction(ball3D, pos3D_gap, CONFIG.filter.weight_func)
                        tracker_3D.update(0, ball3D, pos3D_gap, True)
                        posStar3D_gap = tracker_3D.predict_position_until_landing(ball3D, 1/CONFIG.runtime.fps) # We use posStar3D for landing predictions
                        ball3D.predictedLandingPosition = posStar3D_gap.copy()

                        # Output to state
                        st = state_by_frame.get(k)
                        st.ball_3d_position_tbd = pos3D_gap.copy()
                        st.landing_prediction_raw_tbd = posStar3D_gap.copy()
                        ball3D.predictedSmoothedLandingPosition = landingFilter.landing_prediction(posStar3D_gap).copy()
                        st.landing_prediction_smoothed_tbd = ball3D.predictedSmoothedLandingPosition
                        # annotate_landing_circle(k, posStar3D_gap, depthMeta["ball2D_r"]) # Raw posStar3D_gap annotation
                        if CONFIG.runtime.save_video: annotate_landing_circle(k, ball3D.predictedSmoothedLandingPosition, depthMeta["ball2D_r"])

                    # -----------------------
                    # Part 2: real observation at s
                    # -----------------------
                    tracker_3D.predict_next_position(ball3D, 1/CONFIG.runtime.fps)
                    tracker_3D.check_observation_prediction_compatible(ball3D, pos3D)
                    pos3D = tracker_3D.filter_observation_prediction(ball3D, pos3D, CONFIG.filter.weight_func)
                    tracker_3D.update(0, ball3D, pos3D, depthMeta["ballUpdated"])  # We predict, check and only then update
                    posStar3D = tracker_3D.predict_position_until_landing(ball3D, 1/CONFIG.runtime.fps)
                    ball3D.predictedLandingPosition = posStar3D.copy()

                    # Output to state
                    st = state_by_frame.get(s)
                    st.ball_3d_position_tbd = pos3D.copy()
                    st.landing_prediction_raw_tbd = posStar3D.copy()
                    ball3D.predictedSmoothedLandingPosition = landingFilter.landing_prediction(posStar3D).copy()
                    st.landing_prediction_smoothed_tbd = ball3D.predictedSmoothedLandingPosition
                    # annotate_landing_circle(s, posStar3D, depthMeta["ball2D_r"]) # Raw posStar3D annotation
                    if CONFIG.runtime.save_video: annotate_landing_circle(s, ball3D.predictedSmoothedLandingPosition, depthMeta["ball2D_r"])

                    # -----------------------
                    # Part 3: s+1 ... f1-1
                    # -----------------------
                    for k in range(s + 1, f1):
                        j = len(ball2D.pPast) - (f1 - k)
                        if j < 0 or j >= len(ball2D.pPast):
                            continue

                        ball2D_h, ball2D_w = ball2D.pPast[j]
                        pos3D_gap = tracker_3D.educated_guess_position(ball2D_h, ball2D_w, ball3D, eyes3D)
                        tracker_3D.predict_next_position(ball3D, 1/CONFIG.runtime.fps)
                        tracker_3D.check_observation_prediction_compatible(ball3D, pos3D_gap)
                        pos3D_gap = tracker_3D.filter_observation_prediction(ball3D, pos3D_gap, CONFIG.filter.weight_func)
                        tracker_3D.update(0, ball3D, pos3D_gap, True)
                        posStar3D_gap = tracker_3D.predict_position_until_landing(ball3D, 1/CONFIG.runtime.fps)  # We use posStar3D for landing predictions
                        ball3D.predictedLandingPosition = posStar3D_gap.copy()

                        # Output to state
                        st = state_by_frame.get(k)
                        st.ball_3d_position_tbd = pos3D_gap.copy()
                        st.landing_prediction_raw_tbd = posStar3D_gap.copy()
                        ball3D.predictedSmoothedLandingPosition = landingFilter.landing_prediction(posStar3D_gap).copy()
                        st.landing_prediction_smoothed_tbd = ball3D.predictedSmoothedLandingPosition
                        # annotate_landing_circle(k, posStar3D_gap, depthMeta["ball2D_r"]) # Raw posStar3D_gap annotation
                        if CONFIG.runtime.save_video: annotate_landing_circle(k, ball3D.predictedSmoothedLandingPosition, depthMeta["ball2D_r"])

                else: # Cannot make predictions
                    tracker_3D.update(0, ball3D, pos3D, depthMeta["ballUpdated"])  # Predict = None, check = None, update

                tIntervals = gapsForRescale.copy()
                tIntervals.append(frame_id - frameLastRequest)
                tracker_3D.update_predictability(0, ball3D, depthMeta["ballUpdated"], tIntervals)
                current_ball3D = tracker_3D.balls[0]
                state.can_predict = (current_ball3D is not None and current_ball3D.canPredict)

                frameLastReceived = frame_id
                depthFuture = None # We can now submit a new task

            elif depthFuture is not None and depthFuture.done() and not continuousTracking: # Job done but can't receive
                state.depth_job_status = DepthJobStatus.DONE
                continuousTracking = True # We are working with a ball different from the one at request time.
                depthFuture = None # Can still submit a new task
                landingFilter.reset()

            # 3) Depth task submission, we should only perform if the ball was seen
            elif depthFuture is None and ball2D.updated: # Submit a new job
                state.depth_job_status = DepthJobStatus.SUBMITTED
                corner_h_crop_3D, corner_w_crop_3D = TL
                mapLx_crop_3D[:] = mapLx[corner_h_crop_3D:corner_h_crop_3D + CONFIG.crop.crop_h_3d, corner_w_crop_3D:corner_w_crop_3D + CONFIG.crop.crop_w_3d]
                mapLy_crop_3D[:] = mapLy[corner_h_crop_3D:corner_h_crop_3D + CONFIG.crop.crop_h_3d, corner_w_crop_3D:corner_w_crop_3D + CONFIG.crop.crop_w_3d]
                mapRx_crop_3D[:] = mapRx[corner_h_crop_3D:corner_h_crop_3D + CONFIG.crop.crop_h_3d, corner_w_crop_3D:corner_w_crop_3D + CONFIG.crop.crop_w_3d]
                mapRy_crop_3D[:] = mapRy[corner_h_crop_3D:corner_h_crop_3D + CONFIG.crop.crop_h_3d, corner_w_crop_3D:corner_w_crop_3D + CONFIG.crop.crop_w_3d]

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
                ball2D_h, ball2D_w = ball2D.position.copy()
                ball2D_r = ball2D.radius
                depthFuture = depthPool.submit(depthJob, frame_id, ballIndex, ball2D_h, ball2D_w, ball2D_r,
                                               Lc, Rc, relCenter)  # We have the observed coordinates
                depthMeta = {
                    "frameSubmitted": frame_id,
                    "frameProtectStart": frameLastReceived,
                    "ballIndex": ballIndex,
                    "ballUpdated": ball2D.updated,
                    "ball2D_h": ball2D_h,
                    "ball2D_w": ball2D_w,
                    "ball2D_r": ball2D_r,
                }
                framesSinceLastRequest = frame_id - frameLastRequest
                frameLastRequest = frame_id
                # if CONFIG.runtime.show_print: print("Request: ", framesSinceLastRequest)
                gapsForRescale.append(framesSinceLastRequest)


            elif depthFuture is None and not ball2D.updated: # Should not submit a new job
                tracker_3D.update_predictability(0, ball3D, ball2D.updated)
                current_ball3D = tracker_3D.balls[0]
                state.can_predict = (current_ball3D is not None and current_ball3D.canPredict)
            else: # depthFuture is not None, but the job is not done yet
                state.depth_job_status = DepthJobStatus.PENDING

            # Writing atm data with tracker_3D
            current_ball3D = tracker_3D.balls[0]

            state.ball_3d_position_atm = (
                None if current_ball3D is None or len(current_ball3D.pPrimePast) == 0
                else current_ball3D.pPrimePast[-1].copy()
            )

            state.landing_prediction_raw_atm = (
                None if current_ball3D is None or current_ball3D.predictedLandingPosition is None or
                        len(current_ball3D.predictedLandingPosition) == 0
                else current_ball3D.predictedLandingPosition.copy()
            )

            state.landing_prediction_smoothed_atm = (
                None if current_ball3D is None or current_ball3D.predictedSmoothedLandingPosition is None or
                        len(current_ball3D.predictedSmoothedLandingPosition) == 0
                else current_ball3D.predictedSmoothedLandingPosition.copy()
            )

            state.confidence = compute_confidence(state)
            add_state(state)

            # out.write(rectL_downscaled)
            frame_id += 1
            if frame_id % CONFIG.runtime.fps == 0 and CONFIG.runtime.show_print:
                print(time.time() - tst)
                tst = time.time()

        if CONFIG.runtime.save_video: flush_ready_frames(force=True)

        t1 = time.time()
        if CONFIG.runtime.show_print: print(f"Time taken for video with {frame_id} frames: {t1 - t0} seconds")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nUnhandled error: {e}")
        raise
    finally:
        cleanup(
            cap=cap,
            csvfile=csvfile,
            out=out,
            strictLaxPool=strictLaxPool,
            depthPool=depthPool,
        )

if __name__ == "__main__":
    from config import CONFIG
    run(CONFIG)