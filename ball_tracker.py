import argparse
import time

import cv2
import numpy as np
import pyrealsense2 as rs


def parse_vector(text, length, name):
    values = [float(item) for item in text.split(",")]
    if len(values) != length:
        raise ValueError(f"{name} must have {length} values, got {len(values)}")
    return np.array(values, dtype=np.float32)


class BallKalmanFilter:
    def __init__(self, initial_position, initial_velocity, gravity, process_noise, measurement_noise):
        self.state = np.zeros(6, dtype=np.float32)
        self.state[:3] = initial_position
        self.state[3:] = initial_velocity
        self.gravity = gravity.astype(np.float32)
        self.P = np.eye(6, dtype=np.float32) * 0.1
        self.Q = np.eye(6, dtype=np.float32) * process_noise
        self.R = np.eye(3, dtype=np.float32) * measurement_noise

    def predict(self, dt):
        if dt <= 0:
            return self.state[:3].copy()
        A = np.eye(6, dtype=np.float32)
        A[0, 3] = dt
        A[1, 4] = dt
        A[2, 5] = dt

        B = np.zeros((6, 3), dtype=np.float32)
        B[0, 0] = 0.5 * dt * dt
        B[1, 1] = 0.5 * dt * dt
        B[2, 2] = 0.5 * dt * dt
        B[3, 0] = dt
        B[4, 1] = dt
        B[5, 2] = dt

        self.state = A @ self.state + B @ self.gravity
        self.P = A @ self.P @ A.T + self.Q
        return self.state[:3].copy()

    def update(self, measurement):
        H = np.zeros((3, 6), dtype=np.float32)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        z = measurement.reshape(3, 1)
        x = self.state.reshape(6, 1)
        y = z - H @ x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        I = np.eye(6, dtype=np.float32)
        self.P = (I - K @ H) @ self.P
        self.state = x.reshape(6)
        return self.state[:3].copy()


class BallTracker:
    def __init__(self, args):
        self.args = args
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
        self.config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        self.align = rs.align(rs.stream.color)
        self.profile = None
        self.intrinsics = None
        self.kf = BallKalmanFilter(
            initial_position=args.init_pos,
            initial_velocity=args.init_vel,
            gravity=args.gravity,
            process_noise=args.process_noise,
            measurement_noise=args.measurement_noise,
        )
        self.last_time = None

    def start(self):
        self.profile = self.pipeline.start(self.config)
        color_stream = self.profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    def stop(self):
        self.pipeline.stop()

    def project_point(self, point):
        pixel = rs.rs2_project_point_to_pixel(
            self.intrinsics,
            [float(point[0]), float(point[1]), float(point[2])],
        )
        return int(pixel[0]), int(pixel[1])

    def expected_radius_px(self, depth):
        if depth <= 0:
            return None
        return (self.args.diameter / 2.0) * self.intrinsics.fx / depth

    def build_roi(self, center, radius_px, shape):
        if center is None or radius_px is None:
            return (0, 0, shape[1], shape[0])
        extra = int(radius_px * 2.5) + 10
        x0 = max(center[0] - extra, 0)
        y0 = max(center[1] - extra, 0)
        x1 = min(center[0] + extra, shape[1])
        y1 = min(center[1] + extra, shape[0])
        return (x0, y0, x1, y1)

    def detect_ball(self, color_image, depth_image, prediction):
        predicted_center = None
        expected_radius = None
        if prediction is not None and prediction[2] > 0:
            predicted_center = self.project_point(prediction)
            expected_radius = self.expected_radius_px(prediction[2])

        x0, y0, x1, y1 = self.build_roi(predicted_center, expected_radius, color_image.shape)
        color_roi = color_image[y0:y1, x0:x1]
        depth_roi = depth_image[y0:y1, x0:x1]

        mask = np.ones(depth_roi.shape, dtype=np.uint8) * 255
        if prediction is not None and prediction[2] > 0:
            depth_center = prediction[2]
            depth_min = max(depth_center - self.args.depth_tol, 0.1)
            depth_max = depth_center + self.args.depth_tol
            mask = cv2.inRange(depth_roi, depth_min, depth_max)

        if self.args.hsv_lower is not None and self.args.hsv_upper is not None:
            hsv = cv2.cvtColor(color_roi, cv2.COLOR_BGR2HSV)
            color_mask = cv2.inRange(hsv, self.args.hsv_lower, self.args.hsv_upper)
            mask = cv2.bitwise_and(mask, color_mask)

        mask = cv2.medianBlur(mask, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_candidate = None
        best_score = 0.0
        for contour in contours:
            if cv2.contourArea(contour) < self.args.min_area:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if radius <= 0:
                continue
            if expected_radius is not None:
                ratio = radius / expected_radius
                if ratio < self.args.radius_ratio_min or ratio > self.args.radius_ratio_max:
                    continue
            score = cv2.contourArea(contour)
            if score > best_score:
                best_score = score
                best_candidate = (int(cx), int(cy), radius, contour)

        if best_candidate is None:
            return None, None, (x0, y0, x1, y1)

        cx, cy, radius, contour = best_candidate
        full_cx = cx + x0
        full_cy = cy + y0

        contour_mask = np.zeros_like(depth_roi, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        depths = depth_roi[contour_mask == 255]
        if depths.size == 0:
            return None, None, (x0, y0, x1, y1)

        depth_median = float(np.median(depths))
        point_3d = rs.rs2_deproject_pixel_to_point(
            self.intrinsics,
            [float(full_cx), float(full_cy)],
            depth_median,
        )
        return (full_cx, full_cy, radius), np.array(point_3d, dtype=np.float32), (x0, y0, x1, y1)

    def run(self):
        self.start()
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                frames = self.align.process(frames)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_frame.get_units()
                color_image = np.asanyarray(color_frame.get_data())

                now = time.time()
                if self.last_time is None:
                    dt = 0.0
                else:
                    dt = now - self.last_time
                self.last_time = now

                prediction = self.kf.predict(dt)
                detection, position_3d, roi = self.detect_ball(color_image, depth_image, prediction)
                if position_3d is not None:
                    self.kf.update(position_3d)

                if detection is not None:
                    cx, cy, radius = detection
                    cv2.circle(color_image, (cx, cy), int(radius), (0, 255, 0), 2)
                    cv2.circle(color_image, (cx, cy), 3, (0, 255, 0), -1)
                    text = f"X:{position_3d[0]:.2f} Y:{position_3d[1]:.2f} Z:{position_3d[2]:.2f}"
                    cv2.putText(color_image, text, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                x0, y0, x1, y1 = roi
                cv2.rectangle(color_image, (x0, y0), (x1, y1), (255, 0, 0), 1)

                cv2.imshow("Ball Tracking", color_image)
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break
        finally:
            self.stop()
            cv2.destroyAllWindows()


def build_arg_parser():
    parser = argparse.ArgumentParser(description="RealSense D455f ball tracking with motion prior.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--diameter", type=float, default=0.10)
    parser.add_argument("--depth-tol", type=float, default=0.35)
    parser.add_argument("--min-area", type=float, default=80.0)
    parser.add_argument("--radius-ratio-min", type=float, default=0.5)
    parser.add_argument("--radius-ratio-max", type=float, default=1.8)
    parser.add_argument("--process-noise", type=float, default=0.01)
    parser.add_argument("--measurement-noise", type=float, default=0.02)
    parser.add_argument("--init-pos", type=str, default="0.0,0.0,1.0")
    parser.add_argument("--init-vel", type=str, default="0.0,0.0,0.0")
    parser.add_argument("--gravity", type=str, default="0.0,9.81,0.0")
    parser.add_argument("--hsv-lower", type=str, default=None)
    parser.add_argument("--hsv-upper", type=str, default=None)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.init_pos = parse_vector(args.init_pos, 3, "init-pos")
    args.init_vel = parse_vector(args.init_vel, 3, "init-vel")
    args.gravity = parse_vector(args.gravity, 3, "gravity")
    if args.hsv_lower is not None and args.hsv_upper is not None:
        args.hsv_lower = np.array(parse_vector(args.hsv_lower, 3, "hsv-lower"), dtype=np.uint8)
        args.hsv_upper = np.array(parse_vector(args.hsv_upper, 3, "hsv-upper"), dtype=np.uint8)
    else:
        args.hsv_lower = None
        args.hsv_upper = None
    tracker = BallTracker(args)
    tracker.run()


if __name__ == "__main__":
    main()
