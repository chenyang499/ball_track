import argparse

import cv2
import numpy as np
import pyrealsense2 as rs


def parse_float_list(text, length, name):
    """解析逗号分隔的浮点数列表。"""
    values = [float(item) for item in text.split(",")]
    if len(values) != length:
        raise ValueError(f"{name} must have {length} values, got {len(values)}")
    return values


def rs_intrinsics_to_matrix(intrinsics):
    """将 RealSense 内参转换为 OpenCV 相机矩阵与畸变系数。"""
    camera_matrix = np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float32).reshape(5, 1)
    return camera_matrix, dist_coeffs


def build_object_points(pattern_size, spacing, asymmetric=False):
    """构造圆点阵列的 3D 坐标。"""
    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), np.float32)
    if asymmetric:
        points = []
        for r in range(rows):
            for c in range(cols):
                x = (2 * c + (r % 2)) * spacing
                y = r * spacing
                points.append([x, y, 0.0])
        objp = np.array(points, dtype=np.float32)
    else:
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp *= spacing
    return objp


def build_blob_detector():
    """构造用于圆点检测的 blob detector。"""
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 5000
    params.filterByCircularity = False
    params.filterByInertia = False
    params.filterByConvexity = False
    return cv2.SimpleBlobDetector_create(params)


def draw_axes(image, camera_matrix, dist_coeffs, rvec, tvec, axis_length):
    """在图像上绘制坐标轴。"""
    axis = np.float32(
        [
            [0, 0, 0],
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, axis_length],
        ]
    )
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
    origin = tuple(imgpts[0].ravel().astype(int))
    x_axis = tuple(imgpts[1].ravel().astype(int))
    y_axis = tuple(imgpts[2].ravel().astype(int))
    z_axis = tuple(imgpts[3].ravel().astype(int))

    cv2.line(image, origin, x_axis, (0, 0, 255), 2)
    cv2.line(image, origin, y_axis, (0, 255, 0), 2)
    cv2.line(image, origin, z_axis, (255, 0, 0), 2)
    return origin


def main():
    parser = argparse.ArgumentParser(description="Real-time circle grid pose visualization.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--pattern-size", required=True, help="Dot grid size as cols,rows.")
    parser.add_argument("--spacing", type=float, required=True, help="Dot spacing (m).")
    parser.add_argument("--asymmetric", action="store_true", help="Use asymmetric circle grid.")
    parser.add_argument("--camera-matrix", default=None, help="Camera intrinsics 3x3 row-major list.")
    parser.add_argument("--dist-coeffs", default=None, help="Dist coeffs list.")
    parser.add_argument("--axis-length", type=float, default=0.05, help="Axis length in meters.")
    args = parser.parse_args()

    cols, rows = parse_float_list(args.pattern_size, 2, "pattern-size")
    pattern_size = (int(cols), int(rows))
    objp = build_object_points(pattern_size, args.spacing, asymmetric=args.asymmetric)
    blob_detector = build_blob_detector()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(config)
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    rs_intrinsics = color_stream.get_intrinsics()

    if args.camera_matrix is None:
        camera_matrix, dist_coeffs = rs_intrinsics_to_matrix(rs_intrinsics)
    else:
        camera_matrix = np.array(parse_float_list(args.camera_matrix, 9, "camera-matrix")).reshape(3, 3)
        if args.dist_coeffs is None:
            dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        else:
            dist_coeffs = np.array(parse_float_list(args.dist_coeffs, 5, "dist-coeffs")).reshape(5, 1)

    flags = cv2.CALIB_CB_SYMMETRIC_GRID
    if args.asymmetric:
        flags = cv2.CALIB_CB_ASYMMETRIC_GRID

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            found, centers = cv2.findCirclesGrid(
                gray,
                pattern_size,
                flags=flags,
                blobDetector=blob_detector,
            )
            if found:
                success, rvec, tvec = cv2.solvePnP(objp, centers, camera_matrix, dist_coeffs)
                if success:
                    cv2.drawChessboardCorners(image, pattern_size, centers, found)
                    origin = draw_axes(image, camera_matrix, dist_coeffs, rvec, tvec, args.axis_length)
                    cv2.circle(image, origin, 4, (0, 255, 255), -1)

            cv2.imshow("Circle Grid Pose", image)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
