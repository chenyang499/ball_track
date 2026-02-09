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
    parser = argparse.ArgumentParser(description="Real-time ChArUco board pose visualization.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--squares-x", type=int, required=True, help="ChArUco squares in X direction.")
    parser.add_argument("--squares-y", type=int, required=True, help="ChArUco squares in Y direction.")
    parser.add_argument("--square-length", type=float, required=True, help="Chessboard square size (m).")
    parser.add_argument("--marker-length", type=float, required=True, help="ArUco marker size (m).")
    parser.add_argument("--dictionary", type=str, default="DICT_4X4_50", help="ArUco dictionary name.")
    parser.add_argument("--camera-matrix", default=None, help="Camera intrinsics 3x3 row-major list.")
    parser.add_argument("--dist-coeffs", default=None, help="Dist coeffs list.")
    parser.add_argument("--axis-length", type=float, default=0.05, help="Axis length in meters.")
    args = parser.parse_args()

    dictionary_id = getattr(cv2.aruco, args.dictionary, None)
    if dictionary_id is None:
        raise ValueError(f"Unknown ArUco dictionary: {args.dictionary}")
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        args.square_length,
        args.marker_length,
        aruco_dict,
    )
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

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

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(image, corners, ids)
                charuco_corners, charuco_ids, _ = cv2.aruco.interpolateCornersCharuco(
                    corners,
                    ids,
                    gray,
                    board,
                    camera_matrix,
                    dist_coeffs,
                )
                if charuco_ids is not None and len(charuco_ids) > 3:
                    success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                        charuco_corners,
                        charuco_ids,
                        board,
                        camera_matrix,
                        dist_coeffs,
                        None,
                        None,
                    )
                    if success:
                        cv2.aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids)
                        origin = draw_axes(image, camera_matrix, dist_coeffs, rvec, tvec, args.axis_length)
                        cv2.circle(image, origin, 4, (0, 255, 255), -1)

            cv2.imshow("ChArUco Pose", image)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
