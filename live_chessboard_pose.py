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


def build_object_points(board_size, square_size):
    """构造棋盘格内角点的 3D 坐标。"""
    cols, rows = board_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def draw_axes(image, camera_matrix, dist_coeffs, rvec, tvec, axis_length):
    """在图像上绘制棋盘格坐标轴。"""
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
    parser = argparse.ArgumentParser(description="Real-time chessboard pose visualization.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--board-size", required=True, help="Inner corners as cols,rows.")
    parser.add_argument("--square-size", type=float, required=True, help="Chessboard square size (m).")
    parser.add_argument("--camera-matrix", required=True, help="Camera intrinsics 3x3 row-major list.")
    parser.add_argument("--dist-coeffs", default="0,0,0,0,0", help="Dist coeffs list.")
    parser.add_argument("--axis-length", type=float, default=0.05, help="Axis length in meters.")
    args = parser.parse_args()

    board_cols, board_rows = parse_float_list(args.board_size, 2, "board-size")
    board_size = (int(board_cols), int(board_rows))
    camera_matrix = np.array(parse_float_list(args.camera_matrix, 9, "camera-matrix")).reshape(3, 3)
    dist_coeffs = np.array(parse_float_list(args.dist_coeffs, 5, "dist-coeffs")).reshape(5, 1)
    objp = build_object_points(board_size, args.square_size)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            found, corners = cv2.findChessboardCorners(gray, board_size)
            if found:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                success, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
                if success:
                    cv2.drawChessboardCorners(image, board_size, corners, found)
                    origin = draw_axes(
                        image,
                        camera_matrix,
                        dist_coeffs,
                        rvec,
                        tvec,
                        args.axis_length,
                    )
                    cv2.circle(image, origin, 4, (0, 255, 255), -1)
                    text = f"t=[{tvec[0][0]:.3f},{tvec[1][0]:.3f},{tvec[2][0]:.3f}]"
                    cv2.putText(image, text, (origin[0] + 10, origin[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            cv2.imshow("Chessboard Pose", image)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
