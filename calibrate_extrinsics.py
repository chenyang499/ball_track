import argparse
import csv

import cv2
import numpy as np


def parse_float_list(text, length, name):
    """解析逗号分隔的浮点数列表。"""
    values = [float(item) for item in text.split(",")]
    if len(values) != length:
        raise ValueError(f"{name} must have {length} values, got {len(values)}")
    return values


def load_joint_samples(csv_path):
    """读取 CSV 采样数据。"""
    samples = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            samples.append(
                {
                    "image": row["image"].strip(),
                    "theta1": float(row["theta1"]),
                    "theta2": float(row["theta2"]),
                }
            )
    if not samples:
        raise ValueError("No samples found in CSV.")
    return samples


def fk_planar_2link(theta1, theta2, link1, link2):
    """平面 2 连杆正运动学，返回 base->gripper 齐次矩阵。"""
    c1 = np.cos(theta1)
    s1 = np.sin(theta1)
    c12 = np.cos(theta1 + theta2)
    s12 = np.sin(theta1 + theta2)
    x = link1 * c1 + link2 * c12
    y = link1 * s1 + link2 * s12
    transform = np.eye(4, dtype=np.float64)
    transform[0, 0] = c12
    transform[0, 1] = -s12
    transform[1, 0] = s12
    transform[1, 1] = c12
    transform[0, 3] = x
    transform[1, 3] = y
    return transform


def extract_rvec_tvec(transform):
    """将 4x4 齐次矩阵转换为 Rodrigues 旋转向量与平移向量。"""
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    rvec, _ = cv2.Rodrigues(rotation)
    return rvec.reshape(3, 1), translation.reshape(3, 1)


def solve_board_pose(image, board_size, square_size, camera_matrix, dist_coeffs):
    """检测棋盘格并使用 PnP 求解标定板位姿。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, board_size)
    if not found:
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    success, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
    if not success:
        return None
    return rvec, tvec


def calibrate_hand_eye(samples, link1, link2, board_size, square_size, camera_matrix, dist_coeffs):
    """组装机械臂与棋盘格的位姿对，执行手眼标定。"""
    r_gripper2base = []
    t_gripper2base = []
    r_target2cam = []
    t_target2cam = []

    for sample in samples:
        image = cv2.imread(sample["image"])
        if image is None:
            raise FileNotFoundError(f"Image not found: {sample['image']}")

        board_pose = solve_board_pose(image, board_size, square_size, camera_matrix, dist_coeffs)
        if board_pose is None:
            continue
        rvec, tvec = board_pose
        # 棋盘格在相机坐标系下的位姿
        r_target2cam.append(rvec)
        t_target2cam.append(tvec)

        # 机械臂正运动学得到 base->gripper，再取逆得到 gripper->base
        base_to_gripper = fk_planar_2link(
            sample["theta1"],
            sample["theta2"],
            link1,
            link2,
        )
        gripper_to_base = np.linalg.inv(base_to_gripper)
        rvec_g2b, tvec_g2b = extract_rvec_tvec(gripper_to_base)
        r_gripper2base.append(rvec_g2b)
        t_gripper2base.append(tvec_g2b)

    if len(r_target2cam) < 3:
        raise ValueError("Need at least 3 valid chessboard detections for calibration.")

    r_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        r_gripper2base,
        t_gripper2base,
        r_target2cam,
        t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )
    return r_cam2gripper, t_cam2gripper


def main():
    parser = argparse.ArgumentParser(description="2-link arm to camera extrinsic calibration.")
    parser.add_argument("--samples", required=True, help="CSV with image,theta1,theta2 columns.")
    parser.add_argument("--link-lengths", required=True, help="Link lengths in meters, e.g. 0.3,0.2")
    parser.add_argument("--board-size", default="7,6", help="Inner corners as cols,rows.")
    parser.add_argument("--square-size", type=float, default=0.024, help="Chessboard square size (m).")
    parser.add_argument("--camera-matrix", required=True, help="Camera intrinsics 3x3 row-major list.")
    parser.add_argument("--dist-coeffs", default="0,0,0,0,0", help="Dist coeffs list.")
    parser.add_argument("--output", default="extrinsics.yaml", help="Output YAML file.")
    args = parser.parse_args()

    # 解析参数
    link1, link2 = parse_float_list(args.link_lengths, 2, "link-lengths")
    board_cols, board_rows = parse_float_list(args.board_size, 2, "board-size")
    board_size = (int(board_cols), int(board_rows))
    camera_matrix = np.array(parse_float_list(args.camera_matrix, 9, "camera-matrix")).reshape(3, 3)
    dist_coeffs = np.array(parse_float_list(args.dist_coeffs, 5, "dist-coeffs")).reshape(5, 1)

    # 加载样本并执行标定
    samples = load_joint_samples(args.samples)
    r_cam2gripper, t_cam2gripper = calibrate_hand_eye(
        samples,
        link1,
        link2,
        board_size,
        args.square_size,
        camera_matrix,
        dist_coeffs,
    )

    # 保存结果
    fs = cv2.FileStorage(args.output, cv2.FILE_STORAGE_WRITE)
    fs.write("R_cam2gripper", r_cam2gripper)
    fs.write("t_cam2gripper", t_cam2gripper)
    fs.release()
    print(f"Saved extrinsics to {args.output}")


if __name__ == "__main__":
    main()
