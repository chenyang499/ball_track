# Ball Tracking with RealSense D455f

该示例工程用于使用 RealSense D455f 追踪直径 10cm 的抛射球体，融合机械臂先验（初始位置/速度）并使用抛物线运动模型进行滤波。

## 功能特性

- 结合深度和颜色的球体检测（可选颜色阈值）。
- 利用球直径 + 相机内参估计像素半径进行几何约束。
- 6 维状态（位置 + 速度）滤波，带已知重力加速度。
- 可视化检测结果与预测位置。

## 环境依赖

- Python 3.9+
- `pyrealsense2`
- `opencv-python`
- `numpy`

安装依赖：

```bash
pip install -r requirements.txt
```

## 使用方式

```bash
python ball_tracker.py \
  --diameter 0.10 \
  --init-pos 0.0,0.0,1.0 \
  --init-vel 0.0,0.0,0.0 \
  --gravity 0.0,9.81,0.0 \
  --depth-tol 0.35
```

## 相机-机械臂外参标定

该脚本适用于 2 连杆平面机械臂，使用固定在末端执行器上的棋盘格标定板完成手眼标定。

### 数据采集准备

1. 将棋盘格固定在末端执行器，保持相机可见。
2. 采集多组关节角与对应图像，记录到 CSV：

```csv
image,theta1,theta2
data/img_0001.png,0.10,0.20
data/img_0002.png,0.20,0.25
```

### 运行标定

```bash
python calibrate_extrinsics.py \
  --samples data/samples.csv \
  --link-lengths 0.30,0.20 \
  --board-size 7,6 \
  --square-size 0.024 \
  --camera-matrix 600,0,320,0,600,240,0,0,1 \
  --dist-coeffs 0,0,0,0,0 \
  --output extrinsics.yaml
```

输出为 `extrinsics.yaml`，包含相机到末端执行器的旋转和平移矩阵，可用于后续将机械臂先验转换到相机坐标系。

### 常用参数说明

- `--diameter`：球体直径（米）
- `--init-pos`：初始位置（相机坐标系，单位米）
- `--init-vel`：初始速度（相机坐标系，单位 m/s）
- `--gravity`：重力向量（相机坐标系，单位 m/s^2）
- `--depth-tol`：预测深度容忍范围（米）
- `--hsv-lower/--hsv-upper`：HSV 颜色阈值（可选，用于彩色分割）

## 说明

1. 请确保已完成机械臂坐标系到相机坐标系的外参标定。
2. 如果球颜色明显，建议提供 HSV 阈值以增强检测鲁棒性。
3. 如需更高精度，可增加多帧观测或使用球面拟合（RANSAC）。
