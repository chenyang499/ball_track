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
