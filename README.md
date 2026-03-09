# FR3 桌面抓取仿真 — MuJoCo + GraspNet

基于 MuJoCo 仿真的 **Franka FR3 机械臂** 视觉抓取与放置系统。将真实硬件管线（RealSense D435i → GraspNet → franky）完整映射到纯仿真环境（MuJoCo 渲染 → GraspNet 推理 → FrankaSimController）。

---

## 目录结构

```
mujoco_fr3/
├── fr3_table_grasp.xml          # 主 MuJoCo 场景
├── requirements.txt             # Python 依赖
├── franka_fr3/assets/           # 28 OBJ 视觉 mesh + 8 STL 碰撞 mesh (Menagerie)
├── captures/                    # D435i 捕获的 RGB / Depth (.npy + .png)
├── scripts/
│   ├── fr3_controller.py        # FrankaSimController — 核心控制器
│   ├── sim_graspnet_demo.py     # GraspNet 视觉抓取-放置完整管线
│   ├── demo_pick_place.py       # 硬编码关节位置 pick-and-place 演示
│   ├── capture_d435i.py         # D435i 仿真相机 RGB-D 采集
│   └── view_scene.py            # MuJoCo 原生 viewer
└── franka-graspnet/             # GraspNet 子项目
    ├── checkpoints/graspnet/    # 预训练权重 (checkpoint-rs.tar)
    ├── franka_graspnet/         # 推理 & 真实机器人控制
    ├── models/                  # 网络模型定义
    ├── pointnet2/               # PointNet++ CUDA 扩展
    └── scripts/                 # 真实机器人演示脚本
```

---

## 仿真场景

`fr3_table_grasp.xml` 包含以下内容：

| 组件 | 说明 |
|---|---|
| **FR3 机械臂** | 7-DOF，MuJoCo Menagerie 官方 mesh、惯性参数与关节参数 |
| **双指夹爪** | 平行夹爪（Franka Hand），slide joint × 2，range 0–0.04m |
| **D435i 相机** | eye-in-hand 安装，RGB (fovy=42.5°) + Depth (fovy=58°)，640×480 |
| **桌子** | 桌面高度 z≈0.76m |
| **可抓取物体** | 5 个：`target_cube`（方块）、`orange`（橘子圆柱）、`lemon`（柠檬方块）、`tomato`（番茄方块）、`small_box`（绿色小盒），freejoint |
| **容器盒** | `box_container`，开放顶部，内部约 14.6cm×14.6cm×6cm |
| **执行器** | 9 个位置 PD 控制器（7 关节 + 2 夹爪） |
| **固定相机** | `overview`（全景）、`grasp_close`（近景） |

---

## 安装

### 基础依赖

```bash
pip install mujoco numpy scipy
```

### GraspNet 依赖（视觉抓取管线）

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install open3d graspnetAPI

# 编译 PointNet++ CUDA 扩展
cd franka-graspnet/pointnet2
pip install .
```

> **注意**：如 GPU 架构不是 sm_120 (RTX 5090)，需修改 `pointnet2/setup.py` 中的 `-gencode` 参数。

---

## 核心控制器 API — `FrankaSimController`

位于 `scripts/fr3_controller.py`，提供 franky 风格的仿真控制接口。

### 初始化

```python
from fr3_controller import FrankaSimController

# 带 viewer
robot = FrankaSimController(viewer=True, realtime=True)

# 无头模式
robot = FrankaSimController(viewer=False, realtime=False)

# 支持 with 语句
with FrankaSimController() as robot:
    ...
```

### 属性

| 属性 | 类型 | 说明 |
|---|---|---|
| `joint_positions` | `ndarray (7,)` | 7 个关节角度 (rad) |
| `joint_velocities` | `ndarray (7,)` | 7 个关节速度 (rad/s) |
| `ee_position` | `ndarray (3,)` | 末端 pinch_site 位置 [x, y, z] |
| `ee_orientation` | `ndarray (3,3)` | 末端旋转矩阵 |
| `gripper_width` | `float` | 夹爪总开合宽度 (m) |
| `sim_time` | `float` | 仿真时间 (s) |

### 方法

```python
# 关节空间运动（线性插值）
robot.move_to_joint_position(target=[0, 0, 0, -1.57, 0, 1.57, -0.78], duration=2.0)

# 笛卡尔空间运动（阻尼最小二乘 IK）
robot.move_to_ee_position(
    target_pos=[0.28, 0, 0.77],    # 目标位置
    duration=2.0,                   # 运动时间
    damping=0.01,                   # 阻尼系数
    target_quat=[0, 1, 0, 0],      # 可选：目标姿态四元数 (w,x,y,z)
)

# 夹爪控制
robot.gripper_open(width=0.04, duration=0.5)
robot.gripper_close(width=0.0, duration=0.5)
robot.gripper_grasp(width=0.0, force=40.0, duration=0.5)

# D435i 相机图像
images = robot.get_camera_images(width=640, height=480)
# images['rgb']   — (480, 640, 3) uint8
# images['depth'] — (480, 640) float32, 单位 m

# 重置到 home 位
robot.reset()

# 关闭
robot.close()
```

---

## 脚本使用

### 1. 启动 Viewer

```bash
python scripts/view_scene.py                        # 默认 overview 相机
python scripts/view_scene.py --camera grasp_close    # 近景相机
python scripts/view_scene.py --xml my_scene.xml      # 自定义场景
```

### 2. 采集 D435i 图像

```bash
python scripts/capture_d435i.py              # 保存到 captures/
python scripts/capture_d435i.py --show       # 保存并 matplotlib 显示
python scripts/capture_d435i.py --out imgs/  # 自定义输出目录
```

输出文件：`rgb_YYYYMMDD_HHMMSS.npy`、`depth_YYYYMMDD_HHMMSS.npy`、可选 preview PNG。

### 3. 硬编码 Pick-and-Place 演示

```bash
python scripts/demo_pick_place.py
```

使用预定义关节位置的 9 步演示：张开夹爪 → 预抓取 → 下降 → 闭合 → 提升 → 相机拍照 → 放置 → 释放 → 回 home。

### 4. GraspNet 视觉抓取-放置（核心功能）

```bash
# 带可视化窗口（交互式，按 Enter 触发每次抓取）
python scripts/sim_graspnet_demo.py

# 无头模式
echo "" | python scripts/sim_graspnet_demo.py --no-viewer

# 可视化 GraspNet 预测的抓取
python scripts/sim_graspnet_demo.py --vis-grasps
```

**CLI 参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--checkpoint_path` | `franka-graspnet/.../checkpoint-rs.tar` | GraspNet 权重路径 |
| `--num_point` | `20000` | 输入点云采样数 |
| `--num_view` | `300` | GraspNet 视图数 |
| `--collision_thresh` | `0.01` | 碰撞检测阈值 |
| `--angle_threshold_deg` | `60` | 垂直方向角度阈值 |
| `--voxel_size` | `0.01` | 体素下采样大小 |
| `--width` / `--height` | `640` / `480` | 相机分辨率 |
| `--no-viewer` | — | 无头模式 |
| `--vis-grasps` | — | Open3D 可视化抓取预测 |

**管线流程：**

```
1. 初始化 MuJoCo 仿真 + 加载 GraspNet 模型
2. 回 home 位，张开夹爪
3. 循环抓取：
   a. 检测桌面剩余物体
   b. 渲染 D435i RGB-D 图像
   c. GraspNet 推理 → 验证可抓取性
   d. 深度点云物体检测 → 最近物体质心（精度 <2mm）
   e. IK 运动：预抓取位 → 下降 → 闭合夹爪 → 提升
   f. 检测已抓取物体 → 运动到容器盒上方 → 放入 → 释放
   g. 回 home 位，继续下一个物体
4. 所有物体放入盒中后自动结束
```

---

## 坐标系约定

| 坐标系 | 说明 |
|---|---|
| MuJoCo 相机 | OpenGL：X 右, Y 上, -Z 前 |
| GraspNet / 点云 | OpenCV：X 右, Y 下, Z 前 |
| 变换矩阵 | `T_cv2gl = diag(1, -1, -1)` |

---

## 技术要点

- **IK**：阻尼最小二乘法 (DLS)，支持 6-DOF（位置+姿态）控制，姿态基于四元数误差
- **抓取定位**：GraspNet 用于验证可抓取性，深度点云质心用于精确定位（解决 GraspNet 预测 ~2cm XY 偏移问题）
- **物体检测**：深度点云反投影 → 预过滤盒子区域 → 桌面上方点过滤 → `scipy.ndimage.label` 2D 网格聚类
- **多物体抓取**：自动循环检测-抓取-放置，直到所有物体放入盒中；失败自动重试
- **top-down 抓取**：`QUAT_DOWN = [0, 1, 0, 0]`（绕 X 轴旋转 180°），夹爪垂直向下