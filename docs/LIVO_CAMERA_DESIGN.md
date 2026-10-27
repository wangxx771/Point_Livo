# point_lio 相机集成设计（FAST-LIVO2 源码移植版）

> 目标：在 `point_lio`（Point-LIO ROS2 fork）中加入相机，算法层面与
> [hku-mars/FAST-LIVO2](https://github.com/hku-mars/FAST-LIVO2) 源码一致。
> 参考论文：C. Zheng et al., *FAST-LIVO2*, IEEE T-RO 2024（arXiv:2408.14035）。

## 1. 移植方式

**v2 是 FAST-LIVO2 视觉模块的逐函数移植**（v1 为自行实现，已废弃删除）。移植的源码
文件（`C:\Users\SkyRobot\Downloads\FAST-LIVO2-main`）与本仓库文件的对应关系：

| FAST-LIVO2 源码 | 本仓库（point_lio） | 适配 |
|---|---|---|
| `include/visual_point.h` / `src/visual_point.cpp` | 同名 | Sophus SE3 → `SE3Like`（Eigen），去掉 boost |
| `include/feature.h` | 同名 | 同上 |
| `include/frame.h` / `src/frame.cpp` | 同名 | `vk::AbstractCamera` → `camera_models::AbstractCamera` |
| `include/vio.h` / `src/vio.cpp`（VIOManager） | 同名 | 见下 |
| `vk::pinhole_camera` 等（vikit 外部包） | `camera_models.h`（自足实现，同款 Pinhole-radtan / Equi 模型 + vk 语义的 `world2cam/cam2world/isInFrame/scale`） | 自足替换 |
| `vk::shiTomasiScore / interpolateMat_8u` | `vk_port::` 同名函数（`vio.cpp` 内） | 照 vikit 实现移植 |
| `StatesGroup`（19 维 EKF 状态） | IKFoM `state_output`（31 维，末尾追加 `inv_expo_time`）+ 公开的 `P_` | 状态映射见 §3 |
| `VoxelOctoTree` 平面查询（raycast / 法向细化） | `IVoxPlaneAdapter`（对 ivox 做 5 邻点 `esti_plane` 在线拟合） | LiDAR 建图仍用 ivox，零改动 |
| `LIVMapper::handleVIO / sync_packages` | `laserMapping.cpp` 主循环内"扫描中点"调用 | 扫描在图像时刻切分，见 §4 |
| `StatesGroup` 的逆曝光状态 | `state_output.inv_expo_time`（下标 30），初值 1.0 | 语义一致 |

**逐段保留**（未改写算法）：`retrieveFromVisualSparseMap`（网格化视觉子地图 + 深度连续
性剔除 + 按需 raycast）、`updateState`（金字塔粗到细 + 迭代 EKF）、
`generateVisualMapPoints`（Shi-Tomasi 网格选点）、`updateVisualMapPoints`（多观测
维护）、`updateReferencePatch`（平面细化法向 + NCC 打分选参考 patch）、
`getWarpMatrixAffine(Homography)/warpAffine/getBestSearchLevel/calculateNCC/
getImagePatch/computeProjectionJacobian`。

**未移植**（调试/展示用，运行时关闭）：`updateStateInverse`（inverse_composition 分支）、
`projectPatchFromRefToCur`（plot）、`dumpDataForColmap`、`plotTrackedPoints`、
`getInterpolatedPixel`。

## 2. 光度模型（与 FAST-LIVO2 完全一致）

```
res = inv_expo_cur · I_cur(π(X)) − inv_expo_ref · I_ref(patch)
```

- `inv_expo_time` 为状态（随机游走，初值 1.0，Q 对角 `inv_expo_cov`，初值协方差 `exposure_cov_init`）；
- 参考 patch 经仿射/单应 warp（`warpAffine`），当前 patch 按金字塔层级隔点采样（`getImagePatch`）；
- 图像梯度为隔点中心差分 × `inv_expo_cur` × `1/scale`；
- 雅可比（照源码）：`Jdphi = Jimg·Jdpi·p_hat`（p_hat = skew(X_c)）、`Jdp = −Jimg·Jdpi`、
  `JdR = Jdphi·Jdphi_dR + Jdp·Jdp_dR`、`Jdt = Jdp·Jdp_dt`、曝光列 = `I_cur`；
  常量 `Jdphi_dR = Rci`、`Jdp_dR = −Rci·skew(Pic)`、`Jdp_dt = Rci·R_wb`，其中
  `Rci = R_cbᵀ`、`Pci = −R_cbᵀ·t_cb`（相机在 body 系外参）；
- EKF 更新（照源码）：`K_1 = (HᵀH + (cov/img_point_cov)⁻¹)⁻¹`、
  `solution = −K_1·Hᵀz + vec − G·vec`（`vec = state_propagat ⊟ state`，G = K_1·HᵀH），
  误差增大回退 + 位姿增量收敛判据；金字塔从粗到细逐层迭代（`patch_pyrimid_level`）；
  全部层级结束后 `cov −= G·cov`。

**旋转扰动约定转换（关键适配）**：FAST-LIVO2 对世界→机体旋转做右乘扰动
`Rwi ← Rwi·Exp(δ)`；本仓库 IKFoM 的 boxplus 对机体→世界旋转右乘
`R_wb ← R_wb·Exp(dx)`，等价于 `Rwi ← Exp(−dx)·Rwi`。关系 `dx = −Rwi·δ`，
在填充雅可比时直接转换：`JdR ← −JdR·Rwiᵀ`（位置/曝光列不变），之后整套
`H/K_1/G/vec/solution` 均在 boxplus 约定下，`state->boxplus(solution)` 直接应用，
协方差更新 `cov −= G·cov` 同样成立。

## 3. 状态映射

| FAST-LIVO2 StatesGroup（19） | state_output（31） |
|---|---|
| rot_end(0-2)（世界→机体） | rot(3-5) 的转置（机体→世界） |
| pos_end(3-5) | pos(0-2) |
| inv_expo_time(6) | inv_expo_time(30) |
| vel/bias/gravity(7-18) | 对应块（视觉更新不触碰） |
| 无 | offset_R_L_I/offset_T_L_I/omg/acc（视觉更新不触碰） |

视觉更新只涉及 7 列：[pos(0-2), rot(3-5), inv_expo(30)]，`H` 按 `idx7 = {0,1,2,3,4,5,30}`
嵌入 31×31。**追加在末尾，LiDAR/IMU 更新路径的下标全部不变**（仅 esekfom 中 IMU
更新的硬编码 `Matrix<double,30,6>` 改为 `Matrix<scalar_type,n,6>`）。

## 4. 时序与数据流（照 FAST-LIVO2 的 LIVO 模式）

```
image_cbk（li_initialization.cpp）→ cv_bridge mono8 + 时间戳减 time_diff_cam_to_imu
        → image_deque（mutex，上限 30 帧）
sync_packages → 取时间戳落在 [扫描起, 扫描止] 的最新一帧挂到 Measures

主循环（laserMapping.cpp，kf_output 分支）:
  img_capture_time = 图像时间戳 + exposure_time_init
  逐批点：IMU 预测 → 该批 LiDAR 迭代更新（h_model 同步填充 g_pv_list：匹配点
          的世界坐标+平面法向+协方差）
    │ 批次时间 ≥ img_capture_time 时（FAST-LIVO2 的扫描切分）:
    │   ① setPropagatedState(预测后、LiDAR 更新前的状态)
    │   ② LiDAR 更新（点到图像时刻为止）
    │   ③ processFrame(img, g_pv_list): 检索 → 雅可比/EKF 更新 → 生成新点
    │      → 更新观测 → 更新参考 patch
  剩余点继续 → MapIncremental（ivox 建图，与视觉子地图独立）
  兜底：图像时间戳超出扫描末尾 → 在扫描结束时同样顺序执行一次 processFrame
```

`g_pv_list` 由 `h_model_output`/`h_model_input` 填充（FAST-LIVO2 的 `_pv_list`），
容量 24000。视觉地图点用 0.5 m 体素哈希（`feat_map`）维护，与 LiDAR 的 ivox 地图
解耦；法向细化/raycast 通过 `IVoxPlaneAdapter` 从 ivox 在线拟合平面。

## 5. 配置（mid360*.yaml 的 `camera:` 段）

参数名与默认值**与 FAST-LIVO2 一致**：`normal_en: true`、`exposure_estimate_en: true`、
`raycast_en: false`、`max_iterations: 5`、`img_point_cov: 100`、`grid_size: 5`、
`grid_n_height: 17`、`patch_size: 8`、`patch_pyrimid_level: 3`、`outlier_threshold: 1000`、
`inv_expo_cov: 0.2`、`exposure_cov_init: 0.01`、`exposure_time_init: 0.0`。
相机侧新增：`camera_model`（0=pinhole-radtan / 1=等距鱼眼）、`scale`（图像缩放系数，
内参需按此预缩放）、`dist_coeffs`、相机-in-body 外参 `extrinsic_T/R`（FAST-Calib 输出）、
`time_diff_cam_to_imu`。

## 6. 部署与验证

```bash
sudo apt install ros-$ROS_DISTRO-cv-bridge libopencv-dev
sudo apt install ros-$ROS_DISTRO-realsense2-camera   # D435i
colcon build --packages-select point_lio
ros2 launch realsense2_camera rs_launch.py rgb_camera.profile:=640x480x30
# mid360*.yaml: camera.enable: True + 内参/外参 → 正常启动 map/nav launch
```

验证顺序：先编译 → 用 FAST-LIVO2 公开数据集 bag 回放（图像话题 remap）与官方轨迹
对比 → 真机（D435i：内参取出厂 `camera_info`，外参 FAST-Calib，`scale` 用 0.5 时
fx/fy/cx/cy 同步减半）。

## 7. 与官方实现的差异清单（透明说明）

1. **旋转扰动基**：雅可比按 §2 转换到 IKFoM boxplus 约定（数学等价，非近似）；
2. **LiDAR 平面查询**：官方查 VoxelOctoTree 存储的平面；本仓库由 ivox 5 邻点在线
   重拟合（`esti_plane`，与 LiDAR 更新同款拟合）。raycast 默认关闭，影响仅在开启时；
3. **inverse_composition 分支未移植**（官方默认也为 false，走 forward）；
4. debug/plot/colmap 输出未移植；
5. ROS1 → ROS2 胶水（图像订阅、时间戳、日志）。

## 8. 参考

- [hku-mars/FAST-LIVO2](https://github.com/hku-mars/FAST-LIVO2)（ROS1，GPLv2）
- [FAST-LIVO2 论文](https://ieeexplore.ieee.org/abstract/document/10757429)（[arXiv:2408.14035](https://arxiv.org/abs/2408.14035)）
- [FAST-Calib](https://github.com/hku-mars/FAST-Calib)（LiDAR-相机外参）
- R2LIVE / FAST-LIVO（视觉子地图与光度更新的源头）
