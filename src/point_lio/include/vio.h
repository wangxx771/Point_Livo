// Ported from hku-mars/FAST-LIVO2 (include/vio.h + src/vio.cpp), adapted to this
// ROS2 / Point-LIO codebase:
//   - vk::AbstractCamera -> camera_models::AbstractCamera (same pinhole / equi
//     models, same world2cam / cam2world / isInFrame semantics);
//   - Sophus SE3 -> SE3Like (see frame.h);
//   - StatesGroup (19-dim EKF state) -> the IKFoM esekfom state_output (31-dim,
//     pos@0..2, rot@3..5, inv_expo_time@30) with its public cov P_;
//   - the FAST-LIVO2 VoxelOctoTree plane map -> IVoxPlaneAdapter, which re-fits
//     planes from the ivox voxel map on demand (5-neighbour esti_plane, the same
//     fit the LiDAR update uses);
//   - debug/plot/colmap paths removed; inverse-composition branch not ported.
// The photometric model, update math and map-maintenance logic follow the
// FAST-LIVO2 source (updateState / retrieveFromVisualSparseMap /
// generateVisualMapPoints / updateVisualMapPoints / updateReferencePatch).
//
// FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry (T-RO 2024).
#pragma once

#include <Eigen/Dense>

#include <memory>
#include <unordered_map>
#include <vector>

#include <opencv2/opencv.hpp>

#include "camera_models.h"
#include "common_lib.h"
#include "feature.h"
#include "frame.h"
#include "parameters.h"
#include "visual_point.h"

// FAST-LIVO2 common_lib.h capacity constants (kept local to the VIO module).
#define SIZE_LARGE (500)
#define SIZE_SMALL (100)

// Minimal port of FAST-LIVO2's VoxelPlane (voxel_map.h).
struct VoxelPlane
{
  Eigen::Vector3d center_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal_ = Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 6, 6> plane_var_ = Eigen::Matrix<double, 6, 6>::Identity() * 0.0001;
  float radius_ = 0.0f;
  double d_ = 0.0;
  bool is_plane_ = false;
};

// Plane queries over the ivox voxel map, standing in for FAST-LIVO2's
// VoxelOctoTree (which this codebase does not use: the LiDAR map here is ivox).
class IVoxPlaneAdapter
{
public:
  explicit IVoxPlaneAdapter(std::shared_ptr<IVoxType> ivox)
  : ivox_(std::move(ivox))
  {
  }

  /// Re-fit the plane around p_w from the 5 nearest map points (the same fit
  /// the LiDAR update uses). Returns false when no valid plane is available.
  bool findPlane(const Eigen::Vector3d & p_w, VoxelPlane & plane);

private:
  std::shared_ptr<IVoxType> ivox_;
};

// LiDAR point handed to the VIO module (port of FAST-LIVO2's pointWithVar).
struct PointWithVar
{
  Eigen::Vector3d point_w = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal = Eigen::Vector3d::Zero();
  Eigen::Matrix3d var = Eigen::Matrix3d::Zero();
};

// Port of FAST-LIVO2's SubSparseMap.
struct SubSparseMap
{
  vector<float> propa_errors;
  vector<float> errors;
  vector<vector<float>> warp_patch;
  vector<int> search_levels;
  vector<VisualPoint *> voxel_points;
  vector<double> inv_expo_list;
  vector<PointWithVar> add_from_voxel_map;

  SubSparseMap()
  {
    propa_errors.reserve(SIZE_LARGE);
    errors.reserve(SIZE_LARGE);
    warp_patch.reserve(SIZE_LARGE);
    search_levels.reserve(SIZE_LARGE);
    voxel_points.reserve(SIZE_LARGE);
    inv_expo_list.reserve(SIZE_LARGE);
    add_from_voxel_map.reserve(SIZE_SMALL);
  };

  void reset()
  {
    propa_errors.clear();
    errors.clear();
    warp_patch.clear();
    search_levels.clear();
    voxel_points.clear();
    inv_expo_list.clear();
    add_from_voxel_map.clear();
  }
};

// Port of FAST-LIVO2's Warp cache entry.
struct Warp
{
  Matrix2d A_cur_ref;
  int search_level;
  Warp(int level, Matrix2d warp_matrix)
  : search_level(level), A_cur_ref(warp_matrix)
  {
  }
  ~Warp() {}
};

// Port of FAST-LIVO2's VOXEL_LOCATION + VOXEL_POINTS (0.5 m visual-map voxels).
struct VOXEL_LOCATION
{
  int64_t x, y, z;
  VOXEL_LOCATION(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0)
  : x(vx), y(vy), z(vz)
  {
  }
  bool operator==(const VOXEL_LOCATION & other) const
  {
    return (x == other.x && y == other.y && z == other.z);
  }
};

namespace std
{
template <>
struct hash<VOXEL_LOCATION>
{
  size_t operator()(const VOXEL_LOCATION & s) const
  {
    const int64_t hash_p = 116101;  // VOXELMAP_HASH_P
    const int64_t max_n = 10000000000;  // VOXELMAP_MAX_N
    return static_cast<size_t>(
      ((((s.z) * hash_p) % max_n + (s.y)) * hash_p) % max_n + (s.x));
  }
};
}  // namespace std

struct VOXEL_POINTS
{
  std::vector<VisualPoint *> voxel_points;
  int count;
  VOXEL_POINTS(int num)
  : count(num)
  {
  }
  ~VOXEL_POINTS()
  {
    for (VisualPoint * vp : voxel_points) {
      if (vp != nullptr) {
        delete vp;
        vp = nullptr;
      }
    }
  }
};

// Ported replacements for the vikit helpers used by FAST-LIVO2.
namespace vk_port
{
/// Shi-Tomasi corner score (vikit's shiTomasiScore): smaller eigenvalue of the
/// structure tensor over an 8x8 window.
float shiTomasiScore(const cv::Mat & img, int u, int v);

/// Bilinear interpolation of a mono8 image (vikit's interpolateMat_8u).
float interpolateMat_8u(const cv::Mat & img, float u, float v);
}  // namespace vk_port

class VIOManager
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Camera and extrinsics (camera-in-body: X_c = Rci * X_i + Pci).
  camera_models::AbstractCamera * cam;
  M3D Rci, Rcw, Jdphi_dR, Jdp_dR, Jdp_dt;
  V3D Pci, Pcw;

  // Filter state: the IKFoM esekfom state/covariance, plus the IMU-propagated
  // state snapshot used as the update prior (FAST-LIVO2's state_propagat).
  state_output * state;
  Eigen::Matrix<double, 31, 31> * cov;
  state_output state_propagat;

  // Grid bookkeeping (grid_size semantics follow FAST-LIVO2's initializeVIO).
  int grid_size;
  vector<int> grid_num;
  vector<int> map_index;
  vector<int> border_flag;
  vector<int> update_flag;
  vector<float> map_dist;
  vector<float> scan_value;
  vector<float> patch_buffer;

  bool normal_en, inverse_composition_en, exposure_estimate_en, raycast_en;
  bool ncc_en = false;

  int width, height, grid_n_width, grid_n_height, length;
  double image_resize_factor;
  double fx, fy, cx, cy;
  int patch_pyrimid_level, patch_size, patch_size_total, patch_size_half, border, warp_len;
  int max_iterations, total_points;

  double img_point_cov, outlier_threshold, ncc_thre;

  SubSparseMap * visual_submap;
  std::vector<std::vector<V3D>> rays_with_sample_points;

  double compute_jacobian_time, update_ekf_time;
  double ave_total = 0;

  int frame_count = 0;

  Matrix<double, 31, 31> G, H_T_H;
  MatrixXd K;

  unordered_map<VOXEL_LOCATION, VOXEL_POINTS *> feat_map;
  unordered_map<VOXEL_LOCATION, int> sub_feat_map;
  unordered_map<int, Warp *> warp_map;
  vector<VisualPoint *> retrieve_voxel_points;
  vector<PointWithVar> append_voxel_points;
  FramePtr new_frame_;
  std::shared_ptr<IVoxPlaneAdapter> plane_adapter_;

  enum CellType
  {
    TYPE_MAP = 1,
    TYPE_POINTCLOUD,
    TYPE_UNKNOWN
  };

  VIOManager();
  ~VIOManager();

  /// Set the camera-in-body extrinsics and precompute the constant jacobian
  /// matrices (Jdphi_dR / Jdp_dR) used by updateState.
  void setCameraExtrinsics(const Eigen::Matrix3d & R_cb, const Eigen::Vector3d & t_cb);

  /// Set the plane-query adapter over the ivox LiDAR map (raycast / reference
  /// patch normal refinement).
  void setPlaneAdapter(std::shared_ptr<IVoxPlaneAdapter> adapter)
  {
    plane_adapter_ = std::move(adapter);
  }

  void initializeVIO();
  void resetGrid();

  /// Copy the IMU-propagated state in as the update prior.
  void setPropagatedState(const state_output & s) { state_propagat = s; }

  /// FAST-LIVO2's processFrame: retrieve -> jacobian/EKF update -> generate ->
  /// update observations -> update reference patches.
  void processFrame(cv::Mat & img, vector<PointWithVar> & pg, double img_time);

  void computeJacobianAndUpdateEKF(cv::Mat img);
  void updateState(cv::Mat img, int level);
  void retrieveFromVisualSparseMap(cv::Mat img, vector<PointWithVar> & pg);
  void generateVisualMapPoints(cv::Mat img, vector<PointWithVar> & pg);
  void updateVisualMapPoints(cv::Mat img);
  void updateReferencePatch();

  void updateFrameState();
  void insertPointIntoVoxelMap(VisualPoint * pt_new);

  void computeProjectionJacobian(V3D p, MD(2, 3) & J);
  void getImagePatch(cv::Mat img, V2D pc, float * patch_tmp, int level);
  void getWarpMatrixAffine(
    const camera_models::AbstractCamera & cam, const Vector2d & px_ref, const Vector3d & f_ref,
    const double depth_ref, const SE3Like & T_cur_ref, const int level_ref, const int pyramid_level,
    const int halfpatch_size, Matrix2d & A_cur_ref);
  void getWarpMatrixAffineHomography(
    const camera_models::AbstractCamera & cam, const V2D & px_ref, const V3D & xyz_ref,
    const V3D & normal_ref, const SE3Like & T_cur_ref, const int level_ref, Matrix2d & A_cur_ref);
  void warpAffine(
    const Matrix2d & A_cur_ref, const cv::Mat & img_ref, const Vector2d & px_ref,
    const int level_ref, const int search_level, const int pyramid_level, const int halfpatch_size,
    float * patch);
  int getBestSearchLevel(const Matrix2d & A_cur_ref, const int max_level);
  double calculateNCC(float * ref_patch, float * cur_patch, int patch_size);
};
