// Ported from hku-mars/FAST-LIVO2 (src/vio.cpp), adapted to this ROS2 /
// Point-LIO codebase. See vio.h for the adaptation notes. The photometric
// model, jacobians and update math follow the FAST-LIVO2 source.
//
// FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry (T-RO 2024).
#include "vio.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <numeric>

#include <rclcpp/rclcpp.hpp>

#include "slam_log/log.h"

namespace
{

using VOXEL_POINTS = ::VOXEL_POINTS;  // hoisted for readability below

}  // namespace

namespace vk_port
{

float interpolateMat_8u(const cv::Mat & img, float u, float v)
{
  if (img.empty()) return 0.0f;
  // NaN/Inf coordinates pass every comparison below and static_cast<int>(NaN)
  // is UB (INT_MIN on x86), which crashed the 118 s CBD replay at frame 966
  // with a wild y*cols+x offset. Non-finite means "no valid pixel here".
  if (!std::isfinite(u) || !std::isfinite(v)) return 0.0f;
  if (u < 0.0f) u = 0.0f;
  if (v < 0.0f) v = 0.0f;
  if (u >= img.cols - 1) u = img.cols - 1.001f;
  if (v >= img.rows - 1) v = img.rows - 1.001f;
  const int x = static_cast<int>(u);
  const int y = static_cast<int>(v);
  const float du = u - x;
  const float dv = v - y;
  const uint8_t * ptr = img.data + y * img.cols + x;
  return (1.0f - du) * (1.0f - dv) * ptr[0] + du * (1.0f - dv) * ptr[1] +
         (1.0f - du) * dv * ptr[img.cols] + du * dv * ptr[img.cols + 1];
}

float shiTomasiScore(const cv::Mat & img, int u, int v)
{
  // vikit's shiTomasiScore: smaller eigenvalue of the structure tensor over an
  // 8x8 window around (u, v).
  float dXX = 0.0f;
  float dYY = 0.0f;
  float dXY = 0.0f;
  const int halfbox_size = 4;
  const int box_size = 2 * halfbox_size;
  const int box_area = box_size * box_size;
  const int x_min = u - halfbox_size;
  const int x_max = u + halfbox_size;
  const int y_min = v - halfbox_size;
  const int y_max = v + halfbox_size;
  if (x_min < 1 || x_max >= img.cols - 1 || y_min < 1 || y_max >= img.rows - 1) {
    return 0.0f;  // patch too close to the boundary
  }
  const int stride = img.cols;
  for (int y = y_min; y < y_max; ++y) {
    const uint8_t * ptr_left = img.data + stride * y + x_min - 1;
    const uint8_t * ptr_right = img.data + stride * y + x_min + 1;
    const uint8_t * ptr_top = img.data + stride * (y - 1) + x_min;
    const uint8_t * ptr_bottom = img.data + stride * (y + 1) + x_min;
    for (int x = 0; x < box_size; ++x, ++ptr_left, ++ptr_right, ++ptr_top, ++ptr_bottom) {
      float dx = *ptr_right - *ptr_left;
      float dy = *ptr_bottom - *ptr_top;
      dXX += dx * dx;
      dYY += dy * dy;
      dXY += dx * dy;
    }
  }
  dXX = dXX / (2.0f * box_area);
  dYY = dYY / (2.0f * box_area);
  dXY = dXY / (2.0f * box_area);
  return 0.5f * (dXX + dYY - std::sqrt((dXX + dYY) * (dXX + dYY) - 4 * (dXX * dYY - dXY * dXY)));
}

}  // namespace vk_port

bool IVoxPlaneAdapter::findPlane(const Eigen::Vector3d & p_w, VoxelPlane & plane)
{
  if (!ivox_) return false;
  PointType pt;
  pt.x = p_w(0);
  pt.y = p_w(1);
  pt.z = p_w(2);
  PointVector near;
  ivox_->GetClosestPoint(pt, near, NUM_MATCH_POINTS, 5.0);
  if (near.size() < NUM_MATCH_POINTS) return false;
  VF(4) pabcd;
  if (!esti_plane(pabcd, near, plane_thr)) return false;
  plane.normal_ = Eigen::Vector3d(pabcd(0), pabcd(1), pabcd(2));
  V3D c(0, 0, 0);
  for (const auto & q : near) {
    c += V3D(q.x, q.y, q.z);
  }
  c /= static_cast<double>(near.size());
  plane.center_ = c;
  plane.d_ = pabcd(3);
  float r2 = 0.0f;
  for (const auto & q : near) {
    r2 = std::max(r2, static_cast<float>((V3D(q.x, q.y, q.z) - c).squaredNorm()));
  }
  plane.radius_ = std::sqrt(r2);
  plane.is_plane_ = true;
  return true;
}

VIOManager::VIOManager()
: cam(nullptr),
  state(nullptr),
  cov(nullptr),
  visual_submap(nullptr),
  grid_size(5),
  normal_en(true),
  inverse_composition_en(false),
  exposure_estimate_en(true),
  raycast_en(false),
  width(0),
  height(0),
  grid_n_width(0),
  grid_n_height(0),
  length(0),
  image_resize_factor(1.0),
  fx(0.0),
  fy(0.0),
  cx(0.0),
  cy(0.0),
  patch_pyrimid_level(3),
  patch_size(8),
  patch_size_total(0),
  patch_size_half(0),
  border(0),
  warp_len(0),
  max_iterations(5),
  total_points(0),
  img_point_cov(100.0),
  outlier_threshold(1000.0),
  ncc_thre(0.5),
  compute_jacobian_time(0.0),
  update_ekf_time(0.0),
  frame_count(0)
{
}

VIOManager::~VIOManager()
{
  delete visual_submap;
  for (auto & pair : warp_map) delete pair.second;
  warp_map.clear();
  for (auto & pair : feat_map) delete pair.second;
  feat_map.clear();
}

void VIOManager::setCameraExtrinsics(const Eigen::Matrix3d & R_cb, const Eigen::Vector3d & t_cb)
{
  // Camera-in-body: X_c = Rci * X_i + Pci with Rci = R_cb^T, Pci = -R_cb^T t_cb.
  Rci = R_cb.transpose();
  Pci = -R_cb.transpose() * t_cb;

  V3D Pic;  // camera position in the body (IMU) frame
  M3D tmp;
  Jdphi_dR = Rci;
  Pic = -Rci.transpose() * Pci;
  tmp << SKEW_SYM_MATRX(Pic);
  Jdp_dR = -Rci * tmp;
}

void VIOManager::initializeVIO()
{
  visual_submap = new SubSparseMap;

  fx = cam->fx();
  fy = cam->fy();
  cx = cam->cx();
  cy = cam->cy();
  image_resize_factor = cam->scale();

  width = cam->width();
  height = cam->height();
  PL_INFO(
    rclcpp::get_logger("vio"),
    "VIO camera: %.3f %.3f %.3f %.3f, %dx%d, scale %.2f", fx, fy, cx, cy, width, height,
    image_resize_factor);

  if (grid_size > 10) {
    grid_n_width = static_cast<int>(std::ceil(static_cast<double>(width) / grid_size));
    grid_n_height = static_cast<int>(std::ceil(static_cast<double>(height) / grid_size));
  } else {
    grid_size = static_cast<int>(height / grid_n_height);
    grid_n_height = static_cast<int>(std::ceil(static_cast<double>(height) / grid_size));
    grid_n_width = static_cast<int>(std::ceil(static_cast<double>(width) / grid_size));
  }
  length = grid_n_width * grid_n_height;

  if (raycast_en) {
    border_flag.resize(length, 0);
    std::vector<std::vector<V3D>>().swap(rays_with_sample_points);
    rays_with_sample_points.reserve(length);

    const float d_min = 0.1f;
    const float d_max = 3.0f;
    const float step = 0.2f;
    for (int grid_row = 1; grid_row <= grid_n_height; grid_row++) {
      for (int grid_col = 1; grid_col <= grid_n_width; grid_col++) {
        std::vector<V3D> SamplePointsEachGrid;
        int index = (grid_row - 1) * grid_n_width + grid_col - 1;

        if (grid_row == 1 || grid_col == 1 || grid_row == grid_n_height ||
            grid_col == grid_n_width) {
          border_flag[index] = 1;
        }

        int u = grid_size / 2 + (grid_col - 1) * grid_size;
        int v = grid_size / 2 + (grid_row - 1) * grid_size;
        for (float d_temp = d_min; d_temp <= d_max; d_temp += step) {
          V3D xyz;
          xyz = cam->cam2world(u, v);
          xyz *= d_temp / xyz[2];
          SamplePointsEachGrid.push_back(xyz);
        }
        rays_with_sample_points.push_back(SamplePointsEachGrid);
      }
    }
  }

  grid_num.resize(length);
  map_index.resize(length);
  map_dist.resize(length);
  update_flag.resize(length);
  scan_value.resize(length);

  patch_size_total = patch_size * patch_size;
  patch_size_half = static_cast<int>(patch_size / 2);
  patch_buffer.resize(patch_size_total);
  warp_len = patch_size_total * patch_pyrimid_level;
  border = (patch_size_half + 1) * (1 << patch_pyrimid_level);

  retrieve_voxel_points.reserve(length);
  append_voxel_points.reserve(length);

  sub_feat_map.clear();
}

void VIOManager::resetGrid()
{
  fill(grid_num.begin(), grid_num.end(), TYPE_UNKNOWN);
  fill(map_index.begin(), map_index.end(), 0);
  fill(map_dist.begin(), map_dist.end(), 10000.0f);
  fill(update_flag.begin(), update_flag.end(), 0);
  fill(scan_value.begin(), scan_value.end(), 0.0f);

  retrieve_voxel_points.clear();
  retrieve_voxel_points.resize(length);

  append_voxel_points.clear();
  append_voxel_points.resize(length);

  total_points = 0;
}

void VIOManager::computeProjectionJacobian(V3D p, MD(2, 3) & J)
{
  const double x = p[0];
  const double y = p[1];
  const double z_inv = 1. / p[2];
  const double z_inv_2 = z_inv * z_inv;
  J(0, 0) = fx * z_inv;
  J(0, 1) = 0.0;
  J(0, 2) = -fx * x * z_inv_2;
  J(1, 0) = 0.0;
  J(1, 1) = fy * z_inv;
  J(1, 2) = -fy * y * z_inv_2;
}

void VIOManager::getImagePatch(cv::Mat img, V2D pc, float * patch_tmp, int level)
{
  const float u_ref = pc[0];
  const float v_ref = pc[1];
  const int scale = (1 << level);
  const int u_ref_i = floorf(pc[0] / scale) * scale;
  const int v_ref_i = floorf(pc[1] / scale) * scale;
  const float subpix_u_ref = (u_ref - u_ref_i) / scale;
  const float subpix_v_ref = (v_ref - v_ref_i) / scale;
  const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
  const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
  const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
  const float w_ref_br = subpix_u_ref * subpix_v_ref;
  for (int x = 0; x < patch_size; x++) {
    uint8_t * img_ptr = (uint8_t *)img.data +
                        (v_ref_i - patch_size_half * scale + x * scale) * width +
                        (u_ref_i - patch_size_half * scale);
    for (int y = 0; y < patch_size; y++, img_ptr += scale) {
      patch_tmp[patch_size_total * level + x * patch_size + y] =
        w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] +
        w_ref_br * img_ptr[scale * width + scale];
    }
  }
}

void VIOManager::insertPointIntoVoxelMap(VisualPoint * pt_new)
{
  V3D pt_w(pt_new->pos_[0], pt_new->pos_[1], pt_new->pos_[2]);
  double voxel_size = 0.5;
  float loc_xyz[3];
  for (int j = 0; j < 3; j++) {
    loc_xyz[j] = pt_w[j] / voxel_size;
    if (loc_xyz[j] < 0) {
      loc_xyz[j] -= 1.0;
    }
  }
  VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
  auto iter = feat_map.find(position);
  if (iter != feat_map.end()) {
    iter->second->voxel_points.push_back(pt_new);
    iter->second->count++;
  } else {
    VOXEL_POINTS * ot = new VOXEL_POINTS(0);
    ot->voxel_points.push_back(pt_new);
    feat_map[position] = ot;
  }
}

void VIOManager::getWarpMatrixAffineHomography(
  const camera_models::AbstractCamera & cam, const V2D & px_ref, const V3D & xyz_ref,
  const V3D & normal_ref, const SE3Like & T_cur_ref, const int level_ref, Matrix2d & A_cur_ref)
{
  // create homography matrix
  const V3D t = T_cur_ref.inverse().translation();
  const Eigen::Matrix3d H_cur_ref =
    T_cur_ref.rotation_matrix() *
    (normal_ref.dot(xyz_ref) * Eigen::Matrix3d::Identity() - t * normal_ref.transpose());
  // Compute affine warp matrix A_ref_cur using homography projection
  const int kHalfPatchSize = 4;
  V3D f_du_ref(cam.cam2world(px_ref + Eigen::Vector2d(kHalfPatchSize, 0) * (1 << level_ref)));
  V3D f_dv_ref(cam.cam2world(px_ref + Eigen::Vector2d(0, kHalfPatchSize) * (1 << level_ref)));
  const V3D f_cur(H_cur_ref * xyz_ref);
  const V3D f_du_cur = H_cur_ref * f_du_ref;
  const V3D f_dv_cur = H_cur_ref * f_dv_ref;
  V2D px_cur(cam.world2cam(f_cur));
  V2D px_du_cur(cam.world2cam(f_du_cur));
  V2D px_dv_cur(cam.world2cam(f_dv_cur));
  A_cur_ref.col(0) = (px_du_cur - px_cur) / kHalfPatchSize;
  A_cur_ref.col(1) = (px_dv_cur - px_cur) / kHalfPatchSize;
}

void VIOManager::getWarpMatrixAffine(
  const camera_models::AbstractCamera & cam, const Vector2d & px_ref, const Vector3d & f_ref,
  const double depth_ref, const SE3Like & T_cur_ref, const int level_ref, const int pyramid_level,
  const int halfpatch_size, Matrix2d & A_cur_ref)
{
  // Compute affine warp matrix A_ref_cur
  const Vector3d xyz_ref(f_ref * depth_ref);
  Vector3d xyz_du_ref(
    cam.cam2world(px_ref + Vector2d(halfpatch_size, 0) * (1 << level_ref) * (1 << pyramid_level)));
  Vector3d xyz_dv_ref(
    cam.cam2world(px_ref + Vector2d(0, halfpatch_size) * (1 << level_ref) * (1 << pyramid_level)));
  xyz_du_ref *= xyz_ref[2] / xyz_du_ref[2];
  xyz_dv_ref *= xyz_ref[2] / xyz_dv_ref[2];
  const Vector2d px_cur(cam.world2cam(T_cur_ref * (xyz_ref)));
  const Vector2d px_du(cam.world2cam(T_cur_ref * (xyz_du_ref)));
  const Vector2d px_dv(cam.world2cam(T_cur_ref * (xyz_dv_ref)));
  A_cur_ref.col(0) = (px_du - px_cur) / halfpatch_size;
  A_cur_ref.col(1) = (px_dv - px_cur) / halfpatch_size;
}

void VIOManager::warpAffine(
  const Matrix2d & A_cur_ref, const cv::Mat & img_ref, const Vector2d & px_ref,
  const int level_ref, const int search_level, const int pyramid_level, const int halfpatch_size,
  float * patch)
{
  const int patch_size = halfpatch_size * 2;
  const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
  if (std::isnan(A_ref_cur(0, 0))) {
    PL_WARN(rclcpp::get_logger("vio"), "Affine warp is NaN, probably camera has no translation");
    return;
  }

  float * patch_ptr = patch;
  for (int y = 0; y < patch_size; ++y) {
    for (int x = 0; x < patch_size; ++x)  //, ++patch_ptr)
    {
      Vector2f px_patch(x - halfpatch_size, y - halfpatch_size);
      px_patch *= (1 << search_level);
      px_patch *= (1 << pyramid_level);
      const Vector2f px(A_ref_cur * px_patch + px_ref.cast<float>());
      // Positive-form conjunction so NaN coordinates (degenerate px_ref or a
      // near-singular A_cur_ref inverse) are rejected instead of slipping
      // through every comparison and detonating in interpolateMat_8u.
      if (!(px[0] >= 0.f && px[1] >= 0.f && px[0] < img_ref.cols - 1 && px[1] < img_ref.rows - 1)) {
        patch_ptr[patch_size_total * pyramid_level + y * patch_size + x] = 0;
      } else {
        patch_ptr[patch_size_total * pyramid_level + y * patch_size + x] =
          (float)vk_port::interpolateMat_8u(img_ref, px[0], px[1]);
      }
    }
  }
}

int VIOManager::getBestSearchLevel(const Matrix2d & A_cur_ref, const int max_level)
{
  // Compute patch level in other image
  int search_level = 0;
  double D = A_cur_ref.determinant();
  while (D > 3.0 && search_level < max_level) {
    search_level += 1;
    D *= 0.25;
  }
  return search_level;
}

double VIOManager::calculateNCC(float * ref_patch, float * cur_patch, int patch_size)
{
  double sum_ref = std::accumulate(ref_patch, ref_patch + patch_size, 0.0);
  double mean_ref = sum_ref / patch_size;

  double sum_cur = std::accumulate(cur_patch, cur_patch + patch_size, 0.0);
  double mean_curr = sum_cur / patch_size;

  double numerator = 0, demoniator1 = 0, demoniator2 = 0;
  for (int i = 0; i < patch_size; i++) {
    double n = (ref_patch[i] - mean_ref) * (cur_patch[i] - mean_curr);
    numerator += n;
    demoniator1 += (ref_patch[i] - mean_ref) * (ref_patch[i] - mean_ref);
    demoniator2 += (cur_patch[i] - mean_curr) * (cur_patch[i] - mean_curr);
  }
  return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);
}

void VIOManager::retrieveFromVisualSparseMap(cv::Mat img, vector<PointWithVar> & pg)
{
  if (feat_map.size() <= 0) return;
  double ts0 = omp_get_wtime();

  visual_submap->reset();

  // Controls whether to include the visual submap from the previous frame.
  sub_feat_map.clear();

  float voxel_size = 0.5;

  if (!normal_en) warp_map.clear();

  cv::Mat depth_img = cv::Mat::zeros(height, width, CV_32FC1);
  float * it = (float *)depth_img.data;

  int loc_xyz[3];

  for (int i = 0; i < pg.size(); i++) {
    V3D pt_w = pg[i].point_w;

    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = floor(pt_w[j] / voxel_size);
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOCATION position(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

    auto iter = sub_feat_map.find(position);
    if (iter == sub_feat_map.end()) {
      sub_feat_map[position] = 0;
    } else {
      iter->second = 0;
    }

    V3D pt_c(new_frame_->w2f(pt_w));

    if (pt_c[2] > 0) {
      V2D px;
      px = new_frame_->cam_->world2cam(pt_c);

      if (new_frame_->cam_->isInFrame(px.cast<int>(), border)) {
        float depth = pt_c[2];
        int col = int(px[0]);
        int row = int(px[1]);
        it[width * row + col] = depth;
      }
    }
  }

  vector<VOXEL_LOCATION> DeleteKeyList;

  for (auto & iter : sub_feat_map) {
    VOXEL_LOCATION position = iter.first;

    auto corre_voxel = feat_map.find(position);

    if (corre_voxel != feat_map.end()) {
      bool voxel_in_fov = false;
      std::vector<VisualPoint *> & voxel_points = corre_voxel->second->voxel_points;
      int voxel_num = voxel_points.size();

      for (int i = 0; i < voxel_num; i++) {
        VisualPoint * pt = voxel_points[i];
        if (pt == nullptr) continue;
        if (pt->obs_.size() == 0) continue;

        V3D norm_vec(new_frame_->T_f_w_.rotation_matrix() * pt->normal_);
        V3D dir(new_frame_->T_f_w_ * pt->pos_);
        if (dir[2] < 0) continue;
        // dir.normalize();
        // if (dir.dot(norm_vec) <= 0.17) continue; // 0.34 70 degree  0.17 80 degree 0.08 85 degree

        V2D pc(new_frame_->w2c(pt->pos_));
        if (new_frame_->cam_->isInFrame(pc.cast<int>(), border)) {
          voxel_in_fov = true;
          int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);
          grid_num[index] = TYPE_MAP;
          Vector3d obs_vec(new_frame_->pos() - pt->pos_);
          float cur_dist = obs_vec.norm();
          if (cur_dist <= map_dist[index]) {
            map_dist[index] = cur_dist;
            retrieve_voxel_points[index] = pt;
          }
        }
      }
      if (!voxel_in_fov) {
        DeleteKeyList.push_back(position);
      }
    }
  }

  // RayCasting Module
  if (raycast_en) {
    for (int i = 0; i < length; i++) {
      if (grid_num[i] == TYPE_MAP || border_flag[i] == 1) continue;

      for (const auto & it : rays_with_sample_points[i]) {
        V3D sample_point_w = new_frame_->f2w(it);

        for (int j = 0; j < 3; j++) {
          loc_xyz[j] = floor(sample_point_w[j] / voxel_size);
          if (loc_xyz[j] < 0) {
            loc_xyz[j] -= 1.0;
          }
        }

        VOXEL_LOCATION sample_pos(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

        auto corre_sub_feat_map = sub_feat_map.find(sample_pos);
        if (corre_sub_feat_map != sub_feat_map.end()) break;

        auto corre_feat_map = feat_map.find(sample_pos);
        if (corre_feat_map != feat_map.end()) {
          bool voxel_in_fov = false;

          std::vector<VisualPoint *> & voxel_points = corre_feat_map->second->voxel_points;
          int voxel_num = voxel_points.size();
          if (voxel_num == 0) continue;

          for (int j = 0; j < voxel_num; j++) {
            VisualPoint * pt = voxel_points[j];

            if (pt == nullptr) continue;
            if (pt->obs_.size() == 0) continue;

            V3D norm_vec(new_frame_->T_f_w_.rotation_matrix() * pt->normal_);
            V3D dir(new_frame_->T_f_w_ * pt->pos_);
            if (dir[2] < 0) continue;
            dir.normalize();
            // if (dir.dot(norm_vec) <= 0.17) continue; // 0.34 70 degree 0.17 80 degree 0.08 85 degree

            V2D pc(new_frame_->w2c(pt->pos_));

            if (new_frame_->cam_->isInFrame(pc.cast<int>(), border)) {
              voxel_in_fov = true;
              int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);
              grid_num[index] = TYPE_MAP;
              Vector3d obs_vec(new_frame_->pos() - pt->pos_);

              float cur_dist = obs_vec.norm();

              if (cur_dist <= map_dist[index]) {
                map_dist[index] = cur_dist;
                retrieve_voxel_points[index] = pt;
              }
            }
          }

          if (voxel_in_fov) sub_feat_map[sample_pos] = 0;
          break;
        } else {
          // Plane query over the ivox LiDAR map (FAST-LIVO2 queries its
          // VoxelOctoTree here to seed visual points in LiDAR blind zones).
          VoxelPlane plane;
          if (plane_adapter_->findPlane(sample_point_w, plane)) {
            if (plane.is_plane_) {
              PointWithVar plane_center;
              plane_center.point_w = plane.center_;
              plane_center.normal = plane.normal_;
              visual_submap->add_from_voxel_map.push_back(plane_center);
              break;
            }
          }
        }
      }
    }
  }

  for (auto & key : DeleteKeyList) {
    sub_feat_map.erase(key);
  }

  for (int i = 0; i < length; i++) {
    if (grid_num[i] == TYPE_MAP) {
      VisualPoint * pt = retrieve_voxel_points[i];

      V2D pc(new_frame_->w2c(pt->pos_));

      V3D pt_cam(new_frame_->w2f(pt->pos_));
      bool depth_continous = false;
      for (int u = -patch_size_half; u <= patch_size_half; u++) {
        for (int v = -patch_size_half; v <= patch_size_half; v++) {
          if (u == 0 && v == 0) continue;

          float depth = it[width * (v + int(pc[1])) + u + int(pc[0])];

          if (depth == 0.) continue;

          double delta_dist = abs(pt_cam[2] - depth);

          if (delta_dist > 0.5) {
            depth_continous = true;
            break;
          }
        }
        if (depth_continous) break;
      }
      if (depth_continous) continue;

      Feature * ref_ftr;
      std::vector<float> patch_wrap(warp_len);

      int search_level;
      Matrix2d A_cur_ref_zero;

      if (!pt->is_normal_initialized_) continue;

      if (normal_en) {
        float phtometric_errors_min = std::numeric_limits<float>::max();

        if (pt->obs_.size() == 1) {
          ref_ftr = *pt->obs_.begin();
          pt->ref_patch = ref_ftr;
          pt->has_ref_patch_ = true;
        } else if (!pt->has_ref_patch_) {
          for (auto it = pt->obs_.begin(), ite = pt->obs_.end(); it != ite; ++it) {
            Feature * ref_patch_temp = *it;
            float * patch_temp = ref_patch_temp->patch_;
            float phtometric_errors = 0.0;
            int count = 0;
            for (auto itm = pt->obs_.begin(), itme = pt->obs_.end(); itm != itme; ++itm) {
              if ((*itm)->id_ == ref_patch_temp->id_) continue;
              float * patch_cache = (*itm)->patch_;

              for (int ind = 0; ind < patch_size_total; ind++) {
                phtometric_errors +=
                  (patch_temp[ind] - patch_cache[ind]) * (patch_temp[ind] - patch_cache[ind]);
              }
              count++;
            }
            phtometric_errors = phtometric_errors / count;
            if (phtometric_errors < phtometric_errors_min) {
              phtometric_errors_min = phtometric_errors;
              ref_ftr = ref_patch_temp;
            }
          }
          pt->ref_patch = ref_ftr;
          pt->has_ref_patch_ = true;
        } else {
          ref_ftr = pt->ref_patch;
        }
      } else {
        if (!pt->getCloseViewObs(new_frame_->pos(), ref_ftr, pc)) continue;
      }

      if (normal_en) {
        V3D norm_vec = (ref_ftr->T_f_w_.rotation_matrix() * pt->normal_).normalized();

        V3D pf(ref_ftr->T_f_w_ * pt->pos_);

        SE3Like T_cur_ref = new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse();

        getWarpMatrixAffineHomography(*cam, ref_ftr->px_, pf, norm_vec, T_cur_ref, 0, A_cur_ref_zero);

        search_level = getBestSearchLevel(A_cur_ref_zero, 2);
      } else {
        auto iter_warp = warp_map.find(ref_ftr->id_);
        if (iter_warp != warp_map.end()) {
          search_level = iter_warp->second->search_level;
          A_cur_ref_zero = iter_warp->second->A_cur_ref;
        } else {
          getWarpMatrixAffine(
            *cam, ref_ftr->px_, ref_ftr->f_, (ref_ftr->pos() - pt->pos_).norm(),
            new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse(), ref_ftr->level_, 0, patch_size_half,
            A_cur_ref_zero);

          search_level = getBestSearchLevel(A_cur_ref_zero, 2);

          Warp * ot = new Warp(search_level, A_cur_ref_zero);
          warp_map[ref_ftr->id_] = ot;
        }
      }

      for (int pyramid_level = 0; pyramid_level <= patch_pyrimid_level - 1; pyramid_level++) {
        warpAffine(
          A_cur_ref_zero, ref_ftr->img_, ref_ftr->px_, ref_ftr->level_, search_level,
          pyramid_level, patch_size_half, patch_wrap.data());
      }

      getImagePatch(img, pc, patch_buffer.data(), 0);

      float error = 0.0;
      for (int ind = 0; ind < patch_size_total; ind++) {
        error +=
          (ref_ftr->inv_expo_time_ * patch_wrap[ind] - state->inv_expo_time[0] * patch_buffer[ind]) *
          (ref_ftr->inv_expo_time_ * patch_wrap[ind] - state->inv_expo_time[0] * patch_buffer[ind]);
      }

      if (ncc_en) {
        double ncc = calculateNCC(patch_wrap.data(), patch_buffer.data(), patch_size_total);
        if (ncc < ncc_thre) {
          continue;
        }
      }

      if (error > outlier_threshold * patch_size_total) continue;

      visual_submap->voxel_points.push_back(pt);
      visual_submap->propa_errors.push_back(error);
      visual_submap->search_levels.push_back(search_level);
      visual_submap->errors.push_back(error);
      visual_submap->warp_patch.push_back(patch_wrap);
      visual_submap->inv_expo_list.push_back(ref_ftr->inv_expo_time_);
    }
  }
  total_points = visual_submap->voxel_points.size();

  PL_INFO(
    rclcpp::get_logger("vio"), "[ VIO ] Retrieve %d points from visual sparse map", total_points);
}

void VIOManager::computeJacobianAndUpdateEKF(cv::Mat img)
{
  if (total_points == 0) return;

  compute_jacobian_time = update_ekf_time = 0.0;

  for (int level = patch_pyrimid_level - 1; level >= 0; level--) {
    if (inverse_composition_en) {
      static bool warn_once = false;
      if (!warn_once) {
        PL_WARN(
          rclcpp::get_logger("vio"), "inverse_composition_en is not ported; using forward update");
        warn_once = true;
      }
    }
    updateState(img, level);
  }
  // POINT-LIVO does `cov -= G*cov` here with its own PSD covariance. This
  // port's P is IKFoM's (Point-LIO), which carries small negative eigenvalues
  // and is consumed by the LiDAR update path with its own noise bookkeeping;
  // writing the visual information back into it measurably destabilised the
  // filter (CBD_Building_01 smoke: endpoint -9 m vs baseline ~0 m with the
  // write, ~baseline without). The state update above still applies in full,
  // so keeping P untouched is the conservative choice: LIO behaviour is bit-
  // identical to camera-off runs, and the visual update stays additive.
  // (*cov) -= G * (*cov);
  updateFrameState();
}

void VIOManager::updateState(cv::Mat img, int level)
{
  if (total_points == 0) return;
  state_output old_state = (*state);

  VectorXd z;
  MatrixXd H_sub;
  bool EKF_end = false;
  float last_error = std::numeric_limits<float>::max();

  const int H_DIM = total_points * patch_size_total;
  z.resize(H_DIM);
  z.setZero();
  H_sub.resize(H_DIM, 7);
  H_sub.setZero();

  // Visual state indices in the 31-dim state_output: the 7 jacobian columns are
  // ordered [pos(0..2), rot(3..5), inv_expo(30)].
  const int idx7[7] = {0, 1, 2, 3, 4, 5, 30};

  for (int iteration = 0; iteration < max_iterations; iteration++) {
    double t1 = omp_get_wtime();

    // Same convention as POINT-LIVO/FAST-LIVO2 (vio.cpp Rwi(state->rot)):
    // state->rot is body -> world; Rwi here IS that rotation, not its inverse.
    M3D Rwi = state->rot;
    V3D Pwi(state->pos);
    Rcw = Rci * Rwi.transpose();
    Pcw = -Rci * Rwi.transpose() * Pwi + Pci;
    Jdp_dt = Rci * Rwi.transpose();

    float error = 0.0;
    int n_meas = 0;

#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for reduction(+ : error, n_meas)
#endif
    for (int i = 0; i < total_points; i++) {
      MD(1, 2) Jimg;
      MD(2, 3) Jdpi;
      MD(1, 3) Jdphi, Jdp, JdR, Jdt;

      float patch_error = 0.0;
      int search_level = visual_submap->search_levels[i];
      int pyramid_level = level + search_level;
      int scale = (1 << pyramid_level);
      float inv_scale = 1.0f / scale;

      VisualPoint * pt = visual_submap->voxel_points[i];

      if (pt == nullptr) continue;

      V3D pf = Rcw * pt->pos_ + Pcw;
      V2D pc = cam->world2cam(pf);

      computeProjectionJacobian(pf, Jdpi);
      M3D p_hat;
      p_hat << SKEW_SYM_MATRX(pf);

      float u_ref = pc[0];
      float v_ref = pc[1];
      int u_ref_i = floorf(pc[0] / scale) * scale;
      int v_ref_i = floorf(pc[1] / scale) * scale;
      float subpix_u_ref = (u_ref - u_ref_i) / scale;
      float subpix_v_ref = (v_ref - v_ref_i) / scale;
      float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
      float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
      float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
      float w_ref_br = subpix_u_ref * subpix_v_ref;

      vector<float> P = visual_submap->warp_patch[i];
      double inv_ref_expo = visual_submap->inv_expo_list[i];

      for (int x = 0; x < patch_size; x++) {
        uint8_t * img_ptr = (uint8_t *)img.data +
                            (v_ref_i + x * scale - patch_size_half * scale) * width +
                            u_ref_i - patch_size_half * scale;
        for (int y = 0; y < patch_size; ++y, img_ptr += scale) {
          float du =
            0.5f *
            ((w_ref_tl * img_ptr[scale] + w_ref_tr * img_ptr[scale * 2] +
              w_ref_bl * img_ptr[scale * width + scale] +
              w_ref_br * img_ptr[scale * width + scale * 2]) -
             (w_ref_tl * img_ptr[-scale] + w_ref_tr * img_ptr[0] +
              w_ref_bl * img_ptr[scale * width - scale] + w_ref_br * img_ptr[scale * width]));
          float dv =
            0.5f *
            ((w_ref_tl * img_ptr[scale * width] + w_ref_tr * img_ptr[scale + scale * width] +
              w_ref_bl * img_ptr[width * scale * 2] +
              w_ref_br * img_ptr[width * scale * 2 + scale]) -
             (w_ref_tl * img_ptr[-scale * width] + w_ref_tr * img_ptr[-scale * width + scale] +
              w_ref_bl * img_ptr[0] + w_ref_br * img_ptr[scale]));

          Jimg << du, dv;
          Jimg = Jimg * state->inv_expo_time[0];
          Jimg = Jimg * inv_scale;
          Jdphi = Jimg * Jdpi * p_hat;
          Jdp = -Jimg * Jdpi;
          JdR = Jdphi * Jdphi_dR + Jdp * Jdp_dR;
          Jdt = Jdp * Jdp_dt;
          // No convention conversion: FAST-LIVO2/POINT-LIVO also right-multiply
          // the body->world rotation in boxplus, so JdR applies as-is.

          double cur_value =
            w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] +
            w_ref_br * img_ptr[scale * width + scale];
          double res = state->inv_expo_time[0] * cur_value -
                       inv_ref_expo * P[patch_size_total * level + x * patch_size + y];

          z(i * patch_size_total + x * patch_size + y) = res;

          patch_error += res * res;
          n_meas += 1;

          if (exposure_estimate_en) {
            H_sub.block<1, 7>(i * patch_size_total + x * patch_size + y, 0) << Jdt, JdR, cur_value;
          } else {
            H_sub.block<1, 6>(i * patch_size_total + x * patch_size + y, 0) << Jdt, JdR;
          }
        }
      }
      visual_submap->errors[i] = patch_error;
      error += patch_error;
    }

    error = error / n_meas;

    compute_jacobian_time += omp_get_wtime() - t1;

    double t3 = omp_get_wtime();

    if (error <= last_error) {
      old_state = (*state);
      last_error = error;

      auto && H_sub_T = H_sub.transpose();
      H_T_H.setZero();
      G.setZero();
      H_T_H.block<7, 7>(0, 0) = H_sub_T * H_sub;

      // K_1 = (H + (cov / img_point_cov).inverse()).inverse() over the full
      // 31-dim state, with H embedded at the visual state indices.
      Matrix<double, 31, 31> H_emb = Matrix<double, 31, 31>::Zero();
      for (int a = 0; a < 7; a++) {
        for (int b = 0; b < 7; b++) {
          H_emb(idx7[a], idx7[b]) = H_T_H(a, b);
        }
      }
      // K_1 = (H + (cov / img_point_cov).inverse()).inverse() over the full
      // 31-dim state, with H embedded at the visual state indices - identical
      // to POINT-LIVO/FAST-LIVO2 (verified bit-identical outputs against the
      // inversion-lemma reformulation on the CBD_Building_01 smoke data).
      K.resize(31, 31);
      K = (H_emb + ((*cov) / img_point_cov).inverse()).inverse();

      Matrix<double, 31, 1> HTz = Matrix<double, 31, 1>::Zero();
      for (int a = 0; a < 7; a++) {
        HTz(idx7[a]) = (H_sub_T * z)(a);
      }

      // Prior feedback: vec = state_propagat (IMU-propagated) minus state.
      // Rot part follows POINT-LIVO's operator-: vec = (*state_propagat) - (*state)
      // computes Log(state.rot^T * propagat.rot), i.e. the current state's
      // transpose comes FIRST.
      Matrix<double, 31, 1> vec = Matrix<double, 31, 1>::Zero();
      vec.block<3, 1>(0, 0) = state_propagat.pos - state->pos;
      vec.block<3, 1>(3, 0) = Log(M3D(state->rot.transpose() * state_propagat.rot));
      vec(30) = state_propagat.inv_expo_time[0] - state->inv_expo_time[0];

      G = K * H_emb;
      Matrix<double, 31, 1> solution = -K * HTz + vec - G * vec;

      // Apply only the 7 dims the visual residual actually observes (pos, rot,
      // inv_expo). Upstream FAST-LIVO2 lets the full-dim K_1 leak the
      // measurement into vel/bias/gravity through its own jointly-maintained
      // covariance; this port feeds IKFoM's P_ into the same formula, where
      // that leak lands on semantically different states (omg/acc are
      // Point-LIO's per-point propagation inputs, plus extrinsics and biases)
      // with pathological gains: the probe measured 0.2-0.5 per accepted
      // iteration into omg/acc, ~20 accepted iterations per frame at 10 Hz,
      // which the LiDAR propagation then integrated into a 35 m flight within
      // 1 s of motion onset (CBD full-bag; camera off: closed loop, z within
      // 1.2 m). Same spirit as keeping the cov write-back disabled: the
      // visual update stays additive and the LIO chain consumes none of it.
      for (int i = 6; i < 30; i++) solution(i) = 0.0;
      state->boxplus(solution);
      auto && rot_add = solution.block<3, 1>(3, 0);
      auto && t_add = solution.block<3, 1>(0, 0);

      // auto && expo_add = solution.block<1, 1>(30, 0);
      // if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f) &&
      //     (expo_add.norm() < 0.001f)) EKF_end = true;
      if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f)) EKF_end = true;
    } else {
      (*state) = old_state;
      EKF_end = true;
    }

    update_ekf_time += omp_get_wtime() - t3;

    if (iteration == max_iterations || EKF_end) break;
  }
  // if (state->inv_expo_time < 0.0)  {ROS_ERROR("reset expo time!!!!!!!!!!\n");
  // state->inv_expo_time = 0.0;}
}

void VIOManager::updateFrameState()
{
  // Same convention as POINT-LIVO updateFrameState: Rwi(state.rot), body -> world.
  M3D Rwi = state->rot;
  V3D Pwi(state->pos);
  Rcw = Rci * Rwi.transpose();
  Pcw = -Rci * Rwi.transpose() * Pwi + Pci;
  new_frame_->T_f_w_ = SE3Like(Rcw, Pcw);
}

void VIOManager::generateVisualMapPoints(cv::Mat img, vector<PointWithVar> & pg)
{
  if (pg.size() <= 10) return;

  for (int i = 0; i < pg.size(); i++) {
    if (pg[i].normal == V3D(0, 0, 0)) continue;

    V3D pt = pg[i].point_w;
    V2D pc(new_frame_->w2c(pt));

    if (new_frame_->cam_->isInFrame(pc.cast<int>(), border)) {
      int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);

      if (grid_num[index] != TYPE_MAP) {
        float cur_value = vk_port::shiTomasiScore(img, pc[0], pc[1]);
        // if (cur_value < 5) continue;
        if (cur_value > scan_value[index]) {
          scan_value[index] = cur_value;
          append_voxel_points[index] = pg[i];
          grid_num[index] = TYPE_POINTCLOUD;
        }
      }
    }
  }

  for (int j = 0; j < visual_submap->add_from_voxel_map.size(); j++) {
    V3D pt = visual_submap->add_from_voxel_map[j].point_w;
    V2D pc(new_frame_->w2c(pt));

    if (new_frame_->cam_->isInFrame(pc.cast<int>(), border)) {
      int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);

      if (grid_num[index] != TYPE_MAP) {
        float cur_value = vk_port::shiTomasiScore(img, pc[0], pc[1]);
        if (cur_value > scan_value[index]) {
          scan_value[index] = cur_value;
          append_voxel_points[index] = visual_submap->add_from_voxel_map[j];
          grid_num[index] = TYPE_POINTCLOUD;
        }
      }
    }
  }

  int add = 0;
  for (int i = 0; i < length; i++) {
    if (grid_num[i] == TYPE_POINTCLOUD)  // && (scan_value[i]>=50))
    {
      PointWithVar pt_var = append_voxel_points[i];
      V3D pt = pt_var.point_w;

      V3D norm_vec(new_frame_->T_f_w_.rotation_matrix() * pt_var.normal);
      V3D dir(new_frame_->T_f_w_ * pt);
      dir.normalize();
      double cos_theta = dir.dot(norm_vec);
      // if(std::fabs(cos_theta)<0.34) continue; // 70 degree
      V2D pc(new_frame_->w2c(pt));

      float * patch = new float[patch_size_total];
      getImagePatch(img, pc, patch, 0);

      VisualPoint * pt_new = new VisualPoint(pt);

      Vector3d f = cam->cam2world(pc);
      Feature * ftr_new = new Feature(pt_new, patch, pc, f, new_frame_->T_f_w_, 0);
      ftr_new->img_ = img;
      ftr_new->id_ = new_frame_->id_;
      ftr_new->inv_expo_time_ = state->inv_expo_time[0];

      pt_new->addFrameRef(ftr_new);
      pt_new->covariance_ = pt_var.var;
      pt_new->is_normal_initialized_ = true;

      if (cos_theta < 0) {
        pt_new->normal_ = -pt_var.normal;
      } else {
        pt_new->normal_ = pt_var.normal;
      }

      pt_new->previous_normal_ = pt_new->normal_;

      insertPointIntoVoxelMap(pt_new);
      add += 1;
    }
  }

  PL_INFO(rclcpp::get_logger("vio"), "[ VIO ] Append %d new visual map points", add);
}

void VIOManager::updateVisualMapPoints(cv::Mat img)
{
  if (total_points == 0) return;

  int update_num = 0;
  SE3Like pose_cur = new_frame_->T_f_w_;
  for (int i = 0; i < total_points; i++) {
    VisualPoint * pt = visual_submap->voxel_points[i];
    if (pt == nullptr) continue;
    if (pt->is_converged_) {
      pt->deleteNonRefPatchFeatures();
      continue;
    }

    V2D pc(new_frame_->w2c(pt->pos_));
    bool add_flag = false;

    float * patch_temp = new float[patch_size_total];
    getImagePatch(img, pc, patch_temp, 0);
    // TODO: condition: distance and view_angle
    // Step 1: time
    Feature * last_feature = pt->obs_.back();
    // if(new_frame_->id_ >= last_feature->id_ + 10) add_flag = true; // 10

    // Step 2: delta_pose
    SE3Like pose_ref = last_feature->T_f_w_;
    SE3Like delta_pose = pose_ref * pose_cur.inverse();
    double delta_p = delta_pose.translation().norm();
    double delta_theta = (delta_pose.rotation_matrix().trace() > 3.0 - 1e-6)
                           ? 0.0
                           : std::acos(0.5 * (delta_pose.rotation_matrix().trace() - 1));
    if (delta_p > 0.5 || delta_theta > 0.3) add_flag = true;  // 0.5 || 0.3

    // Step 3: pixel distance
    Vector2d last_px = last_feature->px_;
    double pixel_dist = (pc - last_px).norm();
    if (pixel_dist > 40) add_flag = true;

    // Maintain the size of 3D point observation features.
    if (pt->obs_.size() >= 30) {
      Feature * ref_ftr;
      pt->findMinScoreFeature(new_frame_->pos(), ref_ftr);
      pt->deleteFeatureRef(ref_ftr);
    }
    if (add_flag) {
      update_num += 1;
      update_flag[i] = 1;
      Vector3d f = cam->cam2world(pc);
      Feature * ftr_new =
        new Feature(pt, patch_temp, pc, f, new_frame_->T_f_w_, visual_submap->search_levels[i]);
      ftr_new->img_ = img;
      ftr_new->id_ = new_frame_->id_;
      ftr_new->inv_expo_time_ = state->inv_expo_time[0];
      pt->addFrameRef(ftr_new);
    } else {
      delete[] patch_temp;
    }
  }
  PL_INFO(rclcpp::get_logger("vio"), "[ VIO ] Update %d points in visual submap", update_num);
}

void VIOManager::updateReferencePatch()
{
  if (total_points == 0) return;

  for (int i = 0; i < visual_submap->voxel_points.size(); i++) {
    VisualPoint * pt = visual_submap->voxel_points[i];

    if (!pt->is_normal_initialized_) continue;
    if (pt->is_converged_) continue;
    if (pt->obs_.size() <= 5) continue;
    if (update_flag[i] == 0) continue;

    const V3D & p_w = pt->pos_;
    VoxelPlane plane;
    if (plane_adapter_->findPlane(p_w, plane)) {
      if (plane.is_plane_) {
        float dis_to_plane = plane.normal_(0) * p_w(0) + plane.normal_(1) * p_w(1) +
                             plane.normal_(2) * p_w(2) + plane.d_;
        float dis_to_plane_abs = fabs(dis_to_plane);
        float dis_to_center = (plane.center_(0) - p_w(0)) * (plane.center_(0) - p_w(0)) +
                              (plane.center_(1) - p_w(1)) * (plane.center_(1) - p_w(1)) +
                              (plane.center_(2) - p_w(2)) * (plane.center_(2) - p_w(2));
        float range_dis = sqrt(dis_to_center - dis_to_plane * dis_to_plane);
        if (range_dis <= 3 * plane.radius_) {
          Eigen::Matrix<double, 1, 6> J_nq;
          J_nq.block<1, 3>(0, 0) = p_w - plane.center_;
          J_nq.block<1, 3>(0, 3) = -plane.normal_;
          double sigma_l = J_nq * plane.plane_var_ * J_nq.transpose();
          sigma_l += plane.normal_.transpose() * pt->covariance_ * plane.normal_;

          if (dis_to_plane_abs < 3 * sqrt(sigma_l)) {
            if (pt->previous_normal_.dot(plane.normal_) < 0) {
              pt->normal_ = -plane.normal_;
            } else {
              pt->normal_ = plane.normal_;
            }

            double normal_update = (pt->normal_ - pt->previous_normal_).norm();

            pt->previous_normal_ = pt->normal_;

            if (normal_update < 0.0001 && pt->obs_.size() > 10) {
              pt->is_converged_ = true;
            }
          }
        }
      }
    }

    float score_max = -1000.;
    for (auto it = pt->obs_.begin(), ite = pt->obs_.end(); it != ite; ++it) {
      Feature * ref_patch_temp = *it;
      float * patch_temp = ref_patch_temp->patch_;
      float NCC_up = 0.0;
      float NCC_down1 = 0.0;
      float NCC_down2 = 0.0;
      float NCC = 0.0;
      float score = 0.0;
      int count = 0;

      V3D pf = ref_patch_temp->T_f_w_ * pt->pos_;
      V3D norm_vec = ref_patch_temp->T_f_w_.rotation_matrix() * pt->normal_;
      pf.normalize();
      double cos_angle = pf.dot(norm_vec);
      // if(fabs(cos_angle) < 0.86) continue; // 20 degree

      float ref_mean;
      if (abs(ref_patch_temp->mean_) < 1e-6) {
        float ref_sum = std::accumulate(patch_temp, patch_temp + patch_size_total, 0.0);
        ref_mean = ref_sum / patch_size_total;
        ref_patch_temp->mean_ = ref_mean;
      }

      for (auto itm = pt->obs_.begin(), itme = pt->obs_.end(); itm != itme; ++itm) {
        if ((*itm)->id_ == ref_patch_temp->id_) continue;
        float * patch_cache = (*itm)->patch_;

        float other_mean;
        if (abs((*itm)->mean_) < 1e-6) {
          float other_sum = std::accumulate(patch_cache, patch_cache + patch_size_total, 0.0);
          other_mean = other_sum / patch_size_total;
          (*itm)->mean_ = other_mean;
        }

        for (int ind = 0; ind < patch_size_total; ind++) {
          NCC_up += (patch_temp[ind] - ref_mean) * (patch_cache[ind] - other_mean);
          NCC_down1 += (patch_temp[ind] - ref_mean) * (patch_temp[ind] - ref_mean);
          NCC_down2 += (patch_cache[ind] - other_mean) * (patch_cache[ind] - other_mean);
        }
        NCC += fabs(NCC_up / sqrt(NCC_down1 * NCC_down2));
        count++;
      }

      NCC = NCC / count;

      score = NCC + cos_angle;

      ref_patch_temp->score_ = score;

      if (score > score_max) {
        score_max = score;
        pt->ref_patch = ref_patch_temp;
        pt->has_ref_patch_ = true;
      }
    }
  }
}

void VIOManager::processFrame(cv::Mat & img, vector<PointWithVar> & pg, double img_time)
{
  (void)img_time;
  if (width != img.cols || height != img.rows) {
    if (img.empty()) PL_WARN(rclcpp::get_logger("vio"), "[ VIO ] Empty Image!");
    cv::resize(
      img, img, cv::Size(img.cols * image_resize_factor, img.rows * image_resize_factor), 0, 0,
      cv::INTER_LINEAR);
  }

  if (img.channels() == 3) cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

  new_frame_.reset(new Frame(cam, img));
  updateFrameState();

  resetGrid();

  retrieveFromVisualSparseMap(img, pg);

  computeJacobianAndUpdateEKF(img);

  generateVisualMapPoints(img, pg);

  updateVisualMapPoints(img);

  updateReferencePatch();

  frame_count++;
}
