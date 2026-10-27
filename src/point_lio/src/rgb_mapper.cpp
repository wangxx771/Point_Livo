#include "rgb_mapper.h"

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

#include <rclcpp/rclcpp.hpp>

namespace point_lio
{

namespace
{
rclcpp::Logger logger() { return rclcpp::get_logger("rgb_mapper"); }
}  // namespace

void RGBMapper::Init(
  const Config & cfg, camera_models::AbstractCamera * cam, const Eigen::Matrix3d & R_cb,
  const Eigen::Vector3d & t_cb, const std::string & out_dir)
{
  cfg_ = cfg;
  cam_ = cam;
  // Camera-in-body (X_b = R_cb * X_c + t_cb) -> body-in-camera, matching
  // VIOManager::setCameraExtrinsics: Rci = R_cb^T, Pci = -R_cb^T * t_cb.
  Rci_ = R_cb.transpose();
  Pci_ = -R_cb.transpose() * t_cb;
  out_dir_ = out_dir;
  Clear();
  RCLCPP_INFO(
    logger(),
    "[RGB] mapper: cam %dx%d, pub every %d scan(s), blind %.2f m, pcd leaf %.2f m, dir %s",
    cam ? cam->width() : -1, cam ? cam->height() : -1, cfg_.pub_scan_num, cfg_.blind_rgb_points,
    cfg_.filter_size_pcd, out_dir_.c_str());
}

cv::Vec3b RGBMapper::sampleBilinear(const cv::Mat & img, double u, double v) const
{
  // 4-neighbour bilinear on CV_8UC3 (BGR), POINT-LIVO getInterpolatedPixel.
  // frameValid(..., 3) upstream guarantees ui + 1 / vi + 1 stay in bounds.
  const int ui = std::floor(u);
  const int vi = std::floor(v);
  const float su = static_cast<float>(u - ui);
  const float sv = static_cast<float>(v - vi);
  const float w00 = (1.f - su) * (1.f - sv), w10 = su * (1.f - sv);
  const float w01 = (1.f - su) * sv, w11 = su * sv;
  const cv::Vec3b * row0 = img.ptr<cv::Vec3b>(vi);
  const cv::Vec3b * row1 = img.ptr<cv::Vec3b>(vi + 1);
  cv::Vec3b out;
  for (int c = 0; c < 3; c++) {
    out[c] = cv::saturate_cast<uchar>(
      w00 * row0[ui][c] + w10 * row0[ui + 1][c] + w01 * row1[ui][c] + w11 * row1[ui + 1][c]);
  }
  return out;
}

void RGBMapper::ProcessScan(
  const PointCloudXYZI & scan_world, const cv::Mat & img_color, const M3D & rot_wb,
  const V3D & pos_wb)
{
  if (cam_ == nullptr) return;
  if (img_color.empty()) {
    if (++empty_warn_count_ % 100 == 1) {
      RCLCPP_WARN(
        logger(), "[RGB] no color frame attached (seen %d times); is the camera publishing?",
        empty_warn_count_);
    }
    return;
  }

  *wait_pub_ += scan_world;
  scan_count_++;
  if (scan_count_ % cfg_.pub_scan_num != 0) return;

  // World -> camera pose from the end-of-scan state (POINT-LIVO
  // updateFrameState): Rcw = Rci * Rwb^T, Pcw = -Rci * Rwb^T * Pwb + Pci.
  const M3D Rcw = (Rci_ * rot_wb.transpose()).eval();
  const V3D Pcw = (-Rci_ * rot_wb.transpose() * pos_wb + Pci_).eval();

  // Color at the camera-model resolution (POINT-LIVO resizes in processFrame
  // before img_rgb = img.clone()).
  cv::Mat img;
  if (img_color.cols != cam_->width() || img_color.rows != cam_->height()) {
    cv::resize(
      img_color, img, cv::Size(cam_->width(), cam_->height()), 0, 0, cv::INTER_LINEAR);
  } else {
    img = img_color;
  }

  last_pub_cloud_.clear();
  last_pub_cloud_.reserve(wait_pub_->size());
  for (const PointType & p : wait_pub_->points) {
    const V3D pf = Rcw * V3D(p.x, p.y, p.z) + Pcw;
    if (pf.z() < 0) continue;  // behind the camera

    const V2D pc = cam_->world2cam(pf);
    if (!cam_->isInFrame(pc.cast<int>(), 3)) continue;

    const double range = pf.norm();
    if (range <= cfg_.blind_rgb_points) continue;
    if (cfg_.max_range > 0 && range > cfg_.max_range) continue;

    const cv::Vec3b bgr = sampleBilinear(img, pc.x(), pc.y());
    PointTypeRGB q;
    q.x = p.x;
    q.y = p.y;
    q.z = p.z;
    q.r = bgr[2];  // BGR sample -> RGB point
    q.g = bgr[1];
    q.b = bgr[0];
    last_pub_cloud_.push_back(q);
  }
  *colored_total_ += last_pub_cloud_;
  wait_pub_->clear();

  if (cfg_.interval > 0 && scan_count_ % cfg_.interval == 0 && !colored_total_->empty()) {
    const std::string path = out_dir_ + "/rgb_scans_" + std::to_string(dump_index_++) + ".pcd";
    if (pcl::io::savePCDFileBinary(path, *colored_total_) == 0) {
      RCLCPP_INFO(logger(), "[RGB] interval dump: %s (%zu pts)", path.c_str(), colored_total_->size());
    } else {
      RCLCPP_WARN(logger(), "[RGB] interval dump failed: %s", path.c_str());
    }
    colored_total_->clear();
  }
}

bool RGBMapper::Save(
  const std::string & dir_in, const Eigen::Matrix3d & R_corr, const Eigen::Vector3d & t_corr,
  std::string * message)
{
  const std::string dir = dir_in.empty() ? out_dir_ : dir_in;
  if (dir.empty()) {
    if (message) *message = "no output directory";
    return false;
  }
  if (colored_total_->empty()) {
    if (message) *message = "no colored points accumulated";
    return false;
  }

  // Apply the same final PGO rigid correction the intensity map gets
  // (save_maps_to_impl applies pg_correct once at save time).
  PointCloudXYZRGB corrected;
  corrected.reserve(colored_total_->size());
  for (const PointTypeRGB & p : colored_total_->points) {
    const Eigen::Vector3d pw = R_corr * Eigen::Vector3d(p.x, p.y, p.z) + t_corr;
    PointTypeRGB q = p;
    q.x = static_cast<float>(pw.x());
    q.y = static_cast<float>(pw.y());
    q.z = static_cast<float>(pw.z());
    corrected.push_back(q);
  }

  bool ok = true;
  std::string raw_info = "skipped (save_raw=false)";
  if (cfg_.save_raw) {
    const std::string path = dir + "/colored_map.pcd";
    if (pcl::io::savePCDFileBinary(path, corrected) == 0) {
      raw_info = path + " (" + std::to_string(corrected.size()) + " pts)";
    } else {
      raw_info = "FAILED: " + path;
      ok = false;
    }
  }

  std::string ds_info = "skipped (leaf<=0)";
  if (cfg_.filter_size_pcd > 0 && !corrected.empty()) {
    pcl::VoxelGrid<PointTypeRGB> voxel;
    voxel.setLeafSize(
      cfg_.filter_size_pcd, cfg_.filter_size_pcd, cfg_.filter_size_pcd);
    voxel.setInputCloud(std::make_shared<PointCloudXYZRGB>(corrected));
    PointCloudXYZRGB downsampled;
    voxel.filter(downsampled);
    const std::string path = dir + "/colored_map_downsampled.pcd";
    if (pcl::io::savePCDFileBinary(path, downsampled) == 0) {
      ds_info = path + " (" + std::to_string(downsampled.size()) + " pts)";
    } else {
      ds_info = "FAILED: " + path;
      ok = false;
    }
  }

  if (message) *message = raw_info + " ; " + ds_info;
  RCLCPP_INFO(logger(), "[RGB] save: %s", (message ? message->c_str() : ""));
  return ok;
}

void RGBMapper::Clear()
{
  wait_pub_->clear();
  colored_total_->clear();
  last_pub_cloud_.clear();
  scan_count_ = 0;
  dump_index_ = 0;
}

}  // namespace point_lio
