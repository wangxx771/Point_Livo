// POINT-LIVO style RGB map projection: color the accumulated world-frame
// LiDAR points from the scan-synchronized camera frame and publish/save them
// as pcl::PointXYZRGB. Ported from POINT-LIVO LIVMapper::publish_frame_world
// (LIVMapper.cpp:1832-1875) + VIOManager::getInterpolatedPixel.
//
// Decoupled from the VIO (photometric) update on purpose: it consumes the
// end-of-scan filter state directly, so it also works with camera.enable=false.
// The color lives only in this module's buffers (POINT-LIVO semantics: no
// color in the LIO map, one-shot coloring, FOV-missed points dropped).
#pragma once

#include <opencv2/core.hpp>

#include "camera_models.h"
#include "common_lib.h"

namespace point_lio
{

class RGBMapper
{
public:
  struct Config
  {
    int pub_scan_num = 1;          // color + publish every Nth scan
    double blind_rgb_points = 0.0; // min camera-frame range [m] (near gate)
    double filter_size_pcd = 0.15; // voxel leaf for the downsampled colored PCD
    bool save_raw = true;          // also write the unfiltered colored cloud
    double max_range = 0.0;        // drop points farther than this [m] (0 = off)
    int interval = -1;             // flush colored_total_ every N scans (<0 = off)
  };

  void Init(
    const Config & cfg, camera_models::AbstractCamera * cam, const Eigen::Matrix3d & R_cb,
    const Eigen::Vector3d & t_cb, const std::string & out_dir);

  // scan_world: world-frame (odom) XYZI cloud of the scan; img_color: BGR frame
  // synchronized with the scan (any size, resized to the camera dims here);
  // rot_wb/pos_wb: end-of-scan body->world state.
  void ProcessScan(
    const PointCloudXYZI & scan_world, const cv::Mat & img_color, const M3D & rot_wb,
    const V3D & pos_wb);

  const PointCloudXYZRGB & last_cloud() const { return last_pub_cloud_; }
  size_t total_points() const { return colored_total_->size(); }
  const std::string & out_dir() const { return out_dir_; }

  // Apply the PGO rigid correction (map <- odom, mirrors save_maps_to_impl)
  // and write colored_map.pcd (+ colored_map_downsampled.pcd). dir empty =
  // the output dir set at Init.
  bool Save(
    const std::string & dir, const Eigen::Matrix3d & R_corr, const Eigen::Vector3d & t_corr,
    std::string * message);

  void Clear();

private:
  cv::Vec3b sampleBilinear(const cv::Mat & img, double u, double v) const;

  Config cfg_;
  camera_models::AbstractCamera * cam_ = nullptr;  // borrowed, owner keeps it alive
  Eigen::Matrix3d Rci_ = Eigen::Matrix3d::Identity();  // camera <- IMU body
  Eigen::Vector3d Pci_ = Eigen::Vector3d::Zero();      // in camera frame
  std::string out_dir_;
  PointCloudXYZI::Ptr wait_pub_ = std::make_shared<PointCloudXYZI>();
  PointCloudXYZRGB::Ptr colored_total_ = std::make_shared<PointCloudXYZRGB>();
  PointCloudXYZRGB last_pub_cloud_;
  int scan_count_ = 0;
  int dump_index_ = 0;
  int empty_warn_count_ = 0;
};

}  // namespace point_lio
