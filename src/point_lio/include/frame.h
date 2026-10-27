// Ported from hku-mars/FAST-LIVO2 (include/frame.h + vikit usage), adapted to
// ROS2 / Eigen: Sophus SE3 replaced by SE3Like (a minimal frame-to-world pose),
// vk::AbstractCamera replaced by camera_models::AbstractCamera.
//
// FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry (T-RO 2024).
#pragma once

#include <Eigen/Dense>

#include <list>
#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>

#include "camera_models.h"

// T_f_w: transform points from world to frame: X_f = R * X_w + t.
// Mirrors the Sophus SE3 API subset used by the ported FAST-LIVO2 code.
struct SE3Like
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();

  SE3Like() = default;
  SE3Like(const Eigen::Matrix3d & R_in, const Eigen::Vector3d & t_in)
  : R(R_in), t(t_in)
  {
  }

  Eigen::Vector3d operator*(const Eigen::Vector3d & xyz_w) const { return R * xyz_w + t; }
  Eigen::Matrix3d rotation_matrix() const { return R; }
  Eigen::Vector3d translation() const { return t; }

  // T_w_f
  SE3Like inverse() const
  {
    SE3Like o;
    o.R = R.transpose();
    o.t = -R.transpose() * t;
    return o;
  }

  // T_f_w1 * T_f_w2
  SE3Like operator*(const SE3Like & o) const
  {
    SE3Like r;
    r.R = R * o.R;
    r.t = R * o.t + t;
    return r;
  }
};

class VisualPoint;
struct Feature;

typedef std::list<Feature *> Features;
typedef std::vector<cv::Mat> ImgPyr;

/// A frame saves the image, the associated features and the estimated pose.
class Frame
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static int frame_counter_; //!< Counts the number of created frames. Used to set the unique id.
  int id_;                   //!< Unique id of the frame.
  camera_models::AbstractCamera *cam_;  //!< Camera model.
  SE3Like T_f_w_;             //!< Transform (f)rame from (w)orld.
  SE3Like T_f_w_prior_;       //!< Transform (f)rame from (w)orld provided by the IMU prior.
  cv::Mat img_;               //!< Image of the frame.
  Features fts_;              //!< List of features in the image.

  Frame(camera_models::AbstractCamera *cam, const cv::Mat &img);
  ~Frame();

  /// Initialize new frame and create image pyramid.
  void initFrame(const cv::Mat &img);

  /// Return number of point observations.
  inline size_t nObs() const { return fts_.size(); }

  /// Transforms point coordinates in world-frame (w) to camera pixel coordinates (c).
  inline Eigen::Vector2d w2c(const Eigen::Vector3d &xyz_w) const { return cam_->world2cam(T_f_w_ * xyz_w); }

  /// Transforms point coordinates in world-frame (w) to camera pixel coordinates (c) using the IMU prior pose.
  inline Eigen::Vector2d w2c_prior(const Eigen::Vector3d &xyz_w) const { return cam_->world2cam(T_f_w_prior_ * xyz_w); }

  /// Transforms pixel coordinates (c) to frame unit sphere coordinates (f).
  inline Eigen::Vector3d c2f(const Eigen::Vector2d &px) const { return cam_->cam2world(px[0], px[1]); }

  /// Transforms pixel coordinates (c) to frame unit sphere coordinates (f).
  inline Eigen::Vector3d c2f(const double x, const double y) const { return cam_->cam2world(x, y); }

  /// Transforms point coordinates in world-frame (w) to camera-frams (f).
  inline Eigen::Vector3d w2f(const Eigen::Vector3d &xyz_w) const { return T_f_w_ * xyz_w; }

  /// Transforms point from frame unit sphere (f) frame to world coordinate frame (w).
  inline Eigen::Vector3d f2w(const Eigen::Vector3d &f) const { return T_f_w_.inverse() * f; }

  /// Projects Point from unit sphere (f) in camera pixels (c).
  inline Eigen::Vector2d f2c(const Eigen::Vector3d &f) const { return cam_->world2cam(f); }

  /// Return the pose of the frame in the (w)orld coordinate frame.
  inline Eigen::Vector3d pos() const { return T_f_w_.inverse().translation(); }
};

typedef std::unique_ptr<Frame> FramePtr;

/// Some helper functions for the frame object.
namespace frame_utils
{

/// Creates an image pyramid of half-sampled images.
void createImgPyramid(const cv::Mat &img_level_0, int n_levels, ImgPyr &pyr);

} // namespace frame_utils
