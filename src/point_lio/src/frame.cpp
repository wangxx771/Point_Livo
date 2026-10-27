// Ported from hku-mars/FAST-LIVO2 (src/frame.cpp), adapted to ROS2 / Eigen.
//
// FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry (T-RO 2024).
#include "frame.h"

#include <algorithm>
#include <stdexcept>

int Frame::frame_counter_ = 0;

Frame::Frame(camera_models::AbstractCamera * cam, const cv::Mat & img)
: id_(frame_counter_++), cam_(cam), img_(img)
{
  initFrame(img);
}

Frame::~Frame() {}

void Frame::initFrame(const cv::Mat & img)
{
  // Check image.
  if (img.empty() || img.type() != CV_8UC1 || img.cols != cam_->width() || img.rows != cam_->height()) {
    throw std::runtime_error("Frame: provided image has not the same size as the camera model or image is not grayscale");
  }

  // Set keypoints to nullptr on initialization.
  std::for_each(fts_.begin(), fts_.end(), [&](Feature * ftr) { ftr = nullptr; });
}

namespace frame_utils
{

void createImgPyramid(const cv::Mat & img_level_0, int n_levels, ImgPyr & pyr)
{
  pyr.resize(n_levels);
  pyr[0] = img_level_0;
  for (int i = 1; i < n_levels; ++i) {
    pyr[i] = cv::Mat(pyr[i - 1].rows / 2, pyr[i - 1].cols / 2, CV_8U);
    const int area = pyr[i].rows * pyr[i].cols;
    for (int r = 0; r < pyr[i].rows; ++r) {
      const uint8_t * src = pyr[i - 1].data + r * 2 * pyr[i - 1].cols;
      uint8_t * dst = pyr[i].data + r * pyr[i].cols;
      for (int c = 0; c < pyr[i].cols; ++c, src += 2, ++dst) {
        *dst = static_cast<uint8_t>(src[0] / 4 + src[1] / 4 + src[pyr[i - 1].cols] / 4 + src[pyr[i - 1].cols + 1] / 4);
      }
    }
  }
}

} // namespace frame_utils
