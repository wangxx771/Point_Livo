// Camera model abstraction for the visual (photometric) update.
//
// Mirrors the camera-model API used by FAST-LIVO2 (rpg_vikit's
// vk::AbstractCamera), trimmed to the two models that cover almost all
// cameras in practice:
//   - PinholeCamera: pinhole + radtan distortion (k1, k2, p1, p2).
//     This is the plumb_bob / OpenCV model used by Intel RealSense D435i
//     (RGB and IR streams) and most pinhole cameras.
//   - EquiCamera: equidistant fisheye (k1..k4 on theta), the vk::PinholeEqui
//     convention used by FAST-LIVO2's fisheye. Covers omnidirectional cameras.
//
// Projection and unprojection come with analytic Jacobians where the
// photometric update needs them (project2d only). Pixel conventions follow
// OpenCV: origin at top-left corner, u right, v down.
#pragma once

#include <Eigen/Dense>

#include <memory>

namespace camera_models
{

enum class ModelType {
  PINHOLE = 0,  // pinhole + radtan, e.g. RealSense D435i, USB cameras
  EQUI = 1,     // equidistant fisheye, e.g. FAST-LIVO2's fisheye, 360 cameras
};

struct CameraParams
{
  int width = 640;
  int height = 480;
  double fx = 600.0;
  double fy = 600.0;
  double cx = 320.0;
  double cy = 240.0;
  // Image resize factor the intrinsics were scaled by (FAST-LIVO2's "scale":
  // full-res frames get resized by this before processing). 1.0 = full res.
  double scale = 1.0;
  // Pinhole: [k1, k2, p1, p2]; Equi: [k1, k2, k3, k4] (theta distortion).
  double d[4] = {0.0, 0.0, 0.0, 0.0};
};

class AbstractCamera
{
public:
  explicit AbstractCamera(CameraParams params)
  : params_(params)
  {
  }
  virtual ~AbstractCamera() = default;

  /// Project a 3D camera-frame point to pixels. Returns false behind the camera.
  virtual bool project2d(
    const Eigen::Vector3d & pc, Eigen::Vector2d & uv,
    Eigen::Matrix<double, 2, 3> * J = nullptr) const = 0;

  /// Unproject a pixel to a unit-norm camera-frame ray.
  virtual bool unproject2d(
    const Eigen::Vector2d & uv, Eigen::Vector3d & pc,
    Eigen::Matrix<double, 3, 2> * J = nullptr) const = 0;

  /// Pixel inside the image with a border of `border` pixels on every side.
  bool frameValid(const Eigen::Vector2d & uv, int border = 0) const
  {
    return uv(0) >= border && uv(0) < params_.width - border && uv(1) >= border &&
           uv(1) < params_.height - border;
  }

  int width() const { return params_.width; }
  int height() const { return params_.height; }
  double fx() const { return params_.fx; }
  double fy() const { return params_.fy; }
  double cx() const { return params_.cx; }
  double cy() const { return params_.cy; }
  double scale() const { return params_.scale; }

  // ---- vk::AbstractCamera-compatible helpers (FAST-LIVO2 calls) ----

  /// Project a camera-frame 3D point to pixels (distorted), vk::world2cam.
  Eigen::Vector2d world2cam(const Eigen::Vector3d & pc) const
  {
    Eigen::Vector2d uv(0.0, 0.0);
    if (!project2d(pc, uv)) uv << -1.0, -1.0;  // behind camera: vk returns invalid px
    return uv;
  }

  /// Unproject a pixel to a ray with z = 1 (vk::cam2world convention; the ray
  /// is NOT unit norm, scale it by depth / z to reach a given depth plane).
  Eigen::Vector3d cam2world(const double u, const double v) const
  {
    Eigen::Vector3d d(1.0, 1.0, 1.0);
    unproject2d(Eigen::Vector2d(u, v), d);
    return d / d(2);
  }

  Eigen::Vector3d cam2world(const Eigen::Vector2d & px) const
  {
    return cam2world(px(0), px(1));
  }

  /// Integer-pixel in-frame test with border, vk::isInFrame.
  bool isInFrame(const Eigen::Vector2i & uv, int border = 0) const
  {
    return uv(0) >= border && uv(0) < params_.width - border && uv(1) >= border &&
           uv(1) < params_.height - border;
  }

protected:
  CameraParams params_;
};

// ---------------------------------------------------------------------------
// Pinhole + radtan (plumb_bob). RealSense D435i, standard USB cameras.
// ---------------------------------------------------------------------------
class PinholeCamera : public AbstractCamera
{
public:
  explicit PinholeCamera(CameraParams params)
  : AbstractCamera(params)
  {
  }

  bool project2d(
    const Eigen::Vector3d & pc, Eigen::Vector2d & uv,
    Eigen::Matrix<double, 2, 3> * J = nullptr) const override
  {
    if (pc(2) <= 1e-8) return false;
    const double inv_z = 1.0 / pc(2);
    const double x = pc(0) * inv_z;
    const double y = pc(1) * inv_z;
    const double r2 = x * x + y * y;
    const double r4 = r2 * r2;
    const double k1 = params_.d[0], k2 = params_.d[1];
    const double p1 = params_.d[2], p2 = params_.d[3];
    const double radial = 1.0 + k1 * r2 + k2 * r4;
    const double xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    const double yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
    uv << params_.fx * xd + params_.cx, params_.fy * yd + params_.cy;

    if (J != nullptr) {
      // d(xd, yd) / d(x, y) for the radtan distortion.
      const double drdx = 2.0 * k1 * x + 4.0 * k2 * x * r2;
      const double drdy = 2.0 * k1 * y + 4.0 * k2 * y * r2;
      Eigen::Matrix2d D;
      D << radial + x * drdx + 2.0 * p1 * y + 6.0 * p2 * x, x * drdy + 2.0 * p1 * x + 2.0 * p2 * y,
        y * drdx + 2.0 * p1 * x + 2.0 * p2 * y, radial + y * drdy + 6.0 * p1 * y + 2.0 * p2 * x;
      // d(x, y) / d(pc)
      Eigen::Matrix<double, 2, 3> dxy;
      dxy << inv_z, 0.0, -x * inv_z, 0.0, inv_z, -y * inv_z;
      *J = Eigen::DiagonalMatrix<double, 2>(params_.fx, params_.fy) * D * dxy;
    }
    return true;
  }

  bool unproject2d(
    const Eigen::Vector2d & uv, Eigen::Vector3d & pc,
    Eigen::Matrix<double, 3, 2> * J = nullptr) const override
  {
    if (J != nullptr) *J = Eigen::Matrix<double, 3, 2>::Zero();  // not needed by the update
    const double x = (uv(0) - params_.cx) / params_.fx;
    const double y = (uv(1) - params_.cy) / params_.fy;
    const double k1 = params_.d[0], k2 = params_.d[1];
    const double p1 = params_.d[2], p2 = params_.d[3];
    // Fixed-point undistortion, the OpenCV iteration.
    double xu = x, yu = y;
    for (int i = 0; i < 6; i++) {
      const double r2 = xu * xu + yu * yu;
      const double radial = 1.0 + k1 * r2 + k2 * r2 * r2;
      const double dx = 2.0 * p1 * xu * yu + p2 * (r2 + 2.0 * xu * xu);
      const double dy = p1 * (r2 + 2.0 * yu * yu) + 2.0 * p2 * xu * yu;
      xu = (x - dx) / radial;
      yu = (y - dy) / radial;
    }
    pc << xu, yu, 1.0;
    pc.normalize();
    return true;
  }
};

// ---------------------------------------------------------------------------
// Equidistant fisheye. vk::PinholeEqui convention:
// theta = atan(r), theta_d = theta (1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8).
// ---------------------------------------------------------------------------
class EquiCamera : public AbstractCamera
{
public:
  explicit EquiCamera(CameraParams params)
  : AbstractCamera(params)
  {
  }

  bool project2d(
    const Eigen::Vector3d & pc, Eigen::Vector2d & uv,
    Eigen::Matrix<double, 2, 3> * J = nullptr) const override
  {
    if (pc(2) <= 1e-8) return false;
    const double inv_z = 1.0 / pc(2);
    const double x = pc(0) * inv_z;
    const double y = pc(1) * inv_z;
    const double r = std::sqrt(x * x + y * y);
    const double theta = std::atan(r);
    const double t2 = theta * theta;
    const double theta_d = theta * (1.0 + t2 * (params_.d[0] + t2 * (params_.d[1] +
                                    t2 * (params_.d[2] + t2 * params_.d[3]))));
    const double s = r > 1e-8 ? theta_d / r : params_.fx;
    uv << params_.fx * x * s + params_.cx, params_.fy * y * s + params_.cy;

    if (J != nullptr) {
      const double dtd_dt = 1.0 + t2 * (3.0 * params_.d[0] + t2 * (5.0 * params_.d[1] +
                                         t2 * (7.0 * params_.d[2] + t2 * 9.0 * params_.d[3])));
      const double dt_dr = 1.0 / (1.0 + r * r);
      const double ds_dr = r > 1e-8 ? (dtd_dt * dt_dr * r - theta_d) / (r * r) : 0.0;
      Eigen::Matrix2d D;
      D << s + x * x * ds_dr / r, x * y * ds_dr / r,
        x * y * ds_dr / r, s + y * y * ds_dr / r;
      Eigen::Matrix<double, 2, 3> dxy;
      dxy << inv_z, 0.0, -x * inv_z, 0.0, inv_z, -y * inv_z;
      *J = Eigen::DiagonalMatrix<double, 2>(params_.fx, params_.fy) * D * dxy;
    }
    return true;
  }

  bool unproject2d(
    const Eigen::Vector2d & uv, Eigen::Vector3d & pc,
    Eigen::Matrix<double, 3, 2> * J = nullptr) const override
  {
    if (J != nullptr) *J = Eigen::Matrix<double, 3, 2>::Zero();
    const double x = (uv(0) - params_.cx) / params_.fx;
    const double y = (uv(1) - params_.cy) / params_.fy;
    const double theta_d = std::sqrt(x * x + y * y);
    // Newton on theta (1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8) = theta_d.
    double theta = theta_d;
    for (int i = 0; i < 6 && theta > 1e-8; i++) {
      const double t2 = theta * theta;
      const double f = theta * (1.0 + t2 * (params_.d[0] + t2 * (params_.d[1] +
                                  t2 * (params_.d[2] + t2 * params_.d[3])))) - theta_d;
      const double df = 1.0 + t2 * (3.0 * params_.d[0] + t2 * (5.0 * params_.d[1] +
                                     t2 * (7.0 * params_.d[2] + t2 * 9.0 * params_.d[3])));
      theta -= f / df;
    }
    const double r = std::tan(theta);
    const double s = theta_d > 1e-8 ? r / theta_d : 1.0;
    pc << x * s, y * s, 1.0;
    pc.normalize();
    return true;
  }
};

inline std::shared_ptr<AbstractCamera> createCamera(ModelType model, const CameraParams & params)
{
  if (model == ModelType::EQUI) return std::make_shared<EquiCamera>(params);
  return std::make_shared<PinholeCamera>(params);
}

}  // namespace camera_models
