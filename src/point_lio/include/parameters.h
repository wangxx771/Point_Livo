// #ifndef PARAM_H
// #define PARAM_H
#pragma once
#include <ivox/ivox3d.h>
#include <math.h>
#include <omp.h>
#include <pcl/common/transforms.h>
#include <unistd.h>

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <fstream>
#include <geometry_msgs/msg/vector3.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <thread>

#include "IMU_Processing.h"
#include "preprocess.h"

using IVoxType = faster_lio::IVox<3, faster_lio::IVoxNodeType::DEFAULT, PointType>;

extern bool is_first_frame;
extern double lidar_end_time, first_lidar_time, time_con;
extern double last_timestamp_lidar, last_timestamp_imu;
extern int pcd_index;
extern IVoxType::Options ivox_options_;
extern int ivox_nearby_type;
extern state_input state_in;
extern state_output state_out;
extern std::string lid_topic, imu_topic;
extern bool prop_at_freq_of_imu, check_satu, con_frame, cut_frame;
extern bool use_imu_as_input, space_down_sample;
extern bool extrinsic_est_en, publish_odometry_without_downsample;
extern bool imu_propagate_en;
extern int init_map_size, con_frame_num;
extern double match_s, satu_acc, satu_gyro, cut_frame_time_interval;
extern float plane_thr;
extern double filter_size_surf_min, filter_size_map_min;
extern bool imu_en;
extern double imu_time_inte;
extern bool zc_enable_param;
extern double zc_nz2_thresh_param, zc_weight_param, zc_max_hold_s_param;
extern double zc_max_tilt_deg_param;
extern double laser_point_cov, acc_norm;
extern double acc_cov_input, gyr_cov_input, vel_cov;
extern double gyr_cov_output, acc_cov_output, b_gyr_cov, b_acc_cov;
extern double imu_meas_acc_cov, imu_meas_omg_cov;
extern int lidar_type, pcd_save_interval;
extern bool pcd_tiled_en;
extern bool pcd_run_subdir;
extern double pcd_tile_size;
extern double pcd_tile_flush_idle_s;
extern double pcd_tile_voxel;
extern double pcd_max_range;
extern int pcd_tile_max_buffer_pts;
extern std::string pcd_tile_policy;
extern std::string pcd_save_dir;
extern std::string pcd_save_name;
extern std::vector<double> gravity_init, gravity;
extern bool runtime_pos_log, pcd_save_en, path_en;
extern bool scan_pub_en, scan_body_pub_en, tf_send_en;
extern bool map_pub_en;
extern double map_pub_interval;
extern int path_max_poses;
extern double path_pub_interval;
extern shared_ptr<Preprocess> p_pre;
extern shared_ptr<ImuProcess> p_imu;
extern bool is_first_frame;

extern std::vector<double> extrinT;
extern std::vector<double> extrinR;
extern double time_diff_lidar_to_imu;
extern double imu_init_sec;
extern double imu_init_acc_ratio;
extern double imu_init_timeout;   // Give up waiting for still samples after this (s)
extern bool imu_init_dynamic;
extern double imu_init_gyr_tol;
extern double lidar_time_inte, first_imu_time;
extern int cut_frame_num;
extern bool cut_frame_init;
extern double time_update_last, time_current, time_predict_last_const, t_last;

extern bool enable_prior_pcd;
extern string prior_pcd_map_path;
extern std::vector<double> init_pose;

extern MeasureGroup Measures;
extern bool prior_pcd_freeze;
extern bool loop_en;
extern bool gnss_en;

extern double odom_max_vel;
extern int ivox_max_pts_per_node;

void readParameters(std::shared_ptr<rclcpp::Node> & n);
extern ofstream fout_out;
void open_file(const std::string & dir);
Eigen::Matrix<double, 3, 1> SO3ToEuler(const SO3 & orient);
void reset_cov(Eigen::Matrix<double, 24, 24> & P_init);
void reset_cov_output(Eigen::Matrix<double, 31, 31> & P_init_output);

// ---- Camera / visual update (see config/*.yaml, section "camera") ----
// Parameter names/defaults follow FAST-LIVO2 (LIVMapper::readParameters).
extern bool camera_enable;
extern std::string camera_image_topic;
extern int camera_model_id;      // 0 = pinhole (radtan), 1 = equidistant fisheye
extern int camera_width, camera_height;
extern double camera_fx, camera_fy, camera_cx, camera_cy;
extern double camera_scale;      // image resize factor the intrinsics were scaled by
extern std::vector<double> camera_dist_coeffs;   // 4 coefficients
extern std::vector<double> camera_extrinT;       // camera in body (IMU) frame
extern std::vector<double> camera_extrinR;       // 9 elements, row major
extern double time_diff_cam_to_imu;              // subtracted from image stamps
extern double exposure_time_init;  // added to the image stamp => capture time (mid-exposure)
extern bool vis_normal_en;
extern bool vis_inverse_composition_en;  // not ported: forward update only
extern bool vis_exposure_estimate_en;
extern bool vis_raycast_en;
extern bool vis_ncc_en;
extern int vis_max_iterations;
extern int vis_grid_size, vis_grid_n_height;
extern int vis_patch_pyrimid_level, vis_patch_size;
extern double vis_img_point_cov, vis_outlier_threshold, vis_ncc_thre;
extern double exposure_cov;          // inv-exposure random-walk variance (FAST-LIVO2 inv_expo_cov)
extern double exposure_cov_init;     // initial inv-exposure state variance

// ---- RGB map (POINT-LIVO style colored cloud; section "rgb_map") ----
extern bool rgb_map_en;
extern int rgb_pub_scan_num;         // color + publish every Nth scan
extern double rgb_blind_points;      // min camera-frame range [m]
extern double rgb_filter_size_pcd;   // voxel leaf of the downsampled colored PCD
extern bool rgb_save_raw;            // also write the unfiltered colored cloud
extern double rgb_max_range;         // drop points farther than this [m] (0 = off)
extern int rgb_interval;             // flush the colored buffer every N scans (<0 = off)