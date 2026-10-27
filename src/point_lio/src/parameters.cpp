#include "slam_log/log.h"
#include <cmath>
#include "parameters.h"
#include "li_initialization.h"   // time_lag_IMU_wtr_lidar

bool is_first_frame = true;
double lidar_end_time = 0.0, first_lidar_time = 0.0, time_con = 0.0;
double last_timestamp_lidar = -1.0, last_timestamp_imu = -1.0;
int pcd_index = 0;
IVoxType::Options ivox_options_;
int ivox_nearby_type = 6;

std::vector<double> extrinT(3, 0.0);
std::vector<double> extrinR(9, 0.0);
state_input state_in;
state_output state_out;
std::string lid_topic, imu_topic;
bool prop_at_freq_of_imu = true, check_satu = true, con_frame = false, cut_frame = false;
bool use_imu_as_input = false, space_down_sample = true,
     publish_odometry_without_downsample = false;
bool imu_propagate_en = false;
int init_map_size = 10, con_frame_num = 1;
double match_s = 81, satu_acc, satu_gyro, cut_frame_time_interval = 0.1;
float plane_thr = 0.1f;
double filter_size_surf_min = 0.5, filter_size_map_min = 0.5;
bool imu_en = true;
double imu_time_inte = 0.005;
double laser_point_cov = 0.01, acc_norm;
bool zc_enable_param = true;
double zc_nz2_thresh_param = 0.12, zc_weight_param = 2.5, zc_max_hold_s_param = 60.0;
double zc_max_tilt_deg_param = 3.0;
double vel_cov, acc_cov_input, gyr_cov_input;
double gyr_cov_output, acc_cov_output, b_gyr_cov, b_acc_cov;
double imu_meas_acc_cov, imu_meas_omg_cov;
int lidar_type, pcd_save_interval;
bool pcd_tiled_en = true;
bool pcd_run_subdir = true;
double pcd_tile_size = 30.0, pcd_tile_flush_idle_s = 10.0, pcd_tile_voxel = 0.0;
double pcd_max_range = 0.0;
int pcd_tile_max_buffer_pts = 500000;
std::string pcd_tile_policy = "centroid";
std::string pcd_save_dir;
std::string pcd_save_name;
std::vector<double> gravity_init, gravity;
bool runtime_pos_log, pcd_save_en, path_en, extrinsic_est_en = true;
bool scan_pub_en, scan_body_pub_en, tf_send_en;
bool map_pub_en = false;          // opt-in: mapping configs turn it on
double map_pub_interval = 1.0;    // seconds; 0 = every frame
int path_max_poses = 0;           // 0 = keep the whole trajectory
double path_pub_interval = 1.0;   // seconds; 0 = every frame
shared_ptr<Preprocess> p_pre;
shared_ptr<ImuProcess> p_imu;
double time_update_last = 0.0, time_current = 0.0, time_predict_last_const = 0.0, t_last = 0.0;
double time_diff_lidar_to_imu = 0.0;
double imu_init_sec = 1.0, imu_init_acc_ratio = 0.035, imu_init_gyr_tol = 0.05;
double imu_init_timeout = 3.0;
bool imu_init_dynamic = false;

bool enable_prior_pcd;
string prior_pcd_map_path;
std::vector<double> init_pose;

double lidar_time_inte = 0.1, first_imu_time = 0.0;
int cut_frame_num = 1;
bool cut_frame_init = false;       // true;

MeasureGroup Measures;

bool loop_en = false;
bool prior_pcd_freeze = false;
bool gnss_en = false;

double odom_max_vel = 5.0;
int ivox_max_pts_per_node = 40;

// ---- Camera / visual update (defaults follow FAST-LIVO2) ----
bool camera_enable = false;
std::string camera_image_topic = "/camera/color/image_raw";
int camera_model_id = 0;
int camera_width = 640, camera_height = 480;
double camera_fx = 600.0, camera_fy = 600.0, camera_cx = 320.0, camera_cy = 240.0;
double camera_scale = 1.0;
std::vector<double> camera_dist_coeffs(4, 0.0);
std::vector<double> camera_extrinT(3, 0.0);
std::vector<double> camera_extrinR(9, 0.0);
double time_diff_cam_to_imu = 0.0;
double exposure_time_init = 0.0;
bool vis_normal_en = true;
bool vis_inverse_composition_en = false;
bool vis_exposure_estimate_en = true;
bool vis_raycast_en = false;
bool vis_ncc_en = false;
int vis_max_iterations = 5;
int vis_grid_size = 5, vis_grid_n_height = 17;
int vis_patch_pyrimid_level = 3, vis_patch_size = 8;
double vis_img_point_cov = 100.0, vis_outlier_threshold = 1000.0, vis_ncc_thre = 0.5;
double exposure_cov = 0.2, exposure_cov_init = 0.01;

// ---- RGB map (POINT-LIVO style colored cloud projection; see rgb_mapper.h) ----
bool rgb_map_en = false;
int rgb_pub_scan_num = 1;
double rgb_blind_points = 0.0;
double rgb_filter_size_pcd = 0.15;
bool rgb_save_raw = true;
double rgb_max_range = 0.0;
int rgb_interval = -1;


void readParameters(std::shared_ptr<rclcpp::Node> & nh)
{
  p_pre.reset(new Preprocess());
  p_imu.reset(new ImuProcess());
  try {
    nh->declare_parameter<bool>("prop_at_freq_of_imu", true);
    nh->get_parameter("prop_at_freq_of_imu", prop_at_freq_of_imu);

    nh->declare_parameter<bool>("use_imu_as_input", false);
    nh->get_parameter("use_imu_as_input", use_imu_as_input);

    nh->declare_parameter<bool>("check_satu", true);
    nh->get_parameter("check_satu", check_satu);

    nh->declare_parameter<int>("init_map_size", 100);
    nh->get_parameter("init_map_size", init_map_size);

    nh->declare_parameter<bool>("space_down_sample", true);
    nh->get_parameter("space_down_sample", space_down_sample);

    nh->declare_parameter<double>("mapping.satu_acc", 3.0);
    nh->get_parameter("mapping.satu_acc", satu_acc);

    nh->declare_parameter<double>("mapping.satu_gyro", 35.0);
    nh->get_parameter("mapping.satu_gyro", satu_gyro);

    nh->declare_parameter<double>("mapping.acc_norm", 1.0);
    nh->get_parameter("mapping.acc_norm", acc_norm);

    nh->declare_parameter<float>("mapping.plane_thr", 0.05f);
    nh->get_parameter("mapping.plane_thr", plane_thr);

    nh->declare_parameter<int>("point_filter_num", 2);
    nh->get_parameter("point_filter_num", p_pre->point_filter_num);

    nh->declare_parameter<std::string>("common.lid_topic", ".livox.lidar");
    nh->get_parameter("common.lid_topic", lid_topic);

    nh->declare_parameter<std::string>("common.imu_topic", ".livox.imu");
    nh->get_parameter("common.imu_topic", imu_topic);

    nh->declare_parameter<bool>("common.con_frame", false);
    nh->get_parameter("common.con_frame", con_frame);

    nh->declare_parameter<int>("common.con_frame_num", 1);
    nh->get_parameter("common.con_frame_num", con_frame_num);

    nh->declare_parameter<bool>("common.cut_frame", false);
    nh->get_parameter("common.cut_frame", cut_frame);

    nh->declare_parameter<double>("common.cut_frame_time_interval", 0.1);
    nh->get_parameter("common.cut_frame_time_interval", cut_frame_time_interval);

    // Still-detection tolerances. 0.035 of gravity (0.35 m/s^2) excludes the CARLA
    // touchdown transient, which measured |a| = 10.124, while sitting far above
    // sensor noise at rest; gyr_tol 0.05 rad/s rejects the same transient.
    nh->declare_parameter<double>("mapping.imu_init_sec", 1.0);
    nh->get_parameter("mapping.imu_init_sec", imu_init_sec);
    nh->declare_parameter<double>("mapping.imu_init_acc_ratio", 0.035);
    nh->get_parameter("mapping.imu_init_acc_ratio", imu_init_acc_ratio);
    // Dynamic start: for platforms already moving at power-on (vehicles that must
    // depart immediately). The cost is that gravity rests entirely on gravity_init
    // being right; whatever it is off by, the result tilts by.
    nh->declare_parameter<bool>("mapping.imu_init_dynamic", false);
    nh->get_parameter("mapping.imu_init_dynamic", imu_init_dynamic);
    nh->declare_parameter<double>("mapping.imu_init_timeout", 3.0);
    nh->get_parameter("mapping.imu_init_timeout", imu_init_timeout);
    nh->declare_parameter<double>("mapping.imu_init_gyr_tol", 0.05);
    nh->get_parameter("mapping.imu_init_gyr_tol", imu_init_gyr_tol);

    // Lidar-IMU time offset, subtracted from the IMU stamp as upstream FAST-LIO
    // does. The subtraction in imu_cbk was always there but nothing ever assigned
    // these variables, so the calibrated -0.01245161 in mid360.yaml was ignored.
    nh->declare_parameter<double>("common.time_diff_lidar_to_imu", 0.0);
    nh->get_parameter("common.time_diff_lidar_to_imu", time_diff_lidar_to_imu);
    time_lag_IMU_wtr_lidar = time_diff_lidar_to_imu;

    nh->declare_parameter<bool>("prior_pcd.enable", false);
    nh->get_parameter("prior_pcd.enable", enable_prior_pcd);

    nh->declare_parameter<string>("prior_pcd.prior_pcd_map_path", "");
    nh->get_parameter("prior_pcd.prior_pcd_map_path", prior_pcd_map_path);

    nh->declare_parameter<std::vector<double>>("prior_pcd.init_pose", std::vector<double>());
    nh->get_parameter("prior_pcd.init_pose", init_pose);

    nh->declare_parameter<double>("filter_size_surf", 0.5);
    nh->get_parameter("filter_size_surf", filter_size_surf_min);

    nh->declare_parameter<double>("filter_size_map", 0.3);
    nh->get_parameter("filter_size_map", filter_size_map_min);

    nh->declare_parameter<bool>("mapping.imu_en", true);
    nh->get_parameter("mapping.imu_en", imu_en);

    // Vertical degeneracy constraint: z is unobservable when only vertical walls are
    // in view, so a latched height holds it. The test is the mean square z
    // component of matched plane normals; stair treads push it well above the
    // threshold, so climbing is unaffected.
    nh->declare_parameter<bool>("mapping.z_constraint.enable", true);
    nh->get_parameter("mapping.z_constraint.enable", zc_enable_param);
    nh->declare_parameter<double>("mapping.z_constraint.nz2_thresh", 0.12);
    nh->get_parameter("mapping.z_constraint.nz2_thresh", zc_nz2_thresh_param);
    nh->declare_parameter<double>("mapping.z_constraint.weight", 2.5);
    nh->get_parameter("mapping.z_constraint.weight", zc_weight_param);
    nh->declare_parameter<double>("mapping.z_constraint.max_hold_s", 60.0);
    nh->get_parameter("mapping.z_constraint.max_hold_s", zc_max_hold_s_param);
    nh->declare_parameter<double>("mapping.z_constraint.max_tilt_deg", 3.0);
    nh->get_parameter("mapping.z_constraint.max_tilt_deg", zc_max_tilt_deg_param);

    nh->declare_parameter<bool>("mapping.extrinsic_est_en", true);
    nh->get_parameter("mapping.extrinsic_est_en", extrinsic_est_en);

    nh->declare_parameter<double>("mapping.imu_time_inte", 0.005);
    nh->get_parameter("mapping.imu_time_inte", imu_time_inte);

    nh->declare_parameter<double>("mapping.lidar_meas_cov", 0.1);
    nh->get_parameter("mapping.lidar_meas_cov", laser_point_cov);

    nh->declare_parameter<double>("mapping.acc_cov_input", 0.1);
    nh->get_parameter("mapping.acc_cov_input", acc_cov_input);

    nh->declare_parameter<double>("mapping.vel_cov", 20);
    nh->get_parameter("mapping.vel_cov", vel_cov);

    nh->declare_parameter<double>("mapping.gyr_cov_input", 0.1);
    nh->get_parameter("mapping.gyr_cov_input", gyr_cov_input);

    nh->declare_parameter<double>("mapping.gyr_cov_output", 0.1);
    nh->get_parameter("mapping.gyr_cov_output", gyr_cov_output);

    nh->declare_parameter<double>("mapping.acc_cov_output", 0.1);
    nh->get_parameter("mapping.acc_cov_output", acc_cov_output);

    nh->declare_parameter<double>("mapping.b_gyr_cov", 0.0001);
    nh->get_parameter("mapping.b_gyr_cov", b_gyr_cov);

    nh->declare_parameter<double>("mapping.b_acc_cov", 0.0001);
    nh->get_parameter("mapping.b_acc_cov", b_acc_cov);

    nh->declare_parameter<double>("mapping.imu_meas_acc_cov", 0.1);
    nh->get_parameter("mapping.imu_meas_acc_cov", imu_meas_acc_cov);

    nh->declare_parameter<double>("mapping.imu_meas_omg_cov", 0.1);
    nh->get_parameter("mapping.imu_meas_omg_cov", imu_meas_omg_cov);

    nh->declare_parameter<double>("preprocess.blind", 1.0);
    nh->get_parameter("preprocess.blind", p_pre->blind);

    nh->declare_parameter<int>("preprocess.lidar_type", 1);
    nh->get_parameter("preprocess.lidar_type", lidar_type);

    nh->declare_parameter<int>("preprocess.scan_line", 16);
    nh->get_parameter("preprocess.scan_line", p_pre->N_SCANS);

    nh->declare_parameter<int>("preprocess.scan_rate", 10);
    nh->get_parameter("preprocess.scan_rate", p_pre->SCAN_RATE);

    nh->declare_parameter<int>("preprocess.timestamp_unit", 1);
    nh->get_parameter("preprocess.timestamp_unit", p_pre->time_unit);

    nh->declare_parameter<double>("mapping.match_s", 81);
    nh->get_parameter("mapping.match_s", match_s);

    nh->declare_parameter<std::vector<double>>("mapping.gravity", std::vector<double>());
    nh->get_parameter("mapping.gravity", gravity);

    nh->declare_parameter<std::vector<double>>("mapping.gravity_init", std::vector<double>());
    nh->get_parameter("mapping.gravity_init", gravity_init);

    nh->declare_parameter<std::vector<double>>("mapping.extrinsic_T", std::vector<double>());
    nh->get_parameter("mapping.extrinsic_T", extrinT);

    nh->declare_parameter<std::vector<double>>("mapping.extrinsic_R", std::vector<double>());
    nh->get_parameter("mapping.extrinsic_R", extrinR);

    nh->declare_parameter<bool>("odometry.publish_odometry_without_downsample", false);
    nh->get_parameter(
      "odometry.publish_odometry_without_downsample", publish_odometry_without_downsample);

    nh->declare_parameter<bool>("odometry.imu_propagate_en", false);
    nh->get_parameter("odometry.imu_propagate_en", imu_propagate_en);

    nh->declare_parameter<bool>("publish.path_en", true);
    nh->get_parameter("publish.path_en", path_en);

    nh->declare_parameter<bool>("publish.scan_publish_en", true);
    nh->get_parameter("publish.scan_publish_en", scan_pub_en);

    nh->declare_parameter<bool>("publish.scan_bodyframe_pub_en", true);
    nh->get_parameter("publish.scan_bodyframe_pub_en", scan_body_pub_en);

    nh->declare_parameter<bool>("publish.tf_send_en", true);
    nh->get_parameter("publish.tf_send_en", tf_send_en);

    nh->declare_parameter<bool>("publish.map_pub_en", false);
    nh->get_parameter("publish.map_pub_en", map_pub_en);

    nh->declare_parameter<double>("publish.map_pub_interval", 1.0);
    nh->get_parameter("publish.map_pub_interval", map_pub_interval);

    nh->declare_parameter<int>("publish.path_max_poses", 0);
    nh->get_parameter("publish.path_max_poses", path_max_poses);

    nh->declare_parameter<double>("publish.path_pub_interval", 1.0);
    nh->get_parameter("publish.path_pub_interval", path_pub_interval);

    nh->declare_parameter<bool>("runtime_pos_log_enable", false);
    nh->get_parameter("runtime_pos_log_enable", runtime_pos_log);

    nh->declare_parameter<bool>("pcd_save.pcd_save_en", false);
    nh->get_parameter("pcd_save.pcd_save_en", pcd_save_en);

    nh->declare_parameter<int>("pcd_save.interval", -1);
    nh->get_parameter("pcd_save.interval", pcd_save_interval);


    nh->declare_parameter<std::string>("pcd_save.output_dir", "");
    nh->get_parameter("pcd_save.output_dir", pcd_save_dir);

    // By default create map_<timestamp>/ under output_dir and repoint the latest
    // symlink, so two consecutive runs do not silently overwrite each other (tile
    // names depend only on tile coordinates and index, not on the run). Set false
    // when the caller already makes a per-session directory and manages latest,
    // otherwise timestamps nest twice and latest lands on the middle level.
    nh->declare_parameter<bool>("pcd_save.run_subdir", true);
    nh->get_parameter("pcd_save.run_subdir", pcd_run_subdir);

    nh->declare_parameter<string>("pcd_save.file_name", "map.pcd");
    nh->get_parameter("pcd_save.file_name", pcd_save_name);
    nh->declare_parameter<bool>("pcd_save.tiled", true);
    nh->get_parameter("pcd_save.tiled", pcd_tiled_en);

    nh->declare_parameter<double>("pcd_save.tile_size", 30.0);
    nh->get_parameter("pcd_save.tile_size", pcd_tile_size);

    nh->declare_parameter<double>("pcd_save.tile_flush_idle_s", 10.0);
    nh->get_parameter("pcd_save.tile_flush_idle_s", pcd_tile_flush_idle_s);

    nh->declare_parameter<int>("pcd_save.tile_max_buffer_pts", 500000);
    nh->get_parameter("pcd_save.tile_max_buffer_pts", pcd_tile_max_buffer_pts);

    nh->declare_parameter<double>("pcd_save.tile_voxel", 0.05);
    nh->get_parameter("pcd_save.tile_voxel", pcd_tile_voxel);
    nh->declare_parameter<double>("pcd_save.max_range", 0.0);
    nh->get_parameter("pcd_save.max_range", pcd_max_range);

    nh->declare_parameter<std::string>("pcd_save.tile_point_policy", "centroid");
    nh->get_parameter("pcd_save.tile_point_policy", pcd_tile_policy);

    nh->declare_parameter<double>("mapping.lidar_time_inte", 0.1);
    nh->get_parameter("mapping.lidar_time_inte", lidar_time_inte);

    nh->declare_parameter<float>("mapping.ivox_grid_resolution", 0.2);
    nh->get_parameter("mapping.ivox_grid_resolution", ivox_options_.resolution_);

    nh->declare_parameter<int>("ivox_nearby_type", 18);
    nh->get_parameter("ivox_nearby_type", ivox_nearby_type);


    nh->declare_parameter<double>("preprocess.det_range", 100.0);
    nh->get_parameter("preprocess.det_range", p_pre->det_range);

    nh->declare_parameter<bool>("preprocess.tag_intensity_filter", true);
    nh->get_parameter("preprocess.tag_intensity_filter", p_pre->tag_intensity_filter);

    nh->declare_parameter<double>("preprocess.min_intensity", 0.0);
    nh->get_parameter("preprocess.min_intensity", p_pre->min_intensity);

    nh->declare_parameter<int>("outlier_filter.method", 1);
    nh->get_parameter("outlier_filter.method", p_pre->outlier_method);

    nh->declare_parameter<double>("outlier_filter.voxel_res", 0.3);
    nh->get_parameter("outlier_filter.voxel_res", p_pre->outlier_voxel_res);

    nh->declare_parameter<int>("outlier_filter.min_pts_per_voxel", 2);
    nh->get_parameter("outlier_filter.min_pts_per_voxel", p_pre->outlier_min_pts);

    nh->declare_parameter<double>("outlier_filter.max_range", 10.0);
    nh->get_parameter("outlier_filter.max_range", p_pre->outlier_max_range);

    nh->declare_parameter<double>("outlier_filter.radius", 0.5);
    nh->get_parameter("outlier_filter.radius", p_pre->outlier_radius);

    nh->declare_parameter<int>("outlier_filter.min_neighbors", 3);
    nh->get_parameter("outlier_filter.min_neighbors", p_pre->outlier_min_neighbors);

    nh->declare_parameter<bool>("prior_pcd.freeze", false);
    nh->get_parameter("prior_pcd.freeze", prior_pcd_freeze);
    nh->declare_parameter<bool>("loop_closure.enable", false);
    nh->get_parameter("loop_closure.enable", loop_en);

    // ---- GNSS ----
    nh->declare_parameter<bool>("gnss.enable", false);
    nh->get_parameter("gnss.enable", gnss_en);

    nh->declare_parameter<double>("mapping.odom_max_vel", 5.0);
    nh->get_parameter("mapping.odom_max_vel", odom_max_vel);

    // 40 keeps upstream behaviour but is 7x too small: MapIncremental dedups by
    // filter_size_map, so a voxel legitimately holds (ivox_grid_resolution/
    // filter_size_map)^3 points. Once full it overwrites the oldest in a ring and
    // erases the previous pass on revisits. Set it explicitly per scene.
    nh->declare_parameter<int>("mapping.ivox_max_pts_per_node", 200);
    nh->get_parameter("mapping.ivox_max_pts_per_node", ivox_max_pts_per_node);

    // ---- Camera / visual update (FAST-LIVO2-style photometric LIVO) ----
    nh->declare_parameter<bool>("camera.enable", false);
    nh->get_parameter("camera.enable", camera_enable);

    nh->declare_parameter<std::string>("camera.image_topic", "/camera/color/image_raw");
    nh->get_parameter("camera.image_topic", camera_image_topic);

    nh->declare_parameter<int>("camera.camera_model", 0);
    nh->get_parameter("camera.camera_model", camera_model_id);

    nh->declare_parameter<int>("camera.width", 640);
    nh->get_parameter("camera.width", camera_width);
    nh->declare_parameter<int>("camera.height", 480);
    nh->get_parameter("camera.height", camera_height);

    nh->declare_parameter<double>("camera.fx", 600.0);
    nh->get_parameter("camera.fx", camera_fx);
    nh->declare_parameter<double>("camera.fy", 600.0);
    nh->get_parameter("camera.fy", camera_fy);
    nh->declare_parameter<double>("camera.cx", 320.0);
    nh->get_parameter("camera.cx", camera_cx);
    nh->declare_parameter<double>("camera.cy", 240.0);
    nh->get_parameter("camera.cy", camera_cy);
    // Full-res frames are resized by this factor before processing; the
    // intrinsics above must already be scaled accordingly (FAST-LIVO2 "scale").
    nh->declare_parameter<double>("camera.scale", 1.0);
    nh->get_parameter("camera.scale", camera_scale);

    nh->declare_parameter<std::vector<double>>(
      "camera.dist_coeffs", std::vector<double>{0.0, 0.0, 0.0, 0.0});
    nh->get_parameter("camera.dist_coeffs", camera_dist_coeffs);

    // Camera-in-body extrinsics (calibrate with FAST-Calib or Kalibr).
    nh->declare_parameter<std::vector<double>>(
      "camera.extrinsic_T", std::vector<double>{0.0, 0.0, 0.0});
    nh->get_parameter("camera.extrinsic_T", camera_extrinT);
    nh->declare_parameter<std::vector<double>>(
      "camera.extrinsic_R", std::vector<double>{1, 0, 0, 0, 1, 0, 0, 0, 1});
    nh->get_parameter("camera.extrinsic_R", camera_extrinR);

    nh->declare_parameter<double>("camera.time_diff_cam_to_imu", 0.0);
    nh->get_parameter("camera.time_diff_cam_to_imu", time_diff_cam_to_imu);
    // Added to the image stamp to get the capture time (mid-exposure).
    nh->declare_parameter<double>("camera.exposure_time_init", 0.0);
    nh->get_parameter("camera.exposure_time_init", exposure_time_init);

    // VIO options, same names/defaults as FAST-LIVO2's vio: section.
    nh->declare_parameter<bool>("camera.normal_en", true);
    nh->get_parameter("camera.normal_en", vis_normal_en);
    nh->declare_parameter<bool>("camera.inverse_composition_en", false);
    nh->get_parameter("camera.inverse_composition_en", vis_inverse_composition_en);
    nh->declare_parameter<bool>("camera.exposure_estimate_en", true);
    nh->get_parameter("camera.exposure_estimate_en", vis_exposure_estimate_en);
    nh->declare_parameter<bool>("camera.raycast_en", false);
    nh->get_parameter("camera.raycast_en", vis_raycast_en);
    nh->declare_parameter<bool>("camera.ncc_en", false);
    nh->get_parameter("camera.ncc_en", vis_ncc_en);
    nh->declare_parameter<double>("camera.ncc_thre", 0.5);
    nh->get_parameter("camera.ncc_thre", vis_ncc_thre);
    nh->declare_parameter<int>("camera.max_iterations", 5);
    nh->get_parameter("camera.max_iterations", vis_max_iterations);
    nh->declare_parameter<int>("camera.grid_size", 5);
    nh->get_parameter("camera.grid_size", vis_grid_size);
    nh->declare_parameter<int>("camera.grid_n_height", 17);
    nh->get_parameter("camera.grid_n_height", vis_grid_n_height);
    nh->declare_parameter<int>("camera.patch_pyrimid_level", 3);
    nh->get_parameter("camera.patch_pyrimid_level", vis_patch_pyrimid_level);
    nh->declare_parameter<int>("camera.patch_size", 8);
    nh->get_parameter("camera.patch_size", vis_patch_size);
    nh->declare_parameter<double>("camera.img_point_cov", 100.0);
    nh->get_parameter("camera.img_point_cov", vis_img_point_cov);
    nh->declare_parameter<double>("camera.outlier_threshold", 1000.0);
    nh->get_parameter("camera.outlier_threshold", vis_outlier_threshold);
    nh->declare_parameter<double>("camera.inv_expo_cov", 0.2);
    nh->get_parameter("camera.inv_expo_cov", exposure_cov);
    nh->declare_parameter<double>("camera.exposure_cov_init", 0.01);
    nh->get_parameter("camera.exposure_cov_init", exposure_cov_init);

    // ---- RGB map (POINT-LIVO style colored cloud projection) ----
    nh->declare_parameter<bool>("rgb_map.enable", false);
    nh->get_parameter("rgb_map.enable", rgb_map_en);
    nh->declare_parameter<int>("rgb_map.pub_scan_num", 1);
    nh->get_parameter("rgb_map.pub_scan_num", rgb_pub_scan_num);
    nh->declare_parameter<double>("rgb_map.blind_rgb_points", 0.0);
    nh->get_parameter("rgb_map.blind_rgb_points", rgb_blind_points);
    nh->declare_parameter<double>("rgb_map.filter_size_pcd", 0.15);
    nh->get_parameter("rgb_map.filter_size_pcd", rgb_filter_size_pcd);
    nh->declare_parameter<bool>("rgb_map.save_raw", true);
    nh->get_parameter("rgb_map.save_raw", rgb_save_raw);
    nh->declare_parameter<double>("rgb_map.max_range", 0.0);
    nh->get_parameter("rgb_map.max_range", rgb_max_range);
    nh->declare_parameter<int>("rgb_map.interval", -1);
    nh->get_parameter("rgb_map.interval", rgb_interval);
  } catch (const rclcpp::ParameterTypeException & e) {
    PL_ERROR(nh->get_logger(), "Parameter type exception: %s", e.what());
  } catch (const std::exception & e) {
    PL_ERROR(nh->get_logger(), "Exception: %s", e.what());
  }


  if (p_pre->outlier_min_pts < 1) p_pre->outlier_min_pts = 1;
  if (ivox_max_pts_per_node < NUM_MATCH_POINTS) ivox_max_pts_per_node = NUM_MATCH_POINTS;
  if (map_pub_interval < 0.0) map_pub_interval = 0.0;
  if (path_pub_interval < 0.0) path_pub_interval = 0.0;
  if (path_max_poses < 0) path_max_poses = 0;
  ivox_options_.max_pts_per_node_ = static_cast<std::size_t>(ivox_max_pts_per_node);

  if (ivox_nearby_type == 0) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
  } else if (ivox_nearby_type == 6) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
  } else if (ivox_nearby_type == 18) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
  } else if (ivox_nearby_type == 26) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
  } else {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
  }
  p_imu->gravity_ << VEC_FROM_ARRAY(gravity);
}

Eigen::Matrix<double, 3, 1> SO3ToEuler(const SO3 & rot)
{
  double sy = sqrt(rot(0, 0) * rot(0, 0) + rot(1, 0) * rot(1, 0));
  bool singular = sy < 1e-6;
  double x, y, z;
  if (!singular) {
    x = atan2(rot(2, 1), rot(2, 2));
    y = atan2(-rot(2, 0), sy);
    z = atan2(rot(1, 0), rot(0, 0));
  } else {
    x = atan2(-rot(1, 2), rot(1, 1));
    y = atan2(-rot(2, 0), sy);
    z = 0;
  }
  Eigen::Matrix<double, 3, 1> ang(x, y, z);
  return ang;
}

ofstream fout_out;

void open_file(const std::string & dir)
{
  fout_out.open(dir + "/mat_out.txt", ios::out);
  if (!fout_out) LOG(WARNING) << "cannot open " << dir << "/mat_out.txt; state dump disabled";
}

void reset_cov(Eigen::Matrix<double, 24, 24> & P_init)
{
  P_init = MD(24, 24)::Identity() * 0.1;
  P_init.block<3, 3>(21, 21) = MD(3, 3)::Identity() * 0.0001;
  P_init.block<6, 6>(15, 15) = MD(6, 6)::Identity() * 0.001;
}

void reset_cov_output(Eigen::Matrix<double, 31, 31> & P_init_output)
{
  P_init_output = MD(31, 31)::Identity() * 0.01;
  P_init_output.block<3, 3>(21, 21) = MD(3, 3)::Identity() * 0.0001;
  P_init_output.block<6, 6>(24, 24) = MD(6, 6)::Identity() * 0.001;
  P_init_output(30, 30) = exposure_cov_init;  // exposure state
}