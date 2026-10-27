"""Build a map from a recorded bag, with the live map and trajectory in RViz2.

    ros2 launch point_lio map_bag.launch.py bag:=/path/to/bag
    ros2 launch point_lio map_bag.launch.py bag:=/path/to/bag rviz:=false
    ros2 launch point_lio map_bag.launch.py bag:=/path/to/bag enable_freedom:=true
ros2 launch point_lio map_bag.launch.py \
  bag:=/home/jetson/slam_ws/bags/20261020_111149/bag \
  rate:=0.5 map_dir:=/home/jetson/slam_ws/maps \
  rviz:=true

The bag must contain the sensor topics only (see tools/record_map_bag.sh):
/livox/lidar and /livox/imu. Playback is filtered with --topics, because the
Humble ros2 bag play has no --exclude (unlike Jazzy).

When playback finishes, the accumulated map is saved immediately through the
SaveMaps service, and RViz stays open so the result can be inspected. Ctrl+C
also saves on shutdown.
"""
import os
from datetime import datetime

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction,
    RegisterEventHandler, TimerAction)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup_log_dir(base: str) -> str:
    """Put ROS logs under base/<date>, falling back to ~/.ros/log."""
    day = os.path.join(base, datetime.now().strftime('%Y-%m-%d'))
    try:
        os.makedirs(day, exist_ok=True)
        probe = os.path.join(day, '.write_test')
        with open(probe, 'w'):
            pass
        os.remove(probe)
        return day
    except OSError as e:
        fallback = os.path.expanduser('~/.ros/log')
        print(f'[launch] log dir {day} not writable ({e}), falling back to {fallback}')
        return fallback


MAP_NAME = 'map.pcd'

OUTDOOR = {
    'sensor.max_range': 60.0,
    'sensor.min_z': -10.0,
    'sensor.max_z': 25.0,
    'map.raycast_max_range': 30.0,
    'map.raycast_min_z': -5.0,
    'map.raycast_max_z': 10.0,
    'map.local_map_range': 80.0,
    'map.local_map_max_z': 30.0,
}


def _save_cmd(enable_freedom: bool):
    """Save without stopping the node, so RViz keeps the finished map.

    The colored-map call is unconditional: on runs without rgb_map.enable the
    service just replies "rgb_map.enable is false" and nothing is written."""
    service = '/pgo/save_maps' if enable_freedom else '/laserMapping/save_maps'
    return ExecuteProcess(
        cmd=['ros2', 'service', 'call', service,
             'interface/srv/SaveMaps', '{file_path: "", save_patches: false}'],
        output='screen',
    )


def _save_colored_cmd():
    """Dump the RGB map (POINT-LIVO-style colored cloud) beside map.pcd."""
    return ExecuteProcess(
        cmd=['ros2', 'service', 'call', '/laserMapping/save_colored_pcd',
             'interface/srv/SaveColoredPcd', '{save_path: ""}'],
        output='screen',
    )


def launch_setup(context, *args, **kwargs):
    log_dir = _setup_log_dir(LaunchConfiguration('log_dir').perform(context))
    os.environ['ROS_LOG_DIR'] = log_dir
    print(f'[launch] log dir: {log_dir}')

    bag = LaunchConfiguration('bag').perform(context).strip()
    if not bag:
        raise RuntimeError('bag:=<path> is required (a directory containing metadata.yaml)')
    if not os.path.exists(bag):
        raise RuntimeError(f'bag path does not exist: {bag}')

    rate = LaunchConfiguration('rate').perform(context)
    start_delay = float(LaunchConfiguration('start_delay').perform(context))
    topics = LaunchConfiguration('topics').perform(context).split()

    point_lio_dir = get_package_share_directory('point_lio')
    freedom_dir = get_package_share_directory('freedom')
    cfg_arg = LaunchConfiguration('cfg').perform(context).strip()
    cfg_file = cfg_arg or os.path.join(point_lio_dir, 'config', 'mid360s.yaml')
    freedom_cfg = os.path.join(freedom_dir, 'config', 'mid360_pointlio.yaml')
    rviz_cfg_arg = LaunchConfiguration('rviz_cfg').perform(context).strip()
    rviz_cfg = rviz_cfg_arg or os.path.join(point_lio_dir, 'rviz_cfg', 'mapping.rviz')
    remappings = [('/tf', 'tf'), ('/tf_static', 'tf_static')]

    scene = LaunchConfiguration('scene').perform(context).lower()
    if scene not in ('indoor', 'outdoor'):
        raise RuntimeError(f"scene must be indoor or outdoor, got '{scene}'")

    map_dir = LaunchConfiguration('map_dir').perform(context)
    run_dir = os.path.join(map_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run_dir, exist_ok=True)
    sc_path = os.path.join(run_dir, MAP_NAME + '.sc')

    latest = os.path.join(map_dir, 'latest')
    try:
        if os.path.islink(latest) or os.path.exists(latest):
            os.remove(latest)
        os.symlink(run_dir, latest)
    except OSError as e:
        print(f'[map_bag.launch] failed to update the latest symlink: {e}')
    print(f'[map_bag.launch] map output dir: {run_dir}')

    enable_freedom = LaunchConfiguration('enable_freedom').perform(context).lower() == 'true'
    freedom_params = [freedom_cfg, {'use_sim_time': True,
                                    'save_map_path': run_dir,
                                    'save_map_name': MAP_NAME}]
    if scene == 'outdoor':
        freedom_params.append(OUTDOOR)

    play_cmd = ['ros2', 'bag', 'play', bag, '--clock', '--rate', rate]
    if topics:
        play_cmd += ['--topics', *topics]
    bag_play = ExecuteProcess(cmd=play_cmd, output='screen')

    nodes = [
        Node(
            package='point_lio',
            executable='pointlio_mapping',
            parameters=[cfg_file, {
                'use_sim_time': True,
                'pcd_save.output_dir': run_dir,
                'pcd_save.run_subdir': False,
                'pcd_save.export_patches': LaunchConfiguration('export_patches'),
                'pcd_save.pcd_save_en': not enable_freedom,
                'pcd_save.file_name': MAP_NAME,
                'scan_context.save_path': sc_path,
            }],
            remappings=remappings,
            output='screen',
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_map_to_odom',
            arguments=[
                '--x', '0', '--y', '0', '--z', '0',
                '--roll', '0', '--pitch', '0', '--yaw', '0',
                '--frame-id', 'map', '--child-frame-id', 'odom',
            ],
            parameters=[{'use_sim_time': True}],
            remappings=remappings,
            output='screen',
        ),
    ]

    if enable_freedom:
        nodes.append(
            Node(
                package='freedom',
                executable='freedom_node',
                name='freedom',
                parameters=freedom_params,
                remappings=remappings,
                output='screen',
            )
        )
        print('[map_bag.launch] FreeDOM enabled')
    else:
        print('[map_bag.launch] FreeDOM disabled (Point-LIO only)')

    nodes.append(
        Node(
            condition=IfCondition(LaunchConfiguration('rviz')),
            package='rviz2',
            executable='rviz2',
            name='rviz',
            arguments=['-d', rviz_cfg],
            parameters=[{'use_sim_time': True}],
            remappings=remappings,
            output='screen',
        )
    )

    nodes.append(
        TimerAction(period=start_delay, actions=[bag_play])
    )

    # Let the last DDS messages reach Point-LIO before snapshotting the map.
    nodes.append(
        RegisterEventHandler(
            OnProcessExit(
                target_action=bag_play,
                on_exit=[
                    TimerAction(period=2.0, actions=[
                        LogInfo(msg=f'[map_bag.launch] playback finished; map output: {run_dir}'),
                        _save_cmd(enable_freedom),
                        _save_colored_cmd(),
                    ]),
                ],
            )
        )
    )

    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'bag', default_value='',
            description='Bag directory recorded by tools/record_map_bag.sh (required)'),
        DeclareLaunchArgument(
            'rate', default_value='1.0',
            description='Playback speed'),
        DeclareLaunchArgument(
            'topics', default_value='/livox/lidar /livox/imu',
            description='Sensor topics to play (space separated); empty plays everything'),
        DeclareLaunchArgument(
            'rviz', default_value='true',
            description='Show the live map and trajectory in RViz2'),
        DeclareLaunchArgument(
            'rviz_cfg', default_value='',
            description='RViz2 config; empty uses point_lio/rviz_cfg/mapping.rviz'),
        DeclareLaunchArgument(
            'cfg', default_value='',
            description='Point-LIO config; empty uses config/mid360s.yaml (matches /livox/imu)'),
        DeclareLaunchArgument(
            'enable_freedom', default_value='false',
            description='true: run FreeDOM and save its static map | false: Point-LIO only'),
        DeclareLaunchArgument(
            'scene', default_value='indoor',
            description='indoor | outdoor - affects only FreeDOM range and height limits'),
        DeclareLaunchArgument(
            'export_patches', default_value='false',
            description='Also write poses.txt and patches/ for open3d_loc localization'),
        DeclareLaunchArgument(
            'start_delay', default_value='3.0',
            description='Seconds to wait after starting the nodes before playing the bag'),
        DeclareLaunchArgument(
            'map_dir', default_value='/var/opt/map',
            description='map output root; each launch creates a timestamped subdirectory under it'),
        DeclareLaunchArgument(
            'log_dir', default_value='/var/log/slam',
            description='Log root; a <date> folder is created under it'),
        OpaqueFunction(function=launch_setup),
    ])
