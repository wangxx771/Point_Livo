#!/bin/bash
# Prepare a FAST-LIVO2 / POINT-LIVO ROS1 dataset bag for this workspace's
# LIVO replay: repair/salvage if the bag lacks an index (truncated download),
# convert to ROS2 with the camera images kept, and print the run commands.
#
# Usage: livo_dataset.sh <in.bag> [workdir]
#   workdir defaults to /tmp/livo_data. Produces <workdir>/<name>_ros2/.
set -e

BAG=$1
[ -f "$BAG" ] || { echo "bag not found: $BAG"; exit 1; }
WORK=${2:-/tmp/livo_data}
WS=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
NAME=$(basename "$BAG" .bag)

source /opt/ros/${ROS_DISTRO:-humble}/setup.bash
source "$WS/install/setup.bash"
mkdir -p "$WORK"

# 1. Is the bag readable as-is? rosbags needs a valid index at index_pos;
#    a cut-off download has none and must be salvaged first.
READABLE=$(/usr/bin/python3 - "$BAG" << 'EOF'
import sys
try:
    from rosbags.rosbag1 import Reader
    with Reader(sys.argv[1]) as r:
        print(f"ok {r.message_count} msgs {(r.end_time-r.start_time)/1e9:.1f}s")
except Exception as e:
    print(f"bad {e}")
EOF
)
echo "$READABLE"

SRC_BAG=$BAG
if [[ "$READABLE" == bad* ]]; then
  echo "[livo_dataset] index damaged/truncated -> salvaging complete chunks"
  SRC_BAG=$WORK/${NAME}_salvaged.bag
  /usr/bin/python3 "$WS/src/point_lio/tools/salvage_rosbag1.py" "$BAG" "$SRC_BAG"
fi

# 2. ROS1 -> ROS2 with images (CustomMsg/Imu/Image).
OUT=$WORK/${NAME}_ros2
/usr/bin/python3 "$WS/src/point_lio/tools/rosbag1_to_ros2.py" "$SRC_BAG" "$OUT" --keep-image

echo
echo "================ ready: $OUT ================"
cat << EOF

# A. Deterministic replay (no DDS, reproducible, writes odom.csv):
bash $WS/src/point_lio/tools/run_replay.sh $OUT ${NAME}_livo \\
  $WS/src/point_lio/config/fastlivo2_avia.yaml

# B. Live-style playback + RViz (colored cloud on /cloud_registered_rgb,
#    auto-saves map.pcd + colored_map.pcd when playback ends):
source /opt/ros/humble/setup.bash && source $WS/install/setup.bash
ros2 launch point_lio map_bag.launch.py \\
  bag:=$OUT \\
  cfg:=$WS/src/point_lio/config/fastlivo2_avia.yaml \\
  "topics:=/livox/lidar /livox/imu /left_camera/image" \\
  rate:=1.0 map_dir:=$WS/replay_maps
EOF
