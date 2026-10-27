#!/usr/bin/env bash
# Record a mapping bag: start the Livox MID-360s driver and record only the
# sensor topics (/livox/lidar, /livox/imu) so the bag contains no SLAM output.
#
# Usage: record_map_bag.sh [output_dir] [duration_seconds]
#   output_dir       default: <workspace>/bags/<YYYYmmdd_HHMMSS>
#   duration_seconds default: 0 (record until Ctrl+C)
#
# The resulting bag is <output_dir>/bag. Replay and map it with:
#   ros2 launch point_lio map_bag.launch.py bag:=<output_dir>/bag rviz:=true

set -euo pipefail

WS=${WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
OUT=${1:-$WS/bags/$(date +%Y%m%d_%H%M%S)}
DURATION=${2:-0}

# The ROS setup scripts read variables they do not always define, which is
# fatal under `set -u`; keep nounset off just for sourcing.
set +u
source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
source "$WS/install/setup.bash"
set -u
mkdir -p "$OUT"

DRIVER_PID=""
REC_PID=""

# Background jobs of a non-interactive shell start with SIGINT ignored, so a
# `kill -INT` here would be a no-op. Use SIGTERM instead: rclcpp handles it
# cleanly, so rosbag2 still flushes and writes metadata.yaml on shutdown.
stop_pid() {
  local pid=$1 sig=$2 i
  [ -z "$pid" ] && return 0
  kill -"$sig" "$pid" 2>/dev/null || return 0
  for i in $(seq 1 50); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.1
  done
  return 1
}

# `ros2 launch` does not take its child node down with it on SIGTERM, so the
# driver is started with setsid and stopped as a process group.
stop_group() {
  local pgid=$1 i
  [ -z "$pgid" ] && return 0
  kill -TERM -- "-$pgid" 2>/dev/null || return 0
  for i in $(seq 1 50); do
    kill -0 -- "-$pgid" 2>/dev/null || return 0
    sleep 0.1
  done
  kill -KILL -- "-$pgid" 2>/dev/null || true
}

cleanup() {
  trap - INT TERM EXIT
  if [ -n "$REC_PID" ]; then
    stop_pid "$REC_PID" TERM || stop_pid "$REC_PID" KILL
    wait "$REC_PID" 2>/dev/null || true
  fi
  if [ -n "$DRIVER_PID" ]; then
    stop_group "$DRIVER_PID"
    wait "$DRIVER_PID" 2>/dev/null || true
  fi
  if [ -f "$OUT/bag/metadata.yaml" ]; then
    echo "[record] bag: $OUT/bag"
  else
    echo "[record] no bag written; see $OUT/record.log and $OUT/driver.log"
  fi
}
trap cleanup INT TERM EXIT

if ! ip -4 addr show 2>/dev/null | grep -q '192\.168\.1\.5'; then
  echo "[record] warning: host IP 192.168.1.5 (MID360s_config.json) not found;" \
       "the driver may not receive LiDAR data"
fi

# Do not start a second driver when one is already publishing.
if ros2 topic list 2>/dev/null | grep -qx "/livox/lidar"; then
  echo "[record] Livox driver already running"
else
  echo "[record] starting Livox driver (msg_MID360s_launch.py)"
  setsid ros2 launch livox_ros_driver2 msg_MID360s_launch.py > "$OUT/driver.log" 2>&1 &
  DRIVER_PID=$!
fi

# Wait until the driver actually produces LiDAR and IMU messages.
if ! timeout 30 bash -c 'until ros2 topic list 2>/dev/null | grep -qx "/livox/lidar"; do sleep 1; done'; then
  echo "[record] error: /livox/lidar never appeared; see $OUT/driver.log" >&2
  exit 1
fi
if ! timeout 10 ros2 topic echo /livox/lidar --once > /dev/null; then
  echo "[record] error: no data on /livox/lidar; check the LiDAR connection and $OUT/driver.log" >&2
  exit 1
fi
if ! timeout 10 ros2 topic echo /livox/imu --once > /dev/null; then
  echo "[record] error: no data on /livox/imu; check the LiDAR connection and $OUT/driver.log" >&2
  exit 1
fi

echo "[record] recording /livox/lidar /livox/imu -> $OUT/bag"
echo "[record] press Ctrl+C to stop"
if [ "$DURATION" -gt 0 ]; then
  timeout --signal=TERM --kill-after=10 "$DURATION" \
    ros2 bag record -o "$OUT/bag" /livox/lidar /livox/imu > "$OUT/record.log" 2>&1 &
else
  ros2 bag record -o "$OUT/bag" /livox/lidar /livox/imu > "$OUT/record.log" 2>&1 &
fi
REC_PID=$!

wait "$REC_PID" || true
REC_PID=""
echo "[record] recording stopped"
