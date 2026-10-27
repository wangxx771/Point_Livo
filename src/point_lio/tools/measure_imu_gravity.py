#!/usr/bin/env python3
"""Measure the gravity direction in the Livox IMU frame and print the matching
gravity_init line for config/mid360s.yaml.

With the sensor mounted on a tilted bracket, the level-mount default
[0, 0, -10.03] no longer points down. Static initialisation measures gravity
itself, but imu_init_dynamic=true and imu_en=false build the world frame from
gravity_init, so a wrong value tilts the whole map by the mounting angle.

Keep the sensor perfectly still while measuring: the printed value is the
average specific force, which is only gravity when there is no motion.

Usage:
  measure_imu_gravity.py                          # live /livox/imu, 10 s
  measure_imu_gravity.py --seconds 20
  measure_imu_gravity.py --bag ~/slam_ws/bags/20261020_110000/bag
"""

import argparse
import math
import os
import sys
import time

import numpy as np

G_M_S2 = 9.81


def sample_from_msg(msg):
    a = msg.linear_acceleration
    w = msg.angular_velocity
    return (a.x, a.y, a.z), (w.x, w.y, w.z)


def collect_live(topic, seconds, skip):
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Imu

    rclpy.init()
    node = Node("measure_imu_gravity")
    accs, gyrs, stamps = [], [], []

    def cb(msg):
        a, w = sample_from_msg(msg)
        accs.append(a)
        gyrs.append(w)
        stamps.append(time.monotonic())

    node.create_subscription(Imu, topic, cb, 200)
    print(f"listening on {topic} for {seconds:g} s (skip first {skip:g} s) ...")
    deadline = time.monotonic() + seconds + skip
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()

    if not accs:
        sys.exit(f"error: no messages on {topic}")
    accs, gyrs, stamps = np.array(accs), np.array(gyrs), np.array(stamps)
    keep = stamps >= stamps[0] + skip
    if keep.sum() < 10:
        sys.exit(f"error: only {keep.sum()} samples after the skip window")
    return accs[keep], gyrs[keep]


def bag_storage_id(bag):
    meta = os.path.join(bag, "metadata.yaml")
    if os.path.isfile(meta):
        import yaml
        with open(meta) as f:
            info = yaml.safe_load(f).get("rosbag2_bagfile_information", {})
        sid = info.get("storage_identifier")
        if sid:
            return sid
    return "sqlite3"


def collect_bag(bag, topic, seconds, skip):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Imu

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag, storage_id=bag_storage_id(bag)),
        rosbag2_py.ConverterOptions("", ""),
    )
    accs, gyrs, stamps = [], [], []
    t0 = None
    while reader.has_next():
        name, raw, t_ns = reader.read_next()
        if name != topic:
            continue
        if t0 is None:
            t0 = t_ns
        a, w = sample_from_msg(deserialize_message(raw, Imu))
        accs.append(a)
        gyrs.append(w)
        stamps.append((t_ns - t0) * 1e-9)

    if not accs:
        sys.exit(f"error: no {topic} messages in {bag}")
    accs, gyrs, stamps = np.array(accs), np.array(gyrs), np.array(stamps)
    keep = (stamps >= skip) & (stamps <= skip + seconds)
    if keep.sum() < 10:
        keep = stamps >= skip
    if keep.sum() < 10:
        sys.exit(f"error: only {keep.sum()} samples after the skip window")
    return accs[keep], gyrs[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/livox/imu")
    ap.add_argument("--seconds", type=float, default=10.0,
                    help="averaging window (default 10 s)")
    ap.add_argument("--skip", type=float, default=1.0,
                    help="seconds to discard at the start (default 1 s)")
    ap.add_argument("--bag", help="read the topic from a bag instead of live")
    a = ap.parse_args()

    if a.bag:
        accs, gyrs = collect_bag(a.bag, a.topic, a.seconds, a.skip)
    else:
        accs, gyrs = collect_live(a.topic, a.seconds, a.skip)

    mean_acc = accs.mean(axis=0)
    mean_gyr = gyrs.mean(axis=0)
    acc_std = accs.std(axis=0)
    gnorm = float(np.linalg.norm(mean_acc))
    if gnorm < 1e-6:
        sys.exit("error: mean acceleration is zero, no usable samples")

    # Same convention as laserMapping.cpp: gravity = -acc_hat * 9.81.
    gravity_init = -mean_acc / gnorm * G_M_S2
    tilt = math.degrees(math.acos(np.clip(-gravity_init[2] / G_M_S2, -1.0, 1.0)))
    roll = math.degrees(math.atan2(-gravity_init[1], -gravity_init[2]))
    pitch = math.degrees(math.atan2(gravity_init[0],
                                    math.hypot(gravity_init[1], gravity_init[2])))

    print(f"samples: {len(accs)}")
    print(f"mean acc  [g]      : [{mean_acc[0]: .4f} {mean_acc[1]: .4f} {mean_acc[2]: .4f}]"
          f"  |a| = {gnorm:.4f}")
    print(f"acc std   [g]      : [{acc_std[0]:.4f} {acc_std[1]:.4f} {acc_std[2]:.4f}]")
    print(f"mean gyro [rad/s]  : [{mean_gyr[0]: .4f} {mean_gyr[1]: .4f} {mean_gyr[2]: .4f}]")
    print(f"tilt from level    : {tilt:.2f} deg   (roll {roll:.2f}, pitch {pitch:.2f})")
    print()
    print("paste into config/mid360s.yaml:")
    print(f"            gravity_init: [ {gravity_init[0]:.4f}, {gravity_init[1]:.4f}, "
          f"{gravity_init[2]:.4f} ]")

    if acc_std.max() > 0.02 or np.linalg.norm(mean_gyr) > 0.05:
        print()
        print("warning: the sensor was not still during the measurement "
              "(acc std too large or gyro mean above 0.05 rad/s); re-run at rest.")


if __name__ == "__main__":
    main()
