#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped


def quaternion_to_yaw(q) -> float:
    """
    Quaternion -> yaw [rad]
    ROS orientation: x, y, z, w
    """
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class RLTorchPolicyNode(Node):
    def __init__(self):
        super().__init__("rl_torch_policy_node")

        # -------------------------
        # Parameters
        # -------------------------
        self.declare_parameter("model_path", "/home/jetson/rl_policy/exported_policy.pt")

        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("odom_topic", "/odom")

        self.declare_parameter("steering_topic", "/steering_angle")
        self.declare_parameter("speed_topic", "/speed")
        self.declare_parameter("scan_topic", "/scan")

        self.declare_parameter("control_rate_hz", 20.0)

        # Ezek egyezzenek a training környezettel.
        self.declare_parameter("obs_clip", 20.0)
        self.declare_parameter("lidar_num_beams", 32)
        self.declare_parameter("lidar_max_range", 3.5)

        # Steering visszaskálázás.
        self.declare_parameter("delta_max", 0.45)

        # Speed visszaskálázás.
        # Feltételezés: a modell speed actionje [-1, 1] tartományú.
        self.declare_parameter("min_speed", 1.0)
        self.declare_parameter("max_speed", 3.0)

        # Ha nincs CUDA vagy Jetsonon CPU-n akarod futtatni, legyen "cpu".
        self.declare_parameter("device", "cuda")

        self.model_path = self.get_parameter("model_path").value

        goal_topic = self.get_parameter("goal_topic").value
        odom_topic = self.get_parameter("odom_topic").value
        steering_topic = self.get_parameter("steering_topic").value
        speed_topic = self.get_parameter("speed_topic").value
        scan_topic = self.get_parameter("scan_topic").value

        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)

        self.obs_clip = float(self.get_parameter("obs_clip").value)
        self.lidar_num_beams = int(self.get_parameter("lidar_num_beams").value)
        self.lidar_max_range = float(self.get_parameter("lidar_max_range").value)

        self.delta_max = float(self.get_parameter("delta_max").value)
        self.min_speed = float(self.get_parameter("min_speed").value)
        self.max_speed = float(self.get_parameter("max_speed").value)

        requested_device = str(self.get_parameter("device").value)
        if requested_device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # -------------------------
        # Model load
        # -------------------------
        self.get_logger().info(f"Loading TorchScript model: {self.model_path}")
        self.model = torch.jit.load(self.model_path, map_location=self.device)
        self.model.eval()

        self.get_logger().info(f"Model loaded on device: {self.device}")

        # -------------------------
        # State
        # -------------------------
        self.latest_goal: Optional[PoseStamped] = None
        self.latest_odom: Optional[Odometry] = None
        self.latest_scan: Optional[LaserScan] = None

        self.prev_delta_norm = 0.0

        # -------------------------
        # ROS interfaces
        # -------------------------
        self.goal_sub = self.create_subscription(PoseStamped, goal_topic, self.goal_callback, 10)

        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)

        self.scan_sub = self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10)

        self.ad_enable_sub = self.create_subscription(Bool, '/autonomous_enable', self.ad_enable_cb, 10)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        timer_period = 1.0 / self.control_rate_hz
        self.timer = self.create_timer(timer_period, self.control_loop)

        self.get_logger().info("RL Torch policy node started.")
        #self.get_logger().info(f"Subscribed goal: {goal_topic}")
        #self.get_logger().info(f"Subscribed odom: {odom_topic}")
        #self.get_logger().info(f"Subscribed scan: {scan_topic}")

    def goal_callback(self, msg: PoseStamped):
        self.latest_goal = msg

    def odom_callback(self, msg: Odometry):
        self.latest_odom = msg

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    def ad_enable_cb(self, msg: Bool):
        self.ad_mode = msg.data

        if not self.ad_mode:
            self.stop_vehicle()

    def stop_vehicle(self):
        msg = AckermannDriveStamped()
        msg.drive.speed = 0.0
        self.drive_pub.publish(msg)

    def make_observation(self) -> np.ndarray:
        """
        Observation:
        [
            dx_body,
            dy_body,
            dist,
            sin(heading_err),
            cos(heading_err),
            prev_delta_norm,
            lidar_0_norm,
            ...
            lidar_31_norm
        ]
        """

        assert self.latest_goal is not None
        assert self.latest_odom is not None

        # Vehicle pose from odom
        odom_pose = self.latest_odom.pose.pose

        x = float(odom_pose.position.x)
        y = float(odom_pose.position.y)
        yaw = quaternion_to_yaw(odom_pose.orientation)

        # Goal pose
        goal_pose = self.latest_goal.pose

        goal_x = float(goal_pose.position.x)
        goal_y = float(goal_pose.position.y)

        # World frame-ben célvektor
        dx_world = goal_x - x
        dy_world = goal_y - y

        # World -> body transzformáció
        # Ez megegyezik a training environment logikájával:
        # body_x =  cos(yaw) * world_x + sin(yaw) * world_y
        # body_y = -sin(yaw) * world_x + cos(yaw) * world_y
        c = math.cos(yaw)
        s = math.sin(yaw)

        dx_body = c * dx_world + s * dy_world
        dy_body = -s * dx_world + c * dy_world

        dist = math.sqrt(dx_world * dx_world + dy_world * dy_world)

        heading_err = math.atan2(dy_body, dx_body)

        core = np.array(
            [
                dx_body,
                dy_body,
                dist,
                math.sin(heading_err),
                math.cos(heading_err),
                self.prev_delta_norm,
            ],
            dtype=np.float32,
        )

        core = np.clip(core, -self.obs_clip, self.obs_clip).astype(np.float32)

        if self.latest_scan is None:
            # Ha még nincs scan, feltételezzük, hogy max range van
            lidar_norm = np.ones(
                shape=(self.lidar_num_beams,),
                dtype=np.float32,
            )
        else:
            scan = self.latest_scan
            ranges = np.array(scan.ranges, dtype=np.float32)
            
            # Nan / Inf kezelése
            ranges = np.nan_to_num(ranges, nan=self.lidar_max_range, posinf=self.lidar_max_range, neginf=0.0)
            ranges = np.clip(ranges, 0.0, self.lidar_max_range)

            num_ranges = len(ranges)
            fov = 180.0
            right_limit = fov / 2.0
            left_limit = 360.0 - right_limit

            if num_ranges == 720:
                right_limit = right_limit * 2.0
                left_limit = 720.0 - right_limit
            elif num_ranges == 1080:
                right_limit = right_limit * 3.0
                left_limit = 1080.0 - right_limit

            right_limit_idx = int(right_limit)
            left_limit_idx = int(left_limit)

            # C++ logikának megfelelően a valid pontok:
            # index <= right_limit VAGY index >= left_limit
            # Folytonos array-t készítünk belőle (jobb oldal és bal oldal összefűzése)
            right_side = ranges[left_limit_idx:]
            left_side = ranges[:right_limit_idx + 1]
            valid_ranges = np.concatenate([right_side, left_side])

            # Downsample a kért lidar_num_beams méretre
            if len(valid_ranges) > 0:
                indices = np.linspace(0, len(valid_ranges) - 1, self.lidar_num_beams).astype(int)
                sampled_ranges = valid_ranges[indices]
            else:
                sampled_ranges = np.ones(self.lidar_num_beams, dtype=np.float32) * self.lidar_max_range

            # Normalizálás [0, 1] közé
            lidar_norm = sampled_ranges / self.lidar_max_range
            lidar_norm = np.clip(lidar_norm, 0.0, 1.0).astype(np.float32)

        obs = np.concatenate([core, lidar_norm], axis=0).astype(np.float32)

        return obs

    def run_model(self, obs: np.ndarray) -> tuple[float, float]:
        """
        Modell output:
            action[0] = steering_norm in [-1, 1]
            action[1] = speed_norm in [-1, 1]

        Visszatérés:
            steering_angle_rad
            speed_mps
        """

        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_tensor = self.model(obs_tensor)

        action = action_tensor.squeeze(0).detach().cpu().numpy()

        if action.shape[0] < 2:
            raise RuntimeError(
                f"A modell legalább 2 actiont kell adjon, de csak {action.shape[0]} van."
            )

        steering_norm = float(np.clip(action[0], -1.0, 1.0))
        speed_norm = float(np.clip(action[1], -1.0, 1.0))

        steering_angle = steering_norm * self.delta_max

        # [-1, 1] -> [min_speed, max_speed]
        speed = self.min_speed + 0.5 * (speed_norm + 1.0) * (
            self.max_speed - self.min_speed
        )

        # A következő observationhöz eltároljuk.
        self.prev_delta_norm = steering_norm

        return steering_angle, speed

    def control_loop(self):
        if self.latest_goal is None:
            self.get_logger().warn("No goal_pose received yet.", throttle_duration_sec=2.0)
            return

        if self.latest_odom is None:
            self.get_logger().warn("No odom received yet.", throttle_duration_sec=2.0)
            return
        
        dx= self.latest_odom.pose.pose.position.x - self.latest_goal.pose.position.x
        dy = self.latest_odom.pose.pose.position.y- self.latest_goal.pose.position.y
        distance = math.hypot(dx, dy)

        if distance <= 0.5:
            self.stop_vehicle()
            return
        self.get_logger().warn(f"Distance from goal: {distance:.2f}.", throttle_duration_sec=2.0)

        if not self.ad_mode:
            return

        try:
            obs = self.make_observation()
            steering_angle, speed = self.run_model(obs)

            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = speed
            drive_msg.drive.steering_angle = steering_angle

            self.drive_pub.publish(drive_msg)

        except Exception as e:
            self.get_logger().error(f"RL inference failed: {e}")


def main(args=None):
    rclpy.init(args=args)

    node = RLTorchPolicyNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()