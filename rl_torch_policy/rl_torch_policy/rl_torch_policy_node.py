#!/usr/bin/env python3
from __future__ import annotations

import math
import struct
from typing import Optional

import numpy as np
import torch

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
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
        self.declare_parameter("pointcloud_topic", "/points2")

        self.declare_parameter("control_rate_hz", 20.0)

        # Ezek egyezzenek a training környezettel.
        self.declare_parameter("obs_clip", 20.0)
        self.declare_parameter("lidar_num_beams", 64)
        self.declare_parameter("lidar_max_range", 3.5)
        self.declare_parameter("lidar_fov", 90.0)

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
        pointcloud_topic = self.get_parameter("pointcloud_topic").value

        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)

        self.obs_clip = float(self.get_parameter("obs_clip").value)
        self.lidar_num_beams = int(self.get_parameter("lidar_num_beams").value)
        self.lidar_max_range = float(self.get_parameter("lidar_max_range").value)
        self.lidar_fov = float(self.get_parameter("lidar_fov").value)

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
        self.latest_pc_ranges: Optional[np.ndarray] = None  # sampled ranges from PointCloud2
        self.ad_mode = False

        self.prev_delta_norm = 0.0
        self.prev_v_norm = 0.0

        # -------------------------
        # ROS interfaces
        # -------------------------
        self.goal_sub = self.create_subscription(PoseStamped, goal_topic, self.goal_callback, 10)

        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)

        self.pc_sub = self.create_subscription(PointCloud2, pointcloud_topic, self.pointcloud_callback, 10)

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

    def pointcloud_callback(self, msg: PointCloud2):
        """
        PointCloud2 -> N virtuális sugár fix FOV-on belül.
        A pontfelhő xyz mezőit 2D-re vetítjük, fix szögbinekbe soroljuk,
        és binenként a legkisebb távolságot vesszük.
        Ha egy binben nincs pont, akkor lidar_max_range marad.
        """
        try:
            points_xy = self._parse_pointcloud2_xy(msg)
        except Exception as e:
            self.get_logger().error(f"PointCloud2 parse failed: {e}", throttle_duration_sec=2.0)
            return

        sampled = np.full(self.lidar_num_beams, self.lidar_max_range, dtype=np.float32)

        if points_xy.shape[0] == 0:
            self.latest_pc_ranges = sampled
            return

        x = points_xy[:, 0]
        y = points_xy[:, 1]

        angles = np.arctan2(y, x)  # 0 = előre, + irány a body y tengely felé
        distances = np.sqrt(x * x + y * y)

        angle_min = -0.5 * self.lidar_fov
        angle_max =  0.5 * self.lidar_fov

        valid = (
            np.isfinite(x) &
            np.isfinite(y) &
            np.isfinite(distances) &
            (distances > 0.02) &
            (distances <= self.lidar_max_range) &
            (angles >= angle_min) &
            (angles <= angle_max)
        )

        angles = angles[valid]
        distances = distances[valid]

        if angles.shape[0] == 0:
            self.latest_pc_ranges = sampled
            return

        bin_edges = np.linspace(angle_min, angle_max, self.lidar_num_beams + 1)
        bin_indices = np.digitize(angles, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.lidar_num_beams - 1)

        for i in range(self.lidar_num_beams):
            in_bin = distances[bin_indices == i]
            if in_bin.size > 0:
                sampled[i] = float(np.min(in_bin))

        self.latest_pc_ranges = sampled

    @staticmethod
    def _parse_pointcloud2_xy(msg: PointCloud2) -> np.ndarray:
        """
        PointCloud2 üzenetből kinyeri az x, y koordinátákat.
        Visszatér: (N, 2) float32 numpy array.
        Csak véges (nem NaN/Inf) pontokat tart meg.
        """
        # Megkeressük az x, y, z offset-eket a fieldekből
        field_map = {}
        for field in msg.fields:
            field_map[field.name] = field

        if 'x' not in field_map or 'y' not in field_map:
            raise ValueError("PointCloud2 missing 'x' or 'y' fields")

        x_field = field_map['x']
        y_field = field_map['y']

        point_step = msg.point_step
        data = msg.data

        # Gyors numpy-alapú feldolgozás
        n_points = msg.width * msg.height
        if n_points == 0:
            return np.empty((0, 2), dtype=np.float32)

        # Byte buffer -> numpy
        raw = np.frombuffer(data, dtype=np.uint8)
        if len(raw) < n_points * point_step:
            return np.empty((0, 2), dtype=np.float32)

        # Az x és y float32 értékeket közvetlenül olvassuk
        raw_points = raw[:n_points * point_step].reshape(n_points, point_step)

        x_off = x_field.offset
        y_off = y_field.offset

        x_vals = raw_points[:, x_off:x_off+4].copy().view(np.float32).flatten()
        y_vals = raw_points[:, y_off:y_off+4].copy().view(np.float32).flatten()

        # NaN/Inf szűrés
        valid = np.isfinite(x_vals) & np.isfinite(y_vals)
        x_vals = x_vals[valid]
        y_vals = y_vals[valid]

        # Csak olyan pontok, amelyek a kamera előtt vannak (x > 0)
        forward_mask = x_vals > 0.05
        x_vals = x_vals[forward_mask]
        y_vals = y_vals[forward_mask]

        return np.column_stack([x_vals, y_vals]).astype(np.float32)

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
                self.prev_v_norm
            ],
            dtype=np.float32,
        )

        core = np.clip(core, -self.obs_clip, self.obs_clip).astype(np.float32)

        if self.latest_pc_ranges is None:
            # Ha még nincs pontfelhő, feltételezzük, hogy max range van
            lidar_norm = np.ones(
                shape=(self.lidar_num_beams,),
                dtype=np.float32,
            )
        else:
            sampled_ranges = self.latest_pc_ranges.copy()
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
        self.prev_v_norm = speed_norm

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
        self.get_logger().info(f"Distance from goal: {distance:.2f}.", throttle_duration_sec=2.0)

        if not self.ad_mode:
            return

        try:
            obs = self.make_observation()
            steering_angle, speed = self.run_model(obs)

            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = speed
            drive_msg.drive.steering_angle = steering_angle

            self.drive_pub.publish(drive_msg)
            self.get_logger().info(f"Steering angle: {steering_angle:.2f}, Speed: {speed:.2f}")

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
