#!/usr/bin/env python3
"""
OpenStreetMap Goal Publisher for ROS 2

Működés:
  - Feliratkozik a jármű GPS pozíciójára (novatel_oem7_msgs/INSPVA).
  - Lokális HTTP szervert indít.
  - A böngészőben OpenStreetMap térképet jelenít meg Leaflet.js segítségével.
  - A térképre kattintva a kiválasztott célpontot:
      geometry_msgs/PoseStamped formában publikálja a /goal_pose topicra

A PoseStamped x/y koordinátái lokális méter koordináták:
  - alapból x = East, y = North
  - az origin lehet az első GPS pozíció vagy paraméterből megadott fix pont
  - opcionálisan az ENU koordinátarendszer elforgatható az odom/map frame-hez

FIGYELEM:
A /goal_pose csak akkor fog helyesen illeszkedni az /odom koordinátáihoz, ha:
  - ugyanaz az origin,
  - ugyanaz a tengelyirány,
  - ugyanaz a frame-konvenció.

Használat:
  ros2 run <package> openstreetmap_goal_publisher

Böngésző:
  http://<robot-ip>:8088
"""

from __future__ import annotations

import json
import math
import os
import threading
import webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import NavSatFix


# ---------------------------------------------------------------------------
# Shared state: HTTP thread <-> ROS thread
# ---------------------------------------------------------------------------

_web_state = {
    "vehicle_lat": None,
    "vehicle_lon": None,

    "goal_lat": None,
    "goal_lon": None,

    # HTTP handler ide ír új kattintáskor.
    # A ROS timer ezt egyszer kiolvassa, majd None-ra állítja.
    "pending_goal": None,
}

_web_lock = threading.Lock()


# ---------------------------------------------------------------------------
# HTML template – loaded from external file
# ---------------------------------------------------------------------------

def _load_html_template() -> str:
    """Load the HTML template from the web/ subdirectory next to this module."""
    html_path = Path(__file__).resolve().parent / "web" / "osm_goal_publisher.html"
    return html_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class GoalHttpHandler(BaseHTTPRequestHandler):

    html_page = ""

    def _send_json(self, status: int, obj: dict):
        payload = json.dumps(obj).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        self.wfile.write(payload)

    def do_GET(self):

        if self.path == "/api/state":
            with _web_lock:
                data = {
                    "vehicle_lat": _web_state["vehicle_lat"],
                    "vehicle_lon": _web_state["vehicle_lon"],
                    "goal_lat": _web_state["goal_lat"],
                    "goal_lon": _web_state["goal_lon"],
                }

            self._send_json(200, data)
            return

        payload = self.html_page.encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        self.wfile.write(payload)

    def do_POST(self):

        if self.path != "/api/goal":
            self._send_json(404, {"ok": False, "error": "not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)

            data = json.loads(raw_body.decode("utf-8"))

            lat = float(data["lat"])
            lon = float(data["lon"])

            if not (-90.0 <= lat <= 90.0):
                raise ValueError("invalid latitude")

            if not (-180.0 <= lon <= 180.0):
                raise ValueError("invalid longitude")

            with _web_lock:
                _web_state["goal_lat"] = lat
                _web_state["goal_lon"] = lon
                _web_state["pending_goal"] = (lat, lon)

            self._send_json(
                200,
                {
                    "ok": True,
                    "lat": lat,
                    "lon": lon,
                }
            )

        except Exception as e:
            self._send_json(
                400,
                {
                    "ok": False,
                    "error": str(e),
                }
            )

    def log_message(self, format, *args):
        # HTTP log spam tiltása
        pass


# ---------------------------------------------------------------------------
# ROS 2 Node
# ---------------------------------------------------------------------------

class GoogleMapsGoalPublisher(Node):

    EARTH_RADIUS_M = 6378137.0

    def __init__(self):
        super().__init__("openstreetmap_goal_publisher")

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------

        self.declare_parameter("gps_topic", "fix")

        self.declare_parameter("goal_pose_topic", "/goal_pose")
        self.declare_parameter("goal_frame_id", "map")

        self.declare_parameter("http_port", 8088)
        self.declare_parameter("open_browser", False)

        # Webes térkép kezdőnézete, amíg nincs GPS fix.
        self.declare_parameter("initial_lat", 47.4788)
        self.declare_parameter("initial_lon", 19.0568)
        self.declare_parameter("initial_zoom", 18)

        # ------------------------------------------------------------------
        # GPS -> local XY konfiguráció
        # ------------------------------------------------------------------
        #
        # Ha True:
        #   az első beérkező GPS koordináta lesz x=0, y=0.
        #
        # Ha False:
        #   origin_lat/origin_lon paraméterből vesszük az origót.
        #
        self.declare_parameter("use_first_gps_as_origin", True)
        self.declare_parameter("origin_lat", 47.4979)
        self.declare_parameter("origin_lon", 19.0402)

        # Az originhez tartozó lokális koordináta.
        self.declare_parameter("origin_x", 0.0)
        self.declare_parameter("origin_y", 0.0)

        # Az odom/map x tengelyének iránya az ENU East tengelyhez képest.
        #
        # 0 fok:
        #   local x = East
        #   local y = North
        #
        # Példa:
        #   ha az odom x tengely 90 fokkal CCW van Easthez képest,
        #   akkor 90.0.
        #
        self.declare_parameter("odom_yaw_from_east_deg", 0.0)

        # ------------------------------------------------------------------
        # Read parameters
        # ------------------------------------------------------------------

        self.gps_topic = str(self.get_parameter("gps_topic").value)
        self.goal_pose_topic = str(self.get_parameter("goal_pose_topic").value)
        self.goal_frame_id = str(self.get_parameter("goal_frame_id").value)
        self.http_port = int(self.get_parameter("http_port").value)
        self.open_browser = bool(self.get_parameter("open_browser").value)
        self.initial_lat = float(self.get_parameter("initial_lat").value)
        self.initial_lon = float(self.get_parameter("initial_lon").value)
        self.initial_zoom = int(self.get_parameter("initial_zoom").value)
        self.use_first_gps_as_origin = bool(self.get_parameter("use_first_gps_as_origin").value)

        self.origin_lat: Optional[float]
        self.origin_lon: Optional[float]

        if self.use_first_gps_as_origin:
            self.origin_lat = None
            self.origin_lon = None
        else:
            self.origin_lat = float(self.get_parameter("origin_lat").value)
            self.origin_lon = float(self.get_parameter("origin_lon").value)

        self.origin_x = float(self.get_parameter("origin_x").value)

        self.origin_y = float(self.get_parameter("origin_y").value)

        yaw_deg = float(self.get_parameter("odom_yaw_from_east_deg").value)

        self.odom_yaw_from_east = math.radians(yaw_deg)

        # ------------------------------------------------------------------
        # ROS interfaces
        # ------------------------------------------------------------------

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self.gps_sub = self.create_subscription(NavSatFix, self.gps_topic, self.gps_callback, qos_profile)

        self.goal_pose_pub = self.create_publisher(PoseStamped, self.goal_pose_topic, 10)

        # A HTTP thread csak pending_goal-t állít.
        # A tényleges ROS publish innen, a ROS executor threadből történik.
        self.pending_goal_timer = self.create_timer(0.05, self.process_pending_goal)

        # ------------------------------------------------------------------
        # HTTP server
        # ------------------------------------------------------------------

        html_template = _load_html_template()

        GoalHttpHandler.html_page = (
            html_template
            .replace(
                "__INITIAL_LAT__",
                repr(self.initial_lat)
            )
            .replace(
                "__INITIAL_LON__",
                repr(self.initial_lon)
            )
            .replace(
                "__INITIAL_ZOOM__",
                repr(self.initial_zoom)
            )
        )

        self.http_server = ThreadingHTTPServer(
            ("0.0.0.0", self.http_port),
            GoalHttpHandler,
        )

        self.http_thread = threading.Thread(
            target=self.http_server.serve_forever,
            daemon=True,
        )

        self.http_thread.start()

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------

        url = f"http://192.168.8.18:{self.http_port}"

        self.get_logger().info(f"OpenStreetMap Goal Publisher started: {url}")

        #self.get_logger().info(f"GPS input: {self.gps_topic}")

        #self.get_logger().info(f"Goal PoseStamped output: {self.goal_pose_topic}")

        if self.use_first_gps_as_origin:
            self.get_logger().info("Local coordinate origin will be set from the first GPS fix.")
        else:
            self.get_logger().info(
                "Fixed local coordinate origin: "
                f"lat={self.origin_lat:.8f}, "
                f"lon={self.origin_lon:.8f}, "
                f"x={self.origin_x:.3f}, "
                f"y={self.origin_y:.3f}"
            )

        if self.open_browser:
            webbrowser.open(url)

    # ----------------------------------------------------------------------
    # GPS callback
    # ----------------------------------------------------------------------

    def gps_callback(self, msg: NavSatFix):

        lat = float(msg.latitude)
        lon = float(msg.longitude)

        with _web_lock:
            _web_state["vehicle_lat"] = lat
            _web_state["vehicle_lon"] = lon

        # Első GPS fix lesz az origin.
        if (self.use_first_gps_as_origin and self.origin_lat is None):
            self.origin_lat = lat
            self.origin_lon = lon

            self.get_logger().info(
                "Local origin initialized from first GPS fix: "
                f"lat={lat:.8f}, lon={lon:.8f} "
                f"-> x={self.origin_x:.3f}, y={self.origin_y:.3f}"
            )

    # ----------------------------------------------------------------------
    # HTTP -> ROS
    # ----------------------------------------------------------------------

    def process_pending_goal(self):

        pending: Optional[Tuple[float, float]] = None

        with _web_lock:
            if _web_state["pending_goal"] is not None:
                pending = _web_state["pending_goal"]
                _web_state["pending_goal"] = None

        if pending is None:
            return

        lat, lon = pending

        # A PoseStampedhez kell egy lokális origin.
        if self.origin_lat is None or self.origin_lon is None:
            self.get_logger().warn(
                "Goal clicked, but /goal_pose cannot be published yet "
                "because the local GPS origin is not initialized. "
                "Waiting for the first GPS fix."
            )
            return

        x, y = self.wgs84_to_local_xy(lat, lon)

        self.publish_goal_pose(
            x=x,
            y=y,
        )

        self.get_logger().info(
            "New map goal: "
            f"lat={lat:.8f}, lon={lon:.8f} "
            f"-> local x={x:.3f} m, y={y:.3f} m"
        )

    # ----------------------------------------------------------------------
    # Publishers
    # ----------------------------------------------------------------------

    def publish_goal_pose(self, x: float, y: float):

        msg = PoseStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.goal_frame_id

        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0

        # A goal orientationt az RL node jelenleg nem használja.
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        self.goal_pose_pub.publish(msg)

    # ----------------------------------------------------------------------
    # WGS84 -> local XY
    # ----------------------------------------------------------------------

    def wgs84_to_local_xy(
        self,
        lat: float,
        lon: float,
    ) -> Tuple[float, float]:
        """
        Kis területre alkalmas lokális tangent-plane közelítés.

        Első lépés:
            WGS84 -> ENU-szerű lokális koordináta
            east  [m]
            north [m]

        Második lépés:
            ENU -> odom/map frame forgatás.

        Default:
            odom_yaw_from_east_deg = 0
            x = East
            y = North
        """

        assert self.origin_lat is not None
        assert self.origin_lon is not None

        lat0_rad = math.radians(self.origin_lat)

        dlat_rad = math.radians(lat - self.origin_lat)
        dlon_rad = math.radians(lon - self.origin_lon)

        north = self.EARTH_RADIUS_M * dlat_rad
        east = (
            self.EARTH_RADIUS_M
            * math.cos(lat0_rad)
            * dlon_rad
        )

        # ENU/world -> odom frame
        a = self.odom_yaw_from_east

        c = math.cos(a)
        s = math.sin(a)

        x_rel = c * east + s * north
        y_rel = -s * east + c * north

        x = self.origin_x + x_rel
        y = self.origin_y + y_rel

        return x, y

    # ----------------------------------------------------------------------
    # Shutdown
    # ----------------------------------------------------------------------

    def destroy_node(self):

        try:
            self.http_server.shutdown()
            self.http_server.server_close()
        except Exception:
            pass

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = GoogleMapsGoalPublisher()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
