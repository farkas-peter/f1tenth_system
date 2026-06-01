import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    ublox_gps_dir = get_package_share_directory('ublox_gps')
    ublox_gps_launch_file = os.path.join(ublox_gps_dir, 'launch', 'ublox_gps_node-launch.py')

    ntrip_client_dir = get_package_share_directory('ntrip_client')
    ntrip_client_launch_file = os.path.join(ntrip_client_dir, 'ntrip_client_launch.py')

    # GPS and NTRIP IncludeLaunchDescriptions
    ublox_gps_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(ublox_gps_launch_file)
    )

    ntrip_client_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(ntrip_client_launch_file)
    )

    # map -> odom TF is published dynamically by coord_trans_node

    tf_base_imu = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["0.33", "0", "0.08", "0.5", "-0.5", "0.5", "0.5", "base_link", "imu_link"],
    )

    tf_baselink_camera = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["0.33", "0", "0.08", "0", "0", "0", "base_link", "camera_link"],
    )

    tf_base_lidar = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["0.33", "0", "0.05", "-3.14159", "0", "0", "base_link", "lidar_link"],
    )

    coord_trans_node = Node(
        package='coord_trans',
        executable='coord_trans_node',
        name='coord_trans_node'
    )

    localization_vis_node = Node(
        package='coord_trans',
        executable='localization_vis_node',
        name='localization_vis_node',
        output='screen'
    )

    pcd_node = Node(
        package='realsense',
        executable='pcd',
        name='pcd',
        output='screen'
    )

    return LaunchDescription([
        ublox_gps_launch,
        ntrip_client_launch,
        tf_base_imu,
        tf_baselink_camera,
        tf_base_lidar,
        coord_trans_node,
        localization_vis_node,
        pcd_node
    ])
