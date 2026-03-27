import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # Paths
    localization_dir = get_package_share_directory('localization')
    ekf_config = os.path.join(localization_dir, 'config', 'ekf.yaml')
    navsat_config = os.path.join(localization_dir, 'config', 'navsat.yaml')

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

    tf_base_map = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["0", "0", "0", "0", "0", "0", "map", "odom"],
    )

    tf_base_gps = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["0.16", "0", "0.7", "0", "0", "0", "base_link", "gps_link"],
    )

    tf_base_imu = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["0.33", "0", "0.8", "0.5", "-0.5", "0.5", "0.5", "base_link", "imu_link"],
    )

    tf_base_lidar = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["0.33", "0", "0.55", "0", "0", "0", "base_link", "lidar_link"],
    )

    # Robot Localization Nodes
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_config]
    )

    navsat_transform_node = Node(
        package='robot_localization',
        executable='navsat_transform_node',
        name='navsat_transform',
        output='screen',
        parameters=[navsat_config],
        remappings=[
            ('imu', '/imu'),
            ('gps/fix', '/fix'),
            ('odometry/gps', '/odometry/gps'),
            ('odom', '/odometry/filtered')
        ]
    )

    realsense_imu_node = Node(
        package='realsense',
        executable='imu',
        name='imu_node',
        output='screen'
    )

    localization_vis_node = Node(
        package='coord_trans',
        executable='localization_vis_node',
        name='localization_vis_node',
        output='screen'
    )

    return LaunchDescription([
        ublox_gps_launch,
        ntrip_client_launch,
        tf_base_map,
        tf_base_gps,
        tf_base_imu,
        tf_base_lidar,
        ekf_node,
        navsat_transform_node,
        realsense_imu_node,
        localization_vis_node
    ])
