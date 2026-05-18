import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('rl_torch_policy'),
        'config',
        'rl_torch_policy.yaml'
    )

    localization_launch_file = os.path.join(
        get_package_share_directory('localization'),
        'launch',
        'localization.launch.py'
    )

    sllidar_launch_file = os.path.join(
        get_package_share_directory('sllidar_ros2'),
        'launch',
        'sllidar_a2m8_launch.py'
    )

    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(localization_launch_file)
    )

    sllidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sllidar_launch_file)
    )

    return LaunchDescription([
        localization_launch,
        sllidar_launch,
        Node(
            package='rl_torch_policy',
            executable='rl_torch_policy_node',
            name='rl_torch_policy_node',
            output='screen',
            emulate_tty=True,
            parameters=[config]
        )
    ])
