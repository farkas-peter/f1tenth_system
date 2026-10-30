# MIT License

# Copyright (c) 2020 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import ExecuteProcess
from launch.substitutions import PathJoinSubstitution
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    joy_teleop_config = os.path.join(
        get_package_share_directory('f1tenth_stack'),
        'config',
        'joy_teleop.yaml'
    )
    vesc_config = os.path.join(
        get_package_share_directory('f1tenth_stack'),
        'config',
        'vesc.yaml'
    )
    sensors_config = os.path.join(
        get_package_share_directory('f1tenth_stack'),
        'config',
        'sensors.yaml'
    )
    mux_config = os.path.join(
        get_package_share_directory('f1tenth_stack'),
        'config',
        'mux.yaml'
    )

    joy_la = DeclareLaunchArgument(
        'joy_config',
        default_value=joy_teleop_config,
        description='Descriptions for joy and joy_teleop configs')
    vesc_la = DeclareLaunchArgument(
        'vesc_config',
        default_value=vesc_config,
        description='Descriptions for vesc configs')
    sensors_la = DeclareLaunchArgument(
        'sensors_config',
        default_value=sensors_config,
        description='Descriptions for sensor configs')
    mux_la = DeclareLaunchArgument(
        'mux_config',
        default_value=mux_config,
        description='Descriptions for ackermann mux configs')

    publish_img_la = DeclareLaunchArgument(
        'publish_img',
        default_value='false',
        description='Publish the YOLO detection image on /search/detection/image')

    # the server venv lives in the source tree, not in install/
    gemini_pkg_dir_la = DeclareLaunchArgument(
        'gemini_pkg_dir',
        default_value='/workspace/src/f1tenth_system/gemini_pkg',
        description='Source folder of gemini_pkg (holds .venv and gemini_server.py)')

    ld = LaunchDescription([joy_la, vesc_la, sensors_la, mux_la, publish_img_la,
                            gemini_pkg_dir_la])

    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy',
        parameters=[LaunchConfiguration('joy_config')]
    )
    joy_teleop_node = Node(
        package='joy_teleop',
        executable='joy_teleop',
        name='joy_teleop',
        parameters=[LaunchConfiguration('joy_config')]
    )
    ackermann_to_vesc_node = Node(
        package='vesc_ackermann',
        executable='ackermann_to_vesc_node',
        name='ackermann_to_vesc_node',
        parameters=[LaunchConfiguration('vesc_config')]
    )
    vesc_to_odom_node = Node(
        package='vesc_ackermann',
        executable='vesc_to_odom_node',
        name='vesc_to_odom_node',
        parameters=[LaunchConfiguration('vesc_config')]
    )
    vesc_driver_node = Node(
        package='vesc_driver',
        executable='vesc_driver_node',
        name='vesc_driver_node',
        parameters=[LaunchConfiguration('vesc_config')]
    )
    throttle_interpolator_node = Node(
        package='f1tenth_stack',
        executable='throttle_interpolator',
        name='throttle_interpolator',
        parameters=[LaunchConfiguration('vesc_config')]
    )
    urg_node = Node(
        package='urg_node',
        executable='urg_node_driver',
        name='urg_node',
        parameters=[LaunchConfiguration('sensors_config')]
    )
    ackermann_mux_node = Node(
        package='ackermann_mux',
        executable='ackermann_mux',
        name='ackermann_mux',
        parameters=[LaunchConfiguration('mux_config')],
        remappings=[('ackermann_cmd_out', 'ackermann_drive')]
    )
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_baselink_to_laser',
        arguments=['0.27', '0.0', '0.11', '0.0', '0.0', '0.0', 'base_link', 'laser']
    )
    drive_publisher_node = Node(
        package='cone_pkg',
        executable='drive_publisher',
        name='drive_publisher_node'
    )
    # drives towards /target_point while approaching; speech_to_search_node forwards
    # /approach/drive to /drive only while RB is held, so it is the only node on /drive
    pure_pursuit_control_node = Node(
        package='cone_pkg',
        executable='pure_pursuit_control',
        name='pure_pursuit_control_node',
        remappings=[('/drive', '/approach/drive')]
    )
    intel_yolo_node = Node(
        package='cone_pkg',
        executable='intel_yolo',
        name='intel_yolo_node',
        output = 'screen'
    )
    ekf_node = Node(
        package='cone_pkg',
        executable='ekf',
        name='ekf_node'
    )
    FSS_node = Node(
        package='cone_pkg',
        executable='FSS',
        name='FSS_node'
    )
    speech_to_search_node = Node(
        package='gemini_pkg',
        executable='speech_to_search_node',
        name='speech_to_search_node',
        output='screen',
        parameters=[{'publish_img': LaunchConfiguration('publish_img')}]
    )
    # Gemini FastAPI server on 127.0.0.1:8000 (Python 3.12 venv, not ROS)
    gemini_server = ExecuteProcess(
        cmd=[PathJoinSubstitution([LaunchConfiguration('gemini_pkg_dir'), '.venv', 'bin', 'python']),
             'gemini_server.py'],
        cwd=PathJoinSubstitution([LaunchConfiguration('gemini_pkg_dir'), 'gemini_pkg']),
        name='gemini_server',
        output='screen'
    )

    # finalize
    ld.add_action(joy_node)
    ld.add_action(joy_teleop_node)
    ld.add_action(ackermann_to_vesc_node)
    ld.add_action(vesc_to_odom_node)
    ld.add_action(vesc_driver_node)
    ld.add_action(ackermann_mux_node)
    ld.add_action(pure_pursuit_control_node)
    ld.add_action(gemini_server)
    ld.add_action(speech_to_search_node)

    return ld
