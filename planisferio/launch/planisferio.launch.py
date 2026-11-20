"""Launch Planisferio, the 2D LiDAR SLAM node (and optionally RViz).

Arguments:
    params_file:  parameter file (default: the package's config/planisferio.yaml)
    use_sim_time: 'true' in simulation, so the node follows /clock
    rviz:         'true' opens RViz with the package's display configuration
"""
# Author: Raymundo Gomez

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Return the launch description: three arguments, the node, optional RViz."""
    share = get_package_share_directory('planisferio')
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file', default_value=os.path.join(share, 'config', 'planisferio.yaml'),
            description='Planisferio parameter file'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use the /clock topic (simulation)'),
        DeclareLaunchArgument(
            'rviz', default_value='false',
            description='Open RViz with the map, scan, trajectory and pose graph'),
        Node(
            package='planisferio',
            executable='slam',
            name='planisferio',
            output='screen',
            parameters=[LaunchConfiguration('params_file'), {'use_sim_time': use_sim_time}],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='planisferio_rviz',
            output='log',
            arguments=['-d', os.path.join(share, 'rviz', 'planisferio.rviz')],
            parameters=[{'use_sim_time': use_sim_time}],
            condition=IfCondition(LaunchConfiguration('rviz')),
        ),
    ])
