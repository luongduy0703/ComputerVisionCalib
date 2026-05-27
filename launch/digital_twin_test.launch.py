#!/usr/bin/env python3
"""
Digital Twin Test Launch File
=============================
Spawns Gazebo + robot + controllers + ONE mirror node based on `mode`:

  mode:=real_to_sim  → Pi controls Gazebo (physical arm moves, Gazebo follows)
  mode:=sim_to_real  → Gazebo controls Pi (send trajectory in Gazebo, arm follows)

⚠️ Do NOT run both mirrors simultaneously — it creates a feedback loop!

Usage:
  ros2 launch visual_servoing digital_twin_test.launch.py mode:=real_to_sim
  ros2 launch visual_servoing digital_twin_test.launch.py mode:=sim_to_real
"""

import os
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription, TimerAction,
    SetEnvironmentVariable, DeclareLaunchArgument
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command, FindExecutable, PathJoinSubstitution,
    LaunchConfiguration, PythonExpression
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from launch.conditions import IfCondition


def generate_launch_description():
    pkg_share = FindPackageShare('visual_servoing').find('visual_servoing')

    # Set resource path for Gazebo
    models_path = os.path.join(pkg_share, 'models')
    share_parent = os.path.dirname(pkg_share)
    gz_resource_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    new_gz_resource_path = (
        f"{gz_resource_path}:{models_path}:{share_parent}"
        if gz_resource_path else f"{models_path}:{share_parent}"
    )

    set_gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=new_gz_resource_path
    )

    # Force Gazebo's internal gz-transport to localhost only.
    # Prevents multicast flooding that crashes the Pi's Wi-Fi hotspot.
    set_gz_ip = SetEnvironmentVariable(
        name='GZ_IP',
        value='127.0.0.1'
    )

    # ── Launch Arguments ──
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='real_to_sim',
        description='Mirror mode: real_to_sim (Pi→Gazebo) or sim_to_real (Gazebo→Pi)'
    )
    mode = LaunchConfiguration('mode')

    # URDF via xacro
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]),
        " ",
        PathJoinSubstitution([
            FindPackageShare("visual_servoing"), "urdf", "new_arm", "new_arm.xacro"
        ]),
    ])
    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='log',
        parameters=[robot_description, {'use_sim_time': True}]
    )

    # Gazebo with empty world
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py'
            ])
        ]),
        launch_arguments={'gz_args': 'empty.sdf -r'}.items()
    )

    # Spawn robot
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'robot_arm_twin',
            '-allow_renaming', 'true'
        ],
        output='screen'
    )

    # Joint State Broadcaster
    jsb_spawner = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['joint_state_broadcaster'],
                output='screen'
            )
        ]
    )

    # Arm Controller
    arm_controller_spawner = TimerAction(
        period=12.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['arm_controller'],
                output='screen'
            )
        ]
    )

    # ── Mirror Nodes (only ONE at a time!) ──

    # Real-to-Sim: enabled when mode == 'real_to_sim'
    real_to_sim = TimerAction(
        period=14.0,
        actions=[
            Node(
                package='visual_servoing',
                executable='gazebo_state_mirror',
                name='gazebo_state_mirror',
                output='screen',
                condition=IfCondition(
                    PythonExpression(["'", mode, "' == 'real_to_sim'"])
                )
            )
        ]
    )

    # Sim-to-Real: enabled when mode == 'sim_to_real'
    sim_to_real = TimerAction(
        period=14.0,
        actions=[
            Node(
                package='visual_servoing',
                executable='gazebo_to_real_mirror',
                name='gazebo_to_real_mirror',
                output='screen',
                condition=IfCondition(
                    PythonExpression(["'", mode, "' == 'sim_to_real'"])
                )
            )
        ]
    )

    return LaunchDescription([
        mode_arg,
        set_gz_resource_path,
        set_gz_ip,
        robot_state_publisher,
        gazebo,
        spawn_entity,
        jsb_spawner,
        arm_controller_spawner,
        real_to_sim,
        sim_to_real,
    ])
