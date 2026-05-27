#!/usr/bin/env python3
"""
Visual Servoing Test Launch File
Launches flipped robot with camera and ArUco detection for testing.
Optionally enables Digital Twin mirroring with digital_twin:=true
"""

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, SetEnvironmentVariable, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from launch.conditions import IfCondition


def generate_launch_description():
    # Get package share directory
    pkg_share = FindPackageShare('visual_servoing').find('visual_servoing')
    
    # Set Gazebo resource path
    models_path = os.path.join(pkg_share, 'models')
    worlds_path = os.path.join(pkg_share, 'worlds')
    share_parent = os.path.dirname(pkg_share)
    
    gz_resource_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    if gz_resource_path:
        new_gz_resource_path = f"{gz_resource_path}:{models_path}:{share_parent}"
    else:
        new_gz_resource_path = f"{models_path}:{share_parent}"
    
    set_gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=new_gz_resource_path
    )

    # Force Gazebo's internal gz-transport to localhost only.
    # Without this, gz-transport blasts multicast on ALL interfaces,
    # which floods and crashes the Pi's Wi-Fi hotspot.
    set_gz_ip = SetEnvironmentVariable(
        name='GZ_IP',
        value='127.0.0.1'
    )

    # ── Launch Arguments ──
    digital_twin_mode_arg = DeclareLaunchArgument(
        'digital_twin_mode',
        default_value='none',
        description='Enable Digital Twin mirroring: none, real_to_sim, or sim_to_real'
    )
    digital_twin_mode = LaunchConfiguration('digital_twin_mode')

    
    # Get URDF via xacro - using flipped robot
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [
                    FindPackageShare("visual_servoing"),
                    "urdf",
                    "new_arm",
                    "new_arm.xacro",
                ]
            ),
        ]
    )
    # Wrap in ParameterValue to fix yaml parsing error
    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}

    # Robot State Publisher
    # Output='log' to keep terminal clean (TF_OLD_DATA C++ warnings are harmless)
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='log',
        parameters=[robot_description, {'use_sim_time': True}]
    )

    # Gazebo Fortress with visual servoing world
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            ])
        ]),
        launch_arguments={
            'gz_args': [
                PathJoinSubstitution([
                    FindPackageShare('visual_servoing'),
                    'worlds',
                    'visual_servoing_training.world'
                ]),
                ' -r'
            ]
        }.items()
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'robot_arm_flipped',
            '-allow_renaming', 'true'
        ],
        output='screen'
    )

    # Joint State Broadcaster Spawner
    # NOTE: Delay must be long enough for gz_ros2_control plugin to fully
    # initialize the controller_manager. Too short = broadcaster stays inactive.
    joint_state_broadcaster_spawner = TimerAction(
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

    # Arm Controller Spawner
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

    # Vision ArUco Detector Node
    # Output='log' to keep launch terminal clean (camera matrix log is spammy)
    vision_detector = TimerAction(
        period=8.0,  # Wait for Gazebo camera to be ready
        actions=[
            Node(
                package='visual_servoing',
                executable='vision_aruco_detector',
                name='vision_aruco_detector',
                output='log',
                parameters=[{
                    'image_topic': '/camera/image_raw',
                    'show_gui': False,
                    'use_sim_time': True
                }]
            )
        ]
    )

    # Gazebo-ROS bridge for camera
    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo'
        ],
        output='screen'
    )

    # Gazebo Drawing Visualizer (spawns shapes + pen lines in Gazebo)
    # Output='log' to suppress C++ TF_OLD_DATA warnings (harmless sim-time mismatch)
    gazebo_drawing_visualizer = TimerAction(
        period=10.0,  # Wait for Gazebo + vision to be ready
        actions=[
            Node(
                package='visual_servoing',
                executable='gazebo_drawing_visualizer',
                name='gazebo_drawing_visualizer',
                output='log',
                parameters=[{'use_sim_time': True}]
            )
        ]
    )

    # ── Digital Twin Mirror Nodes (conditional) ──
    # Real-to-Sim: Pi's /pca9685_servo/joint_states → Gazebo
    real_to_sim_mirror = TimerAction(
        period=14.0,  # Wait for controllers to fully load
        actions=[
            Node(
                package='visual_servoing',
                executable='gazebo_state_mirror',
                name='gazebo_state_mirror',
                output='screen',
                condition=IfCondition(
                    PythonExpression(["'", digital_twin_mode, "' == 'real_to_sim'"])
                )
            )
        ]
    )

    # Sim-to-Real: Gazebo's /joint_states → Pi's /pca9685_servo/command
    sim_to_real_mirror = TimerAction(
        period=14.0,
        actions=[
            Node(
                package='visual_servoing',
                executable='gazebo_to_real_mirror',
                name='gazebo_to_real_mirror',
                output='screen',
                condition=IfCondition(
                    PythonExpression(["'", digital_twin_mode, "' == 'sim_to_real'"])
                )
            )
        ]
    )

    return LaunchDescription([
        digital_twin_mode_arg,
        set_gz_resource_path,
        set_gz_ip,
        robot_state_publisher_node,
        gazebo,
        spawn_entity,
        gz_bridge,
        joint_state_broadcaster_spawner,
        arm_controller_spawner,
        vision_detector,
        gazebo_drawing_visualizer,
        real_to_sim_mirror,
        sim_to_real_mirror,
    ])

