"""Launch Gazebo Harmonic simulation with gz_ros2_control for WISERLI-ATAL-C1.1 + Robotiq 2F-85."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share_dir = get_package_share_directory("wiserli_atal_description")
    pkg_share = FindPackageShare("wiserli_atal_description")

    default_xacro_path = PathJoinSubstitution([pkg_share, "urdf", "wiserli_atal.urdf.xacro"])
    default_world_path = PathJoinSubstitution([pkg_share, "worlds", "empty.sdf"])
    default_rviz_path = PathJoinSubstitution([pkg_share, "rviz", "display.rviz"])
    controllers_yaml = PathJoinSubstitution([pkg_share, "config", "ros2_controllers.yaml"])
    gz_bridge_yaml = PathJoinSubstitution([pkg_share, "config", "gz_bridge.yaml"])

    declared_arguments = [
        DeclareLaunchArgument(
            "include_gripper",
            default_value="true",
            description="Attach and spawn controller for the Robotiq 2F-85 gripper.",
        ),
        DeclareLaunchArgument(
            "world",
            default_value=default_world_path,
            description="Path to Gazebo Harmonic SDF world file.",
        ),
        DeclareLaunchArgument(
            "gz_args",
            default_value="-r",
            description="Extra arguments passed to gz sim (e.g. '-r' to start unpaused, '-r -s' for headless).",
        ),
        DeclareLaunchArgument(
            "rviz",
            default_value="true",
            description="Launch RViz2 synchronized with simulation time.",
        ),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=default_rviz_path,
            description="Absolute path to the RViz2 config file.",
        ),
    ]

    # Ensure Gazebo Harmonic resolves package://wiserli_atal_description/meshes/... URIs
    set_gz_resource_path = AppendEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=os.path.dirname(pkg_share_dir),
    )

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            default_xacro_path,
            " use_gazebo:=true",
            " include_gripper:=",
            LaunchConfiguration("include_gripper"),
            " controllers_yaml:=",
            controllers_yaml,
        ]
    )

    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description, {"use_sim_time": True}],
    )

    gz_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"])
        ),
        launch_arguments={
            "gz_args": [LaunchConfiguration("gz_args"), " ", LaunchConfiguration("world")],
            "on_exit_shutdown": "true",
        }.items(),
    )

    gz_bridge_node = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        parameters=[{"config_file": gz_bridge_yaml}],
        output="screen",
    )

    spawn_entity_node = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-topic",
            "/robot_description",
            "-name",
            "wiserli_atal",
            "-allow_renaming",
            "true",
        ],
        output="screen",
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
        output="screen",
    )

    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "arm_controller",
            "--controller-manager",
            "/controller_manager",
        ],
        output="screen",
    )

    gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "gripper_controller",
            "--controller-manager",
            "/controller_manager",
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("include_gripper")),
    )

    # Wait until the robot entity is spawned in Gazebo before starting joint_state_broadcaster
    spawn_jsb_after_entity = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity_node,
            on_exit=[joint_state_broadcaster_spawner],
        )
    )

    # Start arm & gripper controllers once joint_state_broadcaster is loaded
    spawn_controllers_after_jsb = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[arm_controller_spawner, gripper_controller_spawner],
        )
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription(
        declared_arguments
        + [
            set_gz_resource_path,
            gz_sim_launch,
            gz_bridge_node,
            robot_state_publisher_node,
            spawn_entity_node,
            spawn_jsb_after_entity,
            spawn_controllers_after_jsb,
            rviz_node,
        ]
    )
