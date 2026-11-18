"""Launch ros2_control (mock hardware) for WISERLI-ATAL-C1.1 + Robotiq 2F-85 without Gazebo."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
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
    pkg_share = FindPackageShare("wiserli_atal_description")

    default_xacro_path = PathJoinSubstitution([pkg_share, "urdf", "wiserli_atal.urdf.xacro"])
    default_rviz_path = PathJoinSubstitution([pkg_share, "rviz", "display.rviz"])
    controllers_yaml = PathJoinSubstitution([pkg_share, "config", "ros2_controllers.yaml"])

    declared_arguments = [
        DeclareLaunchArgument(
            "include_gripper",
            default_value="true",
            description="Attach and spawn controller for the Robotiq 2F-85 gripper.",
        ),
        DeclareLaunchArgument(
            "rviz",
            default_value="true",
            description="Launch RViz2.",
        ),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=default_rviz_path,
            description="Absolute path to the RViz2 config file.",
        ),
    ]

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            default_xacro_path,
            " use_gazebo:=false",
            " include_gripper:=",
            LaunchConfiguration("include_gripper"),
        ]
    )

    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )

    # In ROS 2 Jazzy, ros2_control_node automatically subscribes to /robot_description
    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[controllers_yaml],
        output="both",
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

    # Spawn arm & gripper controllers once joint_state_broadcaster is active
    delay_controllers_after_jsb = RegisterEventHandler(
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
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription(
        declared_arguments
        + [
            robot_state_publisher_node,
            control_node,
            joint_state_broadcaster_spawner,
            delay_controllers_after_jsb,
            rviz_node,
        ]
    )
