"""Launch RViz2 visualization with joint_state_publisher_gui for WISERLI-ATAL-C1.1 + Robotiq 2F-85."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
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

    declared_arguments = [
        DeclareLaunchArgument(
            "include_gripper",
            default_value="true",
            description="Attach the Robotiq 2F-85 gripper to tool0.",
        ),
        DeclareLaunchArgument(
            "gui",
            default_value="true",
            description="Launch joint_state_publisher_gui for interactive joint sliders.",
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
        output="screen",
        parameters=[robot_description],
    )

    joint_state_publisher_gui_node = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        output="screen",
        condition=IfCondition(LaunchConfiguration("gui")),
    )

    joint_state_publisher_node = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        output="screen",
        condition=UnlessCondition(LaunchConfiguration("gui")),
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
            joint_state_publisher_gui_node,
            joint_state_publisher_node,
            rviz_node,
        ]
    )
