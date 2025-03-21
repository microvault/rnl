import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = get_package_share_directory("playground")
    world_file = os.path.join(pkg_dir, "worlds", "my_world.world")
    target_file = os.path.join(pkg_dir, "worlds", "target.sdf")

    x_pose = "2.0"
    y_pose = "2.0"
    z_pose = "0.2"
    yaw = "1.57"

    gazebo_launch_file = os.path.join(
        get_package_share_directory("gazebo_ros"), "launch", "gazebo.launch.py"
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch_file),
        launch_arguments={"world": world_file, "playback_speed": "4.0"}.items(),
    )

    turtlebot3_launch_file = os.path.join(
        get_package_share_directory("turtlebot3_gazebo"),
        "launch",
        "spawn_turtlebot3.launch.py",
    )
    turtlebot3 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(turtlebot3_launch_file),
        launch_arguments={
            "x_pose": x_pose,
            "y_pose": y_pose,
            "z_pose": z_pose,
            "yaw": yaw,
        }.items(),
    )

    target = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-entity",
            "target",
            "-file",
            target_file,
            "-x",
            "0.0",
            "-y",
            "0.0",
            "-z",
            "0.001",
        ],
        output="screen",
    )

    return LaunchDescription([gazebo, turtlebot3, target])
