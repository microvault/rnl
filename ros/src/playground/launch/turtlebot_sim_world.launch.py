import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory("playground")
    world_file = os.path.join(pkg_dir, "worlds", "my_world.world")
    target_file = os.path.join(pkg_dir, "worlds", "target.sdf")

    # Declara argumento para usar sim_time
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", default_value="true", description="Utilizar clock da simulação"
    )
    use_sim_time = LaunchConfiguration("use_sim_time")

    x_pose = "0.0"
    y_pose = "0.0"
    z_pose = "0.2"
    yaw = "1.57"

    gazebo_launch_file = os.path.join(
        get_package_share_directory("gazebo_ros"), "launch", "gazebo.launch.py"
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch_file),
        launch_arguments={
            "world": world_file,
            "playback_speed": "4.0",
            "use_sim_time": use_sim_time,
        }.items(),
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
            "use_sim_time": use_sim_time,
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
        parameters=[{"use_sim_time": True}],
    )
    return LaunchDescription([use_sim_time_arg, gazebo, turtlebot3, target])
