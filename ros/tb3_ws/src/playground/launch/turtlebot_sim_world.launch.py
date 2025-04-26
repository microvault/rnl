import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # diretórios dos pacotes
    pkg_playground = get_package_share_directory("playground")
    pkg_gazebo_ros = get_package_share_directory("gazebo_ros")
    pkg_tb3_gazebo = get_package_share_directory("turtlebot3_gazebo")
    pkg_tb3_desc = get_package_share_directory("turtlebot3_description")

    # arquivos do mundo / alvo
    world_file = os.path.join(pkg_playground, "worlds", "demo.world")
    target_file = os.path.join(pkg_playground, "worlds", "target.sdf")

    # argumento --use_sim_time
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Usar clock da simulação",
    )
    use_sim_time = LaunchConfiguration("use_sim_time")

    # Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gazebo.launch.py")
        ),
        launch_arguments={
            "world": world_file,
            "playback_speed": "4.0",
            "use_sim_time": use_sim_time,
        }.items(),
    )

    # Robot State Publisher (URDF + TF estático)
    tb3_state_pub = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tb3_desc, "launch", "robot_state_publisher.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    # Diff-drive controller (publica /odom e /tf dinâmico)
    tb3_drive = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tb3_gazebo, "launch", "turtlebot3_differential_drive.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    # LDS-02 LIDAR publisher (publica /scan)
    tb3_lidar = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tb3_gazebo, "launch", "turtlebot3_lds_2d.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    # Spawner do modelo TB3 no Gazebo
    tb3_spawn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tb3_gazebo, "launch", "spawn_turtlebot3.launch.py")
        ),
        launch_arguments={
            "x_pose": "0.0",
            "y_pose": "0.0",
            "z_pose": "0.2",
            "yaw": "1.57",
            "use_sim_time": use_sim_time,
        }.items(),
    )

    # Entidade-alvo (ex.: uma caixinha para navegar até ela)
    target = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-entity", "target",
            "-file", target_file,
            "-x", "0.0", "-y", "0.0", "-z", "0.001",
        ],
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    return LaunchDescription([
        use_sim_time_arg,
        gazebo,
        tb3_state_pub,
        tb3_drive,
        tb3_lidar,
        tb3_spawn,
        target,
    ])
