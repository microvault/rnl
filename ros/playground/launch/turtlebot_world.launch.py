# import os

# from ament_index_python.packages import get_package_share_directory
# from launch import LaunchDescription
# from launch.actions import IncludeLaunchDescription
# from launch.launch_description_sources import PythonLaunchDescriptionSource
# from launch_ros.actions import Node


# def generate_launch_description():
#     pkg_dir = get_package_share_directory("playground")
#     world_file = os.path.join(pkg_dir, "worlds", "my_world.world")
#     target_file = os.path.join(pkg_dir, "worlds", "target.sdf")

#     x_pose = "5.0"
#     y_pose = "5.0"
#     z_pose = "0.2"
#     yaw = "1.57"

#     # Carrega o launch do Gazebo passando o arquivo do mundo
#     gazebo_launch_file = os.path.join(
#         get_package_share_directory("gazebo_ros"), "launch", "gazebo.launch.py"
#     )
#     gazebo = IncludeLaunchDescription(
#         PythonLaunchDescriptionSource(gazebo_launch_file),
#         launch_arguments={"world": world_file, "playback_speed": "4.0"}.items(),
#     )
#     # Carrega o launch do Turtlebot3
#     turtlebot3_launch_file = os.path.join(
#         get_package_share_directory("turtlebot3_gazebo"),
#         "launch",
#         "spawn_turtlebot3.launch.py",
#     )
#     turtlebot3_launch = IncludeLaunchDescription(
#         PythonLaunchDescriptionSource(turtlebot3_launch_file),
#         launch_arguments={
#             "x_pose": x_pose,
#             "y_pose": y_pose,
#             "z_pose": z_pose,
#             "yaw": yaw,
#         }.items(),
#     )

#     main_node = Node(
#         package="playground",
#         executable="environment",
#         name="environment",
#         output="screen",
#     )

#     target_positions = [(2.0, 2.0), (7.0, 2.0), (2.0, 7.0), (7.0, 7.0)]
#     target_nodes = [
#         Node(
#             package="gazebo_ros",
#             executable="spawn_entity.py",
#             arguments=[
#                 "-entity",
#                 f"target_{i+1}",
#                 "-file",
#                 target_file,
#                 "-x",
#                 str(x),
#                 "-y",
#                 str(y),
#                 "-z",
#                 "0.0",
#             ],
#             output="screen",
#         )
#         for i, (x, y) in enumerate(target_positions)
#     ]

#     return LaunchDescription([gazebo, turtlebot3_launch, main_node] + target_nodes)

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    # Lança o bringup do turtlebot real (verifique se o arquivo está correto)
    real_robot_launch_file = os.path.join(
        get_package_share_directory("turtlebot3_bringup"),
        "launch",
        "robot.launch.py"
    )
    real_robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(real_robot_launch_file)
    )

    # Lança o seu node 'environment'
    main_node = Node(
        package="playground",
        executable="environment",
        name="environment",
        output="screen",
    )

    return LaunchDescription([real_robot_launch, main_node])
