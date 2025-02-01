import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('demo')
    world_file = os.path.join(pkg_dir, 'worlds', 'my_world.world')

    x_pose = "5.0"
    y_pose = "5.0"
    z_pose = "0.2"
    yaw = "1.57"

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')]),
        launch_arguments={'world': world_file}.items(),
    )

    turtlebot3_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('turtlebot3_gazebo'), 'launch', 'spawn_turtlebot3.launch.py')]),
        launch_arguments={
            'x_pose': x_pose,
            'y_pose': y_pose,
            'z_pose': z_pose,
            'yaw': yaw,
        }.items(),
    )
    main_node = Node(
        package='demo',
        executable='main.py',
        name='laser_processor',
        output='screen'
    )

    # Teleop node
    teleop = Node(
        package='turtlebot3_teleop',
        executable='teleop_keyboard',
        name='teleop_keyboard',
        prefix='xterm -e',
        output='screen'
    )


    return LaunchDescription([
        gazebo,
        turtlebot3_launch,
        teleop,
        main_node,
    ])
