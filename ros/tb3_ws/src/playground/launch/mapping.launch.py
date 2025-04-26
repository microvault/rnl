from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Lança o seu node 'environment'
    main_node = Node(
        package="playground",
        executable="mapping",
        name="mapping",
        output="screen",
    )

    return LaunchDescription([main_node])
