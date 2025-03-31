from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Lança o bringup do turtlebot real (verifique se o arquivo está correto)
    # real_robot_launch_file = os.path.join(
    #     get_package_share_directory("turtlebot3_bringup"), "launch", "robot.launch.py"
    # )
    # real_robot_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(real_robot_launch_file)
    # )

    # Lança o seu node 'environment'
    main_node = Node(
        package="playground",
        executable="environment",
        name="environment",
        output="screen",
    )

    return LaunchDescription([main_node])
