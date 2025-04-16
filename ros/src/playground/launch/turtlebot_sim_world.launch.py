import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory("playground")

    # Arquivo do mundo que será carregado no Gazebo
    world_file = os.path.join(pkg_dir, "worlds", "my_world.world")

    # Alvo para spawnar como entidade no Gazebo (opcional)
    target_file = os.path.join(pkg_dir, "worlds", "target.sdf")

    # Usar clock da simulação
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Utilizar clock da simulação"
    )
    use_sim_time = LaunchConfiguration("use_sim_time")

    # Arquivo de mapa (yaml) passado como argumento
    map_file_arg = DeclareLaunchArgument(
        "map_file",
        default_value=os.path.join(pkg_dir, "maps", "map.yaml"),
        description="Caminho para o arquivo de mapa (.yaml)"
    )
    map_file = LaunchConfiguration("map_file")

    # Arquivo de configuração do RViz (deixe um .rviz ou .yaml com as suas configs)
    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value=os.path.join(pkg_dir, "rviz", "nav2_default_view.rviz"),
        description="Configurações do RViz"
    )
    rviz_config = LaunchConfiguration("rviz_config")

    # Coordenadas iniciais do TurtleBot3 no Gazebo
    x_pose = "0.0"
    y_pose = "0.0"
    z_pose = "0.2"
    yaw = "1.57"

    # Lança Gazebo
    gazebo_launch_file = os.path.join(
        get_package_share_directory("gazebo_ros"),
        "launch",
        "gazebo.launch.py"
    )
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch_file),
        launch_arguments={
            "world": world_file,
            "playback_speed": "4.0",
            "use_sim_time": use_sim_time,
        }.items(),
    )

    # Lança o Turtlebot3 no Gazebo
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

    # (Opcional) Spawna um alvo, se quiser colocar algum objeto no mundo
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

    # Node para carregar o mapa
    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "yaml_filename": map_file,
            }
        ],
    )

    # Node do AMCL (publica /amcl_pose)
    amcl = Node(
        package="nav2_amcl",
        executable="amcl",
        name="amcl",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time
                # Demais parâmetros do AMCL, se quiser.
            }
        ],
    )

    # Node do RViz (sem todo Navigation2; só mostra o mapa, TF e amcl_pose)
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    return LaunchDescription([
        use_sim_time_arg,
        map_file_arg,
        rviz_config_arg,
        gazebo,
        turtlebot3,
        target,
        map_server,
        amcl,
        rviz_node,
    ])
