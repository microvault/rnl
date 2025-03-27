#!/bin/bash
source /opt/ros/humble/setup.bash
source /usr/share/gazebo/setup.sh

# Primeiro instalar dependências
rosdep install --from-paths src --ignore-src -r -y

# Depois limpar e buildar
rm -rf build/ install/ log/
colcon build

# Configurar ambiente
source install/setup.bash

export TURTLEBOT3_MODEL=burger
export LD_LIBRARY_PATH=/usr/local/lib:/opt/ros/humble/lib/libgazebo_ros_factory.so:$LD_LIBRARY_PATH
export GAZEBO_PLUGIN_PATH=/usr/local/lib:$GAZEBO_PLUGIN_PATH


# ros2 launch playground turtlebot_sim_world.launch.py
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
