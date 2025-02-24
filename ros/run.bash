#!/bin/bash
source /opt/ros/humble/setup.bash
source /usr/share/gazebo/setup.sh

RUN pip uninstall -y numpy
RUN pip install "numpy<2"
RUN pip install --force-reinstall stable-baselines3


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

ls -l src/playground/worlds/my_world.world

# Iniciar o launch
ros2 launch playground turtlebot_world.launch.py
