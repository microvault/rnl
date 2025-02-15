#!/bin/bash
source /opt/ros/humble/setup.bash

# Depois limpar e buildar
rm -rf build/ install/ log/
colcon build

# Configurar ambiente
source install/setup.bash


# Iniciar o launch
ros2 launch playground turtlebot_world.launch.py
