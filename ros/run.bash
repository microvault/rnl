#!/bin/bash
rm -rf build/ install/ log/
colcon build --symlink-install

# Configurar ambiente
source /Users/nicolasalan/microvault/rnl/ros/install/setup.sh

export TURTLEBOT3_MODEL=burger
export LD_LIBRARY_PATH=/usr/local/lib:/opt/ros/humble/lib/libgazebo_ros_factory.so:$LD_LIBRARY_PATH
export GAZEBO_PLUGIN_PATH=/usr/local/lib:$GAZEBO_PLUGIN_PATH

ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py
