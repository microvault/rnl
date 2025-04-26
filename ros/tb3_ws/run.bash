#!/bin/bash

rm -rf build/ install/ log/
colcon build

source install/setup.bash

export TURTLEBOT3_MODEL=burger
export LD_LIBRARY_PATH=/usr/local/lib:/opt/ros/humble/lib/libgazebo_ros_factory.so:$LD_LIBRARY_PATH
export GAZEBO_PLUGIN_PATH=/usr/local/lib:$GAZEBO_PLUGIN_PATH

ros2 run playground sim_environment
