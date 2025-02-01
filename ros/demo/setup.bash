colcon build --packages-select demo
source install/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 launch demo turtlebot3_world.launch.py
