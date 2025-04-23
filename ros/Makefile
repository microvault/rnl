SHELL := /bin/bash
MAP_NAME = map5

.PHONY: build run nav start

env_setup = source /opt/ros/humble/setup.bash && \
            source /usr/share/gazebo/setup.sh && \
            source install/setup.bash && \
            export TURTLEBOT3_MODEL=burger && \
            export LD_LIBRARY_PATH=/usr/local/lib:/opt/ros/humble/lib/libgazebo_ros_factory.so:$$LD_LIBRARY_PATH && \
            export GAZEBO_PLUGIN_PATH=/usr/local/lib:$$GAZEBO_PLUGIN_PATH

build:
	@echo "Instalando dependências e construindo..."
	source /opt/ros/humble/setup.bash && \
	source /usr/share/gazebo/setup.sh && \
	rosdep install --from-paths src --ignore-src -r -y && \
	rm -rf build/ install/ log/ && \
	colcon build

setup:
	$(env_setup) && \
	ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

nav:
	$(env_setup) && \
	ros2 launch turtlebot3_navigation2 navigation2.launch.py map:=src/playground/maps/$(MAP_NAME)/$(MAP_NAME).yaml

map:
	$(env_setup) && \
	ros2 launch turtlebot3_cartographer cartographer.launch.py

save:
	$(env_setup) && \
	ros2 run nav2_map_server map_saver_cli -f src/map

start:
	$(env_setup) && \
	ros2 launch playground turtlebot_sim_world.launch.py

run:
	$(env_setup) && \
	ros2 run playground sim_environment

teleop:
	$(env_setup) && \
	ros2 run turtlebot3_teleop teleop_keyboard

kill:
	pkill -f 'ros2|gzserver|gzclient|gazebo'

add:
	tmux new-window

print:
	$(env_setup) && \
	ros2 run tf2_tools view_frames
