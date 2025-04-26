SHELL := /bin/bash
MAP_NAME = map6
WORLD_SRC   := $(PWD)/src/playground/worlds/demo.world
WORLD_DST   := $(shell ros2 pkg prefix turtlebot3_gazebo)/share/turtlebot3_gazebo/worlds/turtlebot3_world.world

.PHONY: build run nav start world map save teleop add

env_setup = source /opt/ros/humble/setup.bash && \
            source /usr/share/gazebo/setup.sh && \
            source install/setup.bash && \
            export TURTLEBOT3_MODEL=burger && \
            export LD_LIBRARY_PATH=/usr/local/lib:/opt/ros/humble/lib/libgazebo_ros_factory.so:$$LD_LIBRARY_PATH && \
            export GAZEBO_PLUGIN_PATH=/usr/local/lib:$$GAZEBO_PLUGIN_PATH

build:
	@echo ">> Instalando dependências e construindo..."
	source /opt/ros/humble/setup.bash && \
	source /usr/share/gazebo/setup.sh && \
	rosdep install --from-paths src --ignore-src -r -y && \
	rm -rf build/ install/ log/ && \
	colcon build

setup:
	@echo ">> Iniciando setup..."
	$(env_setup) && \
	ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

nav:
	@echo ">> Iniciando navegação..."
	$(env_setup) && \
	ros2 launch turtlebot3_navigation2 navigation2.launch.py map:=src/playground/maps/$(MAP_NAME)/$(MAP_NAME).yaml

map:
	@echo ">> Iniciando mapeamento..."
	$(env_setup) && \
	ros2 launch turtlebot3_cartographer cartographer.launch.py use_sim_time:=true

save:
	$(env_setup) && \
	ros2 run nav2_map_server map_saver_cli -f src/playground/maps/map

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

# Caminhos do mundo
WORLD_SRC := $(CURDIR)/demo.world           # mundo gerado pelo script
WORLD_DST := $(shell ros2 pkg prefix turtlebot3_gazebo)/share/turtlebot3_gazebo/worlds/demo.world

world:
	@echo ">> Linkando mundo customizado..."
	@mkdir -p $(dir $(WORLD_DST))
	@ln -sf $(WORLD_SRC) $(WORLD_DST)
	@echo ">> Iniciando simulação..."
	@bash -c 'source /opt/ros/humble/setup.bash && \
		ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py world:=$(WORLD_DST)'

print:
	$(env_setup) && \
	ros2 run tf2_tools view_frames
