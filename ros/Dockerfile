ARG DEBIAN_FRONTEND=noninteractive
FROM osrf/ros:humble-desktop-full
ENV ROS_DISTRO=humble

SHELL [ "/bin/bash" , "-c"]

RUN apt-get update && apt install -y xterm tmux git wget

RUN mkdir /turtlebot3_ws
RUN mkdir /turtlebot3_ws/src

RUN apt install -y ros-${ROS_DISTRO}-gazebo-ros
RUN apt install -y ros-${ROS_DISTRO}-gazebo-ros-pkgs
RUN apt install -y ros-${ROS_DISTRO}-turtlebot3-gazebo
RUN apt install -y ros-${ROS_DISTRO}-turtlebot3-msgs
RUN apt install -y ros-${ROS_DISTRO}-turtlebot3

RUN apt install -y python3-argcomplete
RUN apt install -y python3-colcon-common-extensions
RUN apt install -y libboost-system-dev
RUN apt install -y build-essential
RUN apt install -y libudev-dev

# config tmux
RUN echo "unbind -n Tab"                                                                    >> ~/.tmux.conf
RUN echo "set -g window-style        'fg=#ffffff,bg=#8445ca'"                               >> ~/.tmux.conf
RUN echo "set -g window-active-style 'fg=#ffffff,bg=#5e2b97'"                               >> ~/.tmux.conf
RUN echo "set-option -g default-shell '/bin/bash'"                                          >> ~/.tmux.conf
RUN echo "run-shell '. /opt/ros/humble/setup.bash'"                                         >> ~/.tmux.conf
RUN echo "set -g mouse on"                                                                  >> ~/.tmux.conf
RUN echo "bind-key -n C-Left select-pane -L"                                                >> ~/.tmux.conf
RUN echo "bind-key -n C-Right select-pane -R"                                               >> ~/.tmux.conf
RUN echo "bind-key -n C-Up select-pane -U"                                                  >> ~/.tmux.conf
RUN echo "bind-key -n C-Down select-pane -D"                                                >> ~/.tmux.conf
RUN echo "bind -n M-Right split-window -h"                                                  >> ~/.tmux.conf
RUN echo "bind -n M-Down split-window -v"                                                   >> ~/.tmux.conf

RUN apt install -y python3-pip
RUN pip install setuptools==58.2.0
RUN pip install stable_baselines3
RUN pip uninstall -y numpy
RUN pip install "numpy<2"
RUN pip install --force-reinstall stable-baselines3


WORKDIR /turtlebot3_ws
