ARG DEBIAN_FRONTEND=noninteractive
FROM osrf/ros:humble-desktop-full

ENV ROS_DISTRO=humble
SHELL ["/bin/bash", "-c"]

# ----------------------------------------------------------------------
# 1) Atualiza cache APT uma única vez e instala tudo junto
# ----------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    xterm tmux git wget python3-pip python3-argcomplete \
    python3-colcon-common-extensions libboost-system-dev \
    build-essential libudev-dev \
    ros-${ROS_DISTRO}-gazebo-ros \
    ros-${ROS_DISTRO}-gazebo-ros-pkgs \
    ros-${ROS_DISTRO}-turtlebot3 \
    ros-${ROS_DISTRO}-turtlebot3-gazebo \
    ros-${ROS_DISTRO}-turtlebot3-msgs \
    ros-${ROS_DISTRO}-cartographer \
    ros-${ROS_DISTRO}-cartographer-ros \
    ros-${ROS_DISTRO}-navigation2 \
    ros-${ROS_DISTRO}-nav2-bringup \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------
# 2) Workspace
# ----------------------------------------------------------------------
RUN mkdir -p /tb3_ws/src
WORKDIR /tb3_ws

# ----------------------------------------------------------------------
# 3) Configura arquivo de navegação do TurtleBot3
# ----------------------------------------------------------------------
RUN wget -q https://raw.githubusercontent.com/ROBOTIS-GIT/turtlebot3/humble-devel/turtlebot3_navigation2/param/burger.yaml \
    -O /opt/ros/${ROS_DISTRO}/share/turtlebot3_navigation2/param/burger.yaml

# ----------------------------------------------------------------------
# 4) Pip packages (mantendo setuptools antigo para compatibilidade)
# ----------------------------------------------------------------------
RUN pip install --no-cache-dir setuptools==58.2.0
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
# ----------------------------------------------------------------------
# 5) Tmux: tema e atalhos básicos
# ----------------------------------------------------------------------
RUN cat >> /root/.tmux.conf <<'EOF'
unbind -n Tab
set -g window-style        'fg=#ffffff,bg=#8445ca'
set -g window-active-style 'fg=#ffffff,bg=#5e2b97'
set-option -g default-shell '/bin/bash'
run-shell '. /opt/ros/${ROS_DISTRO}/setup.bash'
set -g mouse on
bind-key -n C-Left  select-pane -L
bind-key -n C-Right select-pane -R
bind-key -n C-Up    select-pane -U
bind-key -n C-Down  select-pane -D
bind -n M-Right split-window -h
bind -n M-Down  split-window -v
EOF
