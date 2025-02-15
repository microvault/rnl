ARG DEBIAN_FRONTEND=noninteractive
FROM ros:humble-ros-base
ENV ROS_DISTRO=humble

SHELL [ "/bin/bash", "-c" ]

RUN apt-get update && apt install -y git wget

RUN mkdir /turtlebot3_ws && mkdir /turtlebot3_ws/src

RUN apt install -y python3-argcomplete \
    && apt install -y python3-colcon-common-extensions \
    && apt install -y libboost-system-dev \
    && apt install -y build-essential \
    && apt install -y libudev-dev

RUN apt install -y python3-pip
RUN pip install setuptools==58.2.0
RUN pip install stable_baselines3
RUN pip uninstall -y numpy
RUN pip install "numpy<2"
RUN pip install --force-reinstall stable-baselines3

WORKDIR /turtlebot3_ws

COPY ./playground ./src/playground
COPY ./demo.bash ./demo.bash

RUN apt install -y ros-${ROS_DISTRO}-turtlebot3-msgs
RUN apt install -y ros-${ROS_DISTRO}-turtlebot3-teleop

RUN . /opt/ros/humble/setup.sh && colcon build

RUN echo "source /opt/ros/humble/setup.sh" >> /root/.bashrc \
    && echo "source /turtlebot3_ws/install/setup.sh" >> /root/.bashrc

ENV ROS_DOMAIN_ID=30

RUN apt-get install -y iputils-ping

CMD ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source /turtlebot3_ws/install/setup.bash && ./demo.bash"]
