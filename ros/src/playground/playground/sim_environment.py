import math
import os
import random

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import LaserScan
from stable_baselines3 import PPO


def clamp(value, vmin, vmax):
    return max(vmin, min(value, vmax))


def distance_to_goal(x: float, y: float, goal_x: float, goal_y: float) -> float:
    dist = np.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)
    if dist >= 9:
        return 9
    else:
        return dist


def angle_to_goal(
    x: float, y: float, theta: float, goal_x: float, goal_y: float
) -> float:
    o_t = np.array([np.cos(theta), np.sin(theta)])
    g_t = np.array([goal_x - x, goal_y - y])

    cross_product = np.cross(o_t, g_t)
    dot_product = np.dot(o_t, g_t)

    alpha = np.abs(np.arctan2(np.linalg.norm(cross_product), dot_product))

    return alpha


def min_max_scale(features, data_min, data_max):
    scaled = []
    for i, val in enumerate(features):
        mn = data_min[i]
        mx = data_max[i]
        if mx > mn:
            s = (val - mn) / (mx - mn)
        else:
            s = 0.0
        s = max(0.0, min(s, 0.99999994))
        scaled.append(s)
    return scaled


class InferenceModel(Node):
    def __init__(self):
        super().__init__("sim_environment")
        self.position = 0
        self.action = 0
        self.last_states = np.zeros(10)

        # self.position = Pose()
        # self.position.position.x = 1.07
        # self.position.position.y = 1.07
        # self.initialized = False

        self.goal_positions = [(2, 2), (7, 2), (2, 7), (7, 7)]
        self.goal_order = random.sample(
            range(len(self.goal_positions)), len(self.goal_positions)
        )
        self.goal_index = 0
        self.goal_x, self.goal_y = self.goal_positions[self.goal_order[self.goal_index]]

        self.lidar_ranges = [0.0] * 5
        self.last_action_oh = [0, 0, 0]

        self.data_min = [0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 0.0, 0, 0, 0]
        self.data_max = [3.5, 3.5, 3.5, 3.5, 3.5, 9.0, 3.5, 1, 1, 1]

        pkg_dir = get_package_share_directory("playground")
        model_path = os.path.join(pkg_dir, "models", "model")
        self.get_logger().info(f"Model path: {model_path}")
        self.model = PPO.load(model_path)

        qos_profile = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)

        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.laser_callback, qos_profile
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, qos_profile
        )
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

    def laser_callback(self, msg):
        ranges = msg.ranges
        n = len(ranges)
        if n < 5:
            return
        step = (n - 1) / 4.0
        for i in range(5):
            idx = int(round(step * i))
            if idx >= n:
                idx = n - 1
            val = ranges[idx]
            if not math.isfinite(val):
                val = msg.range_max
            self.lidar_ranges[i] = val

    def odom_callback(self, msg):
        # if not self.initialized:
        #     # Mantém x e y iniciais, só atualiza a orientação
        #     self.position.orientation = msg.pose.pose.orientation
        #     self.initialized = True
        # else:
        self.position = msg.pose.pose

    def move_robot(self, action):
        if action == 0:
            vl, vr = 0.10, 0.0
        elif action == 1:
            vl, vr = 0.08, -0.08
        elif action == 2:
            vl, vr = 0.08, 0.08
        else:
            vl, vr = 0.0, 0.0

        twist = Twist()
        twist.linear.x = vl
        twist.angular.z = vr
        self.cmd_vel_pub.publish(twist)

    def update(self):
        self.action, _ = self.model.predict(self.last_states)

        self.move_robot(self.action)

        x = self.position.position.x
        y = self.position.position.y
        z = self.position.orientation.z
        w = self.position.orientation.w
        theta = 2.0 * math.atan2(z, w)

        dist = float(distance_to_goal(x, y, self.goal_x, self.goal_y))
        alpha = float(angle_to_goal(x, y, theta, self.goal_x, self.goal_y))

        clamped_lidar = [float(clamp(val, 0.5, 3.5)) for val in self.lidar_ranges]
        clamped_dist = float(clamp(dist, 1.0, 9.0))
        clamped_alpha = float(clamp(alpha, 0.0, 3.5))

        action_one_hot = np.eye(3)[self.action]

        norm_lidar = (np.array(clamped_lidar, dtype=np.float32) - 0.5) / (3.5 - 0.5)
        norm_dist = np.array(clamped_dist, dtype=np.float32) / 9.0
        norm_alpha = np.array(clamped_alpha, dtype=np.float32) / 3.5

        states = np.concatenate(
            (
                norm_lidar,
                np.array(action_one_hot, dtype=np.int16),
                np.array([norm_dist], dtype=np.float32),
                np.array([norm_alpha], dtype=np.float32),
            )
        )

        self.last_states = states

        if dist <= 1.2:
            self.goal_index += 1
            if self.goal_index >= len(self.goal_order):
                self.goal_order = random.sample(
                    range(len(self.goal_positions)), len(self.goal_positions)
                )
                self.goal_index = 0
            self.goal_x, self.goal_y = self.goal_positions[
                self.goal_order[self.goal_index]
            ]
            self.get_logger().info(f"New goal: ({self.goal_x}, {self.goal_y})")


def main(args=None):
    rclpy.init(args=args)
    navigator = InferenceModel()
    try:
        rclpy.spin_once(navigator)
        while rclpy.ok:

            rclpy.spin_once(navigator)
            navigator.update()
    except KeyboardInterrupt:
        pass
    finally:
        navigator.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
