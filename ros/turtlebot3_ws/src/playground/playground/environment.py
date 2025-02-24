import math
import os
import random
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import LaserScan
from stable_baselines3 import PPO

# Variáveis globais para limites
LIDAR_MIN = 0.0
LIDAR_MAX = 3.5

DIST_MIN = 0.0
DIST_MAX = 2.8284271247461903

ALPHA_MIN = 0.0
ALPHA_MAX = 3.5

FACTOR_REAL_ALPHA = 0.01216


def clamp(value, vmin, vmax):
    return max(vmin, min(value, vmax))


def distance_to_goal(
    x: float, y: float, goal_x: float, goal_y: float, max_value: float
) -> float:
    dist = np.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)
    if dist >= max_value:
        return max_value
    else:
        return dist


def angle_to_goal(
    x: float, y: float, theta: float, goal_x: float, goal_y: float
) -> float:
    # Usa np.abs e np.arctan2 do numpy para precisão
    o_t = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    g_t = np.array([goal_x - x, goal_y - y], dtype=np.float64)
    cross = np.cross(o_t, g_t)
    dot = np.dot(o_t, g_t)
    alpha = 2 * np.arctan2(np.abs(cross), dot)
    return alpha


class InferenceModel(Node):
    def __init__(self):
        super().__init__("environment")
        self.get_logger().info("[INIT] Inicializando InferenceModel")

        # Definindo posição inicial desejada
        self.desired_x = 1.07
        self.desired_y = 1.07
        self.desired_yaw = 4.70

        # Inicializando offsets e pose com os valores desejados
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_yaw = 0.0
        self.offset_computed = False

        self.position = Pose()
        self.position.position.x = self.desired_x
        self.position.position.y = self.desired_y
        s_z = math.sin(self.desired_yaw / 2.0)
        s_w = math.cos(self.desired_yaw / 2.0)
        self.position.orientation.z = s_z
        self.position.orientation.w = s_w

        self.last_states = np.zeros(10)
        self.goal_positions = [(0.35, 0.35)]
        self.goal_order = random.sample(
            range(len(self.goal_positions)), len(self.goal_positions)
        )
        self.goal_index = 0
        self.goal_x, self.goal_y = self.goal_positions[self.goal_order[self.goal_index]]
        self.lidar_ranges = [0.0] * 5
        self.last_action_oh = [0, 0, 0]

        # Configuração dos subscribers e publisher
        qos_profile = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.laser_callback, qos_profile
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, qos_profile
        )
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Carrega o modelo treinado
        pkg_dir = get_package_share_directory("playground")
        model_path = os.path.join(pkg_dir, "models", "model")
        self.get_logger().info(f"[INIT] Model path: {model_path}")
        self.model = PPO.load(model_path)
        self.get_logger().info(f"[INIT] Novo objetivo: ({self.goal_x}, {self.goal_y})")

    def laser_callback(self, msg):
        self.last_scan_stamp = msg.header.stamp
        ranges = msg.ranges
        n = len(ranges)
        if n < 5:
            self.get_logger().warn("Número insuficiente de raios no LaserScan")
            return
        step = (n - 1) / 4.0
        for i in range(5):
            idx = int(round(step * i))
            idx = min(idx, n - 1)
            val = ranges[idx]
            if not math.isfinite(val):
                val = msg.range_max
            self.lidar_ranges[i] = val

    def odom_callback(self, msg):
        self.last_odom_stamp = msg.header.stamp
        real_x = msg.pose.pose.position.x
        real_y = msg.pose.pose.position.y
        real_z = msg.pose.pose.orientation.z
        real_w = msg.pose.pose.orientation.w
        real_yaw = 2.0 * math.atan2(real_z, real_w)

        if not self.offset_computed:
            self.offset_x = self.desired_x - real_x
            self.offset_y = self.desired_y - real_y
            self.offset_yaw = self.desired_yaw - real_yaw
            self.offset_computed = True

        shifted_x = real_x + self.offset_x
        shifted_y = real_y + self.offset_y
        shifted_yaw = real_yaw + self.offset_yaw
        s_z = math.sin(shifted_yaw / 2.0)
        s_w = math.cos(shifted_yaw / 2.0)
        self.position.position.x = shifted_x
        self.position.position.y = shifted_y
        self.position.orientation.z = s_z
        self.position.orientation.w = s_w

    def update(self):
        # Para fins de teste, forçamos a ação 0
        self.action = 0
        # self.action, _ = self.model.predict(self.last_states)

        # # Converte ação em velocidades
        # if self.action == 0:
        #     vl, vr = 0.10, 0.0
        # elif self.action == 1:
        #     vl, vr = 0.08, -0.72
        # elif self.action == 2:
        #     vl, vr = 0.08, 0.72
        # else:
        #     vl, vr = 0.0, 0.0

        # twist = Twist()
        # twist.linear.x = vl
        # twist.angular.z = vr
        # self.cmd_vel_pub.publish(twist)

        x = self.position.position.x
        y = self.position.position.y
        z = self.position.orientation.z
        w = self.position.orientation.w
        theta = 2.0 * math.atan2(z, w)
        dist = distance_to_goal(x, y, self.goal_x, self.goal_y, DIST_MAX)
        alpha = angle_to_goal(x, y, theta, self.goal_x, self.goal_y)
        clamped_lidar = [
            float(clamp(val, LIDAR_MIN, LIDAR_MAX)) for val in self.lidar_ranges
        ]
        clamped_dist = float(clamp(dist, DIST_MIN, DIST_MAX))
        clamped_alpha = float(clamp(alpha, ALPHA_MIN, ALPHA_MAX))
        action_one_hot = np.eye(3)[self.action]
        norm_lidar = (np.array(clamped_lidar, dtype=np.float32) - LIDAR_MIN) / (
            LIDAR_MAX - LIDAR_MIN
        )
        norm_dist = np.array(clamped_dist, dtype=np.float32) / (DIST_MAX - DIST_MIN)
        norm_alpha = (
            np.array(clamped_alpha, dtype=np.float32) / (ALPHA_MAX - ALPHA_MIN)
        ) - FACTOR_REAL_ALPHA

        if norm_alpha < 0.0:
            norm_alpha = 0.0

        self.get_logger().info(f"Distância: {norm_dist}, Alpha: {norm_alpha}")

        states = np.concatenate(
            (
                norm_lidar,
                np.array(action_one_hot, dtype=np.int16),
                np.array([norm_dist], dtype=np.float32),
                np.array([norm_alpha], dtype=np.float32),
            )
        )
        self.last_states = states

        if dist <= 0.3:
            time.sleep(3)
            self.goal_index = (self.goal_index + 1) % len(self.goal_positions)
            self.goal_x, self.goal_y = self.goal_positions[
                self.goal_order[self.goal_index]
            ]
            self.get_logger().info(f"Novo objetivo: ({self.goal_x}, {self.goal_y})")


def main(args=None):
    rclpy.init(args=args)
    navigator = InferenceModel()
    while rclpy.ok():
        rclpy.spin_once(navigator)
        navigator.update()
    navigator.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
