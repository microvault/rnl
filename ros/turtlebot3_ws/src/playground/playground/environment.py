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
from rclpy.time import Time


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


class InferenceModel(Node):
    def __init__(self):
        super().__init__("environment")
        self.get_logger().info("[INIT] Inicializando InferenceModel")

        # Configurações iniciais
        self.desired_x = 1.07
        self.desired_y = 1.07
        self.desired_yaw = 4.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_yaw = 0.0
        self.offset_computed = False
        self.position = Pose()
        self.last_states = np.zeros(10)
        self.goal_positions = [(1.8, 0.35)]
        self.goal_order = random.sample(range(len(self.goal_positions)), len(self.goal_positions))
        self.goal_index = 0
        self.goal_x, self.goal_y = self.goal_positions[self.goal_order[self.goal_index]]
        self.lidar_ranges = [0.0] * 5
        self.last_action_oh = [0, 0, 0]

        # Variáveis de sincronização
        self.last_scan_stamp = None
        self.last_odom_stamp = None
        self.sync_threshold = 0.1  # segundos

        qos_profile = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.laser_callback, qos_profile)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, qos_profile)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Carrega o modelo treinado
        pkg_dir = get_package_share_directory("playground")
        model_path = os.path.join(pkg_dir, "models", "model")
        self.get_logger().info(f"[INIT] Model path: {model_path}")
        self.model = PPO.load(model_path)
        self.get_logger().info(f"[INIT] Novo objetivo: ({self.goal_x}, {self.goal_y})")

    def laser_callback(self, msg):
        self.last_scan_stamp = msg.header.stamp
        now = self.get_clock().now()
        scan_delay = abs((now - Time.from_msg(msg.header.stamp)).nanoseconds) / 1e9
        self.get_logger().debug(f"[DEBUG] Laser callback: scan_delay = {scan_delay:.3f}s")

        ranges = msg.ranges
        n = len(ranges)
        if n < 5:
            self.get_logger().warn("[DEBUG] Número insuficiente de raios no LaserScan")
            return
        step = (n - 1) / 4.0
        for i in range(5):
            idx = int(round(step * i))
            idx = min(idx, n - 1)
            val = ranges[idx]
            if not math.isfinite(val):
                val = msg.range_max
            self.lidar_ranges[i] = val
        self.get_logger().debug(f"[DEBUG] lidar_ranges atualizados: {self.lidar_ranges}")

    def odom_callback(self, msg):
        self.last_odom_stamp = msg.header.stamp
        now = self.get_clock().now()
        odom_delay = abs((now - Time.from_msg(msg.header.stamp)).nanoseconds) / 1e9
        self.get_logger().debug(f"[DEBUG] Odom callback: odom_delay = {odom_delay:.3f}s")

        real_x = msg.pose.pose.position.x
        real_y = msg.pose.pose.position.y
        real_z = msg.pose.pose.orientation.z
        real_w = msg.pose.pose.orientation.w
        real_yaw = 2.0 * math.atan2(real_z, real_w)
        self.get_logger().debug(f"[DEBUG] Odom recebido: x={real_x:.2f}, y={real_y:.2f}, yaw={real_yaw:.2f}")

        if not self.offset_computed:
            self.offset_x = self.desired_x - real_x
            self.offset_y = self.desired_y - real_y
            self.offset_yaw = self.desired_yaw - real_yaw
            self.offset_computed = True
            self.get_logger().info(f"[DEBUG] Offset computado: offset_x={self.offset_x:.2f}, offset_y={self.offset_y:.2f}, offset_yaw={self.offset_yaw:.2f}")

        shifted_x = real_x + self.offset_x
        shifted_y = real_y + self.offset_y
        shifted_yaw = real_yaw + self.offset_yaw
        s_z = math.sin(shifted_yaw / 2.0)
        s_w = math.cos(shifted_yaw / 2.0)
        self.position.position.x = shifted_x
        self.position.position.y = shifted_y
        self.position.orientation.z = s_z
        self.position.orientation.w = s_w
        self.get_logger().debug(f"[DEBUG] Posição atual (shifted): x={shifted_x:.2f}, y={shifted_y:.2f}, yaw={shifted_yaw:.2f}")


    def move_robot(self, action):
        if action == 0:
            vl, vr = 0.10, 0.0
        elif action == 1:
            vl, vr = 0.08, -0.36
        elif action == 2:
            vl, vr = 0.08, 0.36
        else:
            vl, vr = 0.0, 0.0
        twist = Twist()
        twist.linear.x = vl
        twist.angular.z = vr
        self.cmd_vel_pub.publish(twist)
        self.get_logger().debug(f"[DEBUG] Publicando cmd_vel: linear={vl:.2f}, angular={vr:.2f}")

    def update(self):
        now = self.get_clock().now()
        if self.last_scan_stamp is None or self.last_odom_stamp is None:
            self.get_logger().warn("[DEBUG] Aguardando primeiras mensagens de sensor/odom...")
            return

        odom_delay = abs((now - Time.from_msg(self.last_odom_stamp)).nanoseconds) / 1e9
        scan_delay = abs((now - Time.from_msg(self.last_scan_stamp)).nanoseconds) / 1e9
        self.get_logger().debug(f"[DEBUG] Delays - Odom: {odom_delay:.3f}s, Laser: {scan_delay:.3f}s")

        if odom_delay > self.sync_threshold or scan_delay > self.sync_threshold:
            self.get_logger().warn(f"[DEBUG] Mensagens fora de sincronia: Odom delay={odom_delay:.3f}s, Laser delay={scan_delay:.3f}s. Aguardando próximo step...")
            return

        # Processa o step se tudo estiver sincronizado
        self.action, _ = self.model.predict(self.last_states)
        self.get_logger().debug(f"[DEBUG] Ação prevista: {self.action}")
        self.move_robot(self.action)

        x = self.position.position.x
        y = self.position.position.y
        z = self.position.orientation.z
        w = self.position.orientation.w
        theta = 2.0 * math.atan2(z, w)
        dist = float(distance_to_goal(x, y, self.goal_x, self.goal_y))
        alpha = float(angle_to_goal(x, y, theta, self.goal_x, self.goal_y))
        self.get_logger().debug(f"[DEBUG] Distância ao objetivo: {dist:.2f}, Ângulo: {alpha:.2f}")

        clamped_lidar = [float(clamp(val, 0.5, 3.5)) for val in self.lidar_ranges]
        clamped_dist = float(clamp(dist, 1.0, 9.0))
        clamped_alpha = float(clamp(alpha, 0.0, 3.5))
        action_one_hot = np.eye(3)[self.action]

        norm_lidar = (np.array(clamped_lidar, dtype=np.float32) - 0.5) / (3.5 - 0.5)
        norm_lidar = np.where(norm_lidar < 0.1, np.random.uniform(0.1, 0.15, norm_lidar.shape), norm_lidar)
        norm_dist = np.array(clamped_dist, dtype=np.float32) / 9.0
        norm_alpha = np.array(clamped_alpha, dtype=np.float32) / 3.5

        states = np.concatenate((norm_lidar,
                                    np.array(action_one_hot, dtype=np.int16),
                                    np.array([norm_dist], dtype=np.float32),
                                    np.array([norm_alpha], dtype=np.float32)))
        self.last_states = states
        self.get_logger().debug(f"[DEBUG] Estado atualizado: {states}")

        if dist <= 0.5:
            self.get_logger().info("[DEBUG] Objetivo alcançado. Esperando 3 segundos...")
            time.sleep(3)
            self.goal_index = (self.goal_index + 1) % len(self.goal_positions)
            self.goal_x, self.goal_y = self.goal_positions[self.goal_order[self.goal_index]]
            self.get_logger().info(f"[DEBUG] Novo objetivo: ({self.goal_x}, {self.goal_y})")


def main(args=None):
    rclpy.init(args=args)
    navigator = InferenceModel()
    try:
        rclpy.spin_once(navigator)
        while rclpy.ok():
            rclpy.spin_once(navigator)
            navigator.update()
    except KeyboardInterrupt:
        pass
    finally:
        navigator.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
