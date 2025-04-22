import math
import os
import random

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import LaserScan

def clamp(value, vmin, vmax):
    return max(vmin, min(value, vmax))

def distance_to_goal(x: float, y: float, goal_x: float, goal_y: float) -> float:
    dist = np.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)
    return dist if dist < 9 else 9

def angle_to_goal(x: float, y: float, theta: float, goal_x: float, goal_y: float) -> float:
    o_t = np.array([np.cos(theta), np.sin(theta)])
    g_t = np.array([goal_x - x, goal_y - y])
    cross = np.cross(o_t, g_t)
    dot = np.dot(o_t, g_t)
    return float(abs(np.arctan2(np.linalg.norm(cross), dot)))

class InferenceModel(Node):
    def __init__(self):
        super().__init__('sim_environment')
        self.position = None
        self.last_states = np.zeros(10)

        # metas aleatórias
        self.goal_positions = [(2, 2), (7, 2), (2, 7), (7, 7)]
        self.goal_order = random.sample(range(len(self.goal_positions)), len(self.goal_positions))
        self.goal_index = 0
        self.goal_x, self.goal_y = self.goal_positions[self.goal_order[self.goal_index]]

        self.lidar_ranges = [0.0] * 5

        # escala min/max
        self.data_min = [0.5]*5 + [1.0, 0.0, 0, 0, 0]
        self.data_max = [3.5]*5 + [9.0, 3.5, 1, 1, 1]

        pkg_dir = get_package_share_directory('playground')
        model_path = os.path.join(pkg_dir, 'models', 'model')
        self.get_logger().info(f'Model path: {model_path}')

        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        lidar_qos = QoSProfile(
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,   # igual ao publisher
            durability=QoSDurabilityPolicy.VOLATILE
        )

        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, lidar_qos)
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',          # ajusta se usar namespace
            self.amcl_callback,
            qos
        )
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.create_timer(0.1, self.update)

    def laser_callback(self, msg: LaserScan):
        ranges = msg.ranges
        if len(ranges) < 5:
            return
        step = (len(ranges) - 1) / 4.0
        for i in range(5):
            idx = min(int(round(step * i)), len(ranges)-1)
            val = ranges[idx] if math.isfinite(ranges[idx]) else msg.range_max
            self.lidar_ranges[i] = val

    def amcl_callback(self, msg: PoseWithCovarianceStamped):
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
        if self.position is None:
            self.get_logger().info('Ainda não recebi /amcl_pose, aguardando...')
            return

        action = 0

        self.move_robot(action)

        x = self.position.position.x
        y = self.position.position.y
        z = self.position.orientation.z
        w = self.position.orientation.w
        theta = 2.0 * math.atan2(z, w)

        dist = distance_to_goal(x, y, self.goal_x, self.goal_y)
        alpha = angle_to_goal(x, y, theta, self.goal_x, self.goal_y)

        # normaliza
        lidar_clamped = np.clip(self.lidar_ranges, 0.5, 3.5)
        norm_lidar = (lidar_clamped - 0.5) / (3.5 - 0.5)
        norm_dist = dist / 9.0
        norm_alpha = alpha / 3.5

        self.last_states = np.concatenate([
            norm_lidar,
            np.eye(3)[action],
            [norm_dist],
            [norm_alpha]
        ]).astype(np.float32)

        self.get_logger().info(f'Estado: ({self.last_states})')

        if dist <= 0.5:
            self.goal_index += 1
            if self.goal_index >= len(self.goal_positions):
                self.goal_order = random.sample(range(len(self.goal_positions)), len(self.goal_positions))
                self.goal_index = 0
            self.goal_x, self.goal_y = self.goal_positions[self.goal_order[self.goal_index]]
            self.get_logger().info(f'Nova meta: ({self.goal_x}, {self.goal_y})')

def main(args=None):
    rclpy.init(args=args)
    navigator = InferenceModel()
    # rclpy.spin() executa callbacks e timers em background :contentReference[oaicite:2]{index=2}
    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        navigator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
