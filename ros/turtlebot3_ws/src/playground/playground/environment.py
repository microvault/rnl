# import math
# import os
# import random

# import numpy as np
# import rclpy
# from ament_index_python.packages import get_package_share_directory
# from geometry_msgs.msg import Twist
# from nav_msgs.msg import Odometry
# from rclpy.node import Node
# from rclpy.qos import QoSProfile, QoSReliabilityPolicy
# from sensor_msgs.msg import LaserScan
# from stable_baselines3 import PPO
# from geometry_msgs.msg import Pose
# from geometry_msgs.msg import PoseWithCovarianceStamped


# def clamp(value, vmin, vmax):
#     return max(vmin, min(value, vmax))


# def distance_to_goal(x: float, y: float, goal_x: float, goal_y: float) -> float:
#     dist = np.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)
#     if dist >= 9:
#         return 9
#     else:
#         return dist


# def angle_to_goal(
#     x: float, y: float, theta: float, goal_x: float, goal_y: float
# ) -> float:
#     o_t = np.array([np.cos(theta), np.sin(theta)])
#     g_t = np.array([goal_x - x, goal_y - y])

#     cross_product = np.cross(o_t, g_t)
#     dot_product = np.dot(o_t, g_t)

#     alpha = np.abs(np.arctan2(np.linalg.norm(cross_product), dot_product))

#     return alpha


# def min_max_scale(features, data_min, data_max):
#     scaled = []
#     for i, val in enumerate(features):
#         mn = data_min[i]
#         mx = data_max[i]
#         if mx > mn:
#             s = (val - mn) / (mx - mn)
#         else:
#             s = 0.0
#         s = max(0.0, min(s, 0.99999994))
#         scaled.append(s)
#     return scaled


# class InferenceModel(Node):
#     def __init__(self):
#         super().__init__("environment")
#         self.position = 0
#         self.action = 0
#         self.last_states = np.zeros(10)

#         # Yaw inicial em radianos
#         yaw = 4.0
#         # Converte yaw -> quaternion
#         self.z = math.sin(yaw / 2.0)
#         self.w = math.cos(yaw / 2.0)

#         self.position = Pose()
#         self.position.position.x = 1.07
#         self.position.position.y = 1.07
#         self.position.orientation.z = self.z
#         self.position.orientation.w = self.w

#         # Publicador para /initialpose (caso use AMCL/SLAM)
#         self.initial_pose_pub = self.create_publisher(
#             PoseWithCovarianceStamped,
#             "/initialpose",
#             10
#         )

#         self.publish_initial_pose()

#         # self.goal_positions = [(2, 2), (7, 2), (2, 7), (7, 7)] [0.35, 0.35], [0.35, 1.8], [1.8, 0.35], [1.8, 1.8]
#         # self.goal_positions = [(0.35, 0.35), (0.35, 1.8), (1.8, 0.35), (1.8, 1.8)]
#         self.goal_positions = [(0.35, 0.35)]
#         self.goal_order = random.sample(
#             range(len(self.goal_positions)), len(self.goal_positions)
#         )
#         self.goal_index = 0
#         self.goal_x, self.goal_y = self.goal_positions[self.goal_order[self.goal_index]]

#         self.lidar_ranges = [0.0] * 5
#         self.last_action_oh = [0, 0, 0]

#         self.data_min = [0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 0.0, 0, 0, 0]
#         self.data_max = [3.5, 3.5, 3.5, 3.5, 3.5, 9.0, 3.5, 1, 1, 1]

#         pkg_dir = get_package_share_directory("playground")
#         model_path = os.path.join(pkg_dir, "models", "model")
#         self.get_logger().info(f"Model path: {model_path}")
#         self.model = PPO.load(model_path)

#         qos_profile = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)

#         self.scan_sub = self.create_subscription(
#             LaserScan, "/scan", self.laser_callback, qos_profile
#         )
#         self.odom_sub = self.create_subscription(
#             Odometry, "/odom", self.odom_callback, qos_profile
#         )
#         self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

#     def publish_initial_pose(self):
#         msg = PoseWithCovarianceStamped()
#         msg.header.stamp = self.get_clock().now().to_msg()
#         msg.header.frame_id = "map"
#         # Pose inicial desejada
#         msg.pose.pose.position.x = 1.07
#         msg.pose.pose.position.y = 1.07
#         msg.pose.pose.orientation.z = self.z
#         msg.pose.pose.orientation.w = self.w
#         # Covariância zerada (ou mínima)
#         msg.pose.covariance = [0.0] * 36
#         self.initial_pose_pub.publish(msg)
#         self.get_logger().info("Publicada pose inicial em /initialpose")

#     def laser_callback(self, msg):
#         ranges = msg.ranges
#         n = len(ranges)
#         if n < 5:
#             return
#         step = (n - 1) / 4.0
#         for i in range(5):
#             idx = int(round(step * i))
#             if idx >= n:
#                 idx = n - 1
#             val = ranges[idx]
#             if not math.isfinite(val):
#                 val = msg.range_max
#             self.lidar_ranges[i] = val

#     def odom_callback(self, msg):
#         # Se quiser "forçar" usar sempre a pose inicial no seu nó (mas não muda /odom real):
#         if not self.initialized:
#             # Sobrescreve a pose recebida apenas na 1ª vez
#             msg.pose.pose.position.x = 1.07
#             msg.pose.pose.position.y = 1.07
#             msg.pose.pose.orientation.z = -0.7059230410102901
#             msg.pose.pose.orientation.w = 0.7082885430181574
#             self.initialized = True

#         self.position = msg.pose.pose
#         self.get_logger().info(f"States: {msg.pose.pose}")

#     def move_robot(self, action):
#         if action == 0:
#             vl, vr = 0.10, 0.0
#         elif action == 1:
#             vl, vr = 0.08, -0.36
#         elif action == 2:
#             vl, vr = 0.08, 0.36
#         else:
#             vl, vr = 0.0, 0.0

#         twist = Twist()
#         twist.linear.x = vl
#         twist.angular.z = vr
#         self.cmd_vel_pub.publish(twist)

#     def update(self):
#         self.get_logger().info(f"States: {self.last_states}")

#         self.action, _ = self.model.predict(self.last_states)

#         self.move_robot(self.action)

#         x = self.position.position.x
#         y = self.position.position.y
#         z = self.position.orientation.z
#         w = self.position.orientation.w
#         theta = 2.0 * math.atan2(z, w)

#         dist = float(distance_to_goal(x, y, self.goal_x, self.goal_y))
#         alpha = float(angle_to_goal(x, y, theta, self.goal_x, self.goal_y))

#         clamped_lidar = [float(clamp(val, 0.5, 3.5)) for val in self.lidar_ranges]
#         clamped_dist = float(clamp(dist, 1.0, 9.0))
#         clamped_alpha = float(clamp(alpha, 0.0, 3.5))

#         action_one_hot = np.eye(3)[self.action]

#         norm_lidar = (np.array(clamped_lidar, dtype=np.float32) - 0.5) / (3.5 - 0.5)
#         norm_dist = np.array(clamped_dist, dtype=np.float32) / 9.0
#         norm_alpha = np.array(clamped_alpha, dtype=np.float32) / 3.5

#         states = np.concatenate(
#             (
#                 norm_lidar,
#                 np.array(action_one_hot, dtype=np.int16),
#                 np.array([norm_dist], dtype=np.float32),
#                 np.array([norm_alpha], dtype=np.float32),
#             )
#         )

#         self.last_states = states

#         if dist <= 1.2:
#             self.goal_index += 1
#             if self.goal_index >= len(self.goal_order):
#                 self.goal_order = random.sample(
#                     range(len(self.goal_positions)), len(self.goal_positions)
#                 )
#                 self.goal_index = 0
#             self.goal_x, self.goal_y = self.goal_positions[
#                 self.goal_order[self.goal_index]
#             ]
#             self.get_logger().info(f"New goal: ({self.goal_x}, {self.goal_y})")


# def main(args=None):
#     rclpy.init(args=args)
#     navigator = InferenceModel()
#     try:
#         rclpy.spin_once(navigator)
#         while rclpy.ok:

#             rclpy.spin_once(navigator)
#             navigator.update()
#     except KeyboardInterrupt:
#         pass
#     finally:
#         navigator.destroy_node()
#         rclpy.shutdown()


# if __name__ == "__main__":
#     main()


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
from geometry_msgs.msg import Pose

def clamp(value, vmin, vmax):
    return max(vmin, min(value, vmax))

def distance_to_goal(x: float, y: float, goal_x: float, goal_y: float) -> float:
    dist = np.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)
    if dist >= 9:
        return 9
    else:
        return dist

def angle_to_goal(x: float, y: float, theta: float, goal_x: float, goal_y: float) -> float:
    o_t = np.array([np.cos(theta), np.sin(theta)])
    g_t = np.array([goal_x - x, goal_y - y])
    cross_product = np.cross(o_t, g_t)
    dot_product = np.dot(o_t, g_t)
    alpha = np.abs(np.arctan2(np.linalg.norm(cross_product), dot_product))
    return alpha

class InferenceModel(Node):
    def __init__(self):
        super().__init__("environment")

        # Pose "inicial desejada"
        self.desired_x = 1.07
        self.desired_y = 1.07
        self.desired_yaw = 4.0

        # Offsets que vamos calcular quando receber a primeira odometria
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_yaw = 0.0
        self.offset_computed = False

        # Aqui guardamos a "odometria deslocada" que seu código efetivamente usa
        self.position = Pose()

        self.action = 0
        self.last_states = np.zeros(10)

        # Metas de navegação
        self.goal_positions = [(0.35, 0.35), (0.35, 1.8), (1.8, 0.35), (1.8, 1.8)]
        self.goal_order = random.sample(range(len(self.goal_positions)), len(self.goal_positions))
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
        # Lê a odometria "real" do robô
        real_x = msg.pose.pose.position.x
        real_y = msg.pose.pose.position.y
        real_z = msg.pose.pose.orientation.z
        real_w = msg.pose.pose.orientation.w

        # Converte a orientação em yaw (entre -pi e +pi)
        real_yaw = 2.0 * math.atan2(real_z, real_w)

        # Se for a primeira vez, calculamos a diferença entre
        # a odometria real e a pose desejada.
        # Assim, nas próximas mensagens aplicamos esse offset
        # para que "visualmente" ele inicie em (1.07, 1.07, yaw=4)
        if not self.offset_computed:
            # offset = (desejado - real)
            self.offset_x = self.desired_x - real_x
            self.offset_y = self.desired_y - real_y
            self.offset_yaw = self.desired_yaw - real_yaw
            self.offset_computed = True
            # self.get_logger().info("Offset calculado: "
            #                        f"offset_x={self.offset_x}, "
            #                        f"offset_y={self.offset_y}, "
            #                        f"offset_yaw={self.offset_yaw}")

        # Aplica offset pra obter a "odometria deslocada"
        shifted_x = real_x + self.offset_x
        shifted_y = real_y + self.offset_y
        shifted_yaw = real_yaw + self.offset_yaw

        # Converte yaw deslocado de volta pra quaternion
        s_z = math.sin(shifted_yaw / 2.0)
        s_w = math.cos(shifted_yaw / 2.0)

        # Atualiza self.position com a pose "deslocada"
        self.position.position.x = shifted_x
        self.position.position.y = shifted_y
        self.position.orientation.z = s_z
        self.position.orientation.w = s_w

        # self.get_logger().info(
        #     f"Real odom=({real_x:.2f}, {real_y:.2f}, {real_yaw:.2f}) | "
        #     f"Shifted=({shifted_x:.2f}, {shifted_y:.2f}, {shifted_yaw:.2f})"
        # )

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

    def update(self):
        # self.get_logger().info(f"States: {self.last_states}")

        self.action, _ = self.model.predict(self.last_states)
        self.move_robot(self.action)

        x = self.position.position.x
        y = self.position.position.y
        z = self.position.orientation.z
        w = self.position.orientation.w
        theta = 2.0 * math.atan2(z, w)

        dist = float(distance_to_goal(x, y, self.goal_x, self.goal_y))
        alpha = float(angle_to_goal(x, y, theta, self.goal_x, self.goal_y))

        # Lidar clamp
        clamped_lidar = [float(clamp(val, 0.5, 3.5)) for val in self.lidar_ranges]
        clamped_dist = float(clamp(dist, 1.0, 9.0))
        clamped_alpha = float(clamp(alpha, 0.0, 3.5))

        action_one_hot = np.eye(3)[self.action]

        # Normalização
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

        if dist <= 0.5:
            self.goal_index += 1
            if self.goal_index >= len(self.goal_order):
                self.goal_order = random.sample(range(len(self.goal_positions)), len(self.goal_positions))
                self.goal_index = 0
            self.goal_x, self.goal_y = self.goal_positions[self.goal_order[self.goal_index]]
            self.get_logger().info(f"New goal: ({self.goal_x}, {self.goal_y})")


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
