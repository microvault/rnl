import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from ament_index_python.packages import get_package_share_directory
import os
import numpy as np

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
    """ Escala cada valor de features para [0,1] via min-max manual. """
    scaled = []
    for i, val in enumerate(features):
        mn = data_min[i]
        mx = data_max[i]
        if mx > mn:
            s = (val - mn) / (mx - mn)
        else:
            s = 0.0
        # Mantém em [0..1]
        s = max(0.0, min(s, 0.99999994))
        scaled.append(s)
    return scaled

class SimpleSensorReader(Node):
    def __init__(self):
        super().__init__("environment")

        # Monta caminho do modelo (exemplo usando os.path)
        self.position = 0
        self.action = 0
        self.last_states = np.zeros(10)

        pkg_dir = get_package_share_directory("playground")
        model_path = os.path.join(pkg_dir, "models", "model")
        self.get_logger().info(f"Carregando modelo de: {model_path}")
        self.model = PPO.load(model_path)
        qos_profile = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)

        # Subs / Pubs
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.laser_callback, qos_profile)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, qos_profile)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Cliente p/ reset e spawn (se quiser usar)
        self.reset_sim_client = self.create_client(Empty, "/gazebo/reset_simulation")
        self.set_model_state_client = self.create_client(SetModelState, "/gazebo/set_model_state")

        # Objetivo
        self.goal_x, self.goal_y = 7, 7

        # Armazena leituras e ação anterior
        self.lidar_ranges = [0.0]*5
        self.last_action_oh = [0, 0, 0]

        # Faixas p/ clamp e min-max
        # [LIDAR5, dist, alpha, actionOH3] => total 10
        self.data_min = [0.5, 0.5, 0.5, 0.5, 0.5,   1.0, 0.0, 0, 0, 0]
        self.data_max = [3.5, 3.5, 3.5, 3.5, 3.5,   9.0, 3.5, 1, 1, 1]

    def laser_callback(self, msg):
        ranges = msg.ranges
        n = len(ranges)
        if n < 5:
            return
        # 5 índices equidistantes
        indices = []
        step = (n - 1) / 4.0
        for i in range(5):
            idx = int(round(step*i))
            if idx >= n:
                idx = n - 1
            val = ranges[idx]
            if not math.isfinite(val):
                val = msg.range_max
            self.lidar_ranges[i] = val

    def odom_callback(self, msg):
        self.position = msg.pose.pose

    # def action_to_one_hot(self, action):
    #     action = int(action)
    #     oh = [0, 0, 0]
    #     if 0 <= action < 3:
    #         oh[action] = 1
    #     return oh

    def move_robot(self, action):
        if action == 0:
            # self.get_logger().info("Ação: Frente")
            vl, vr = 0.10, 0.0
        elif action == 1:
            # self.get_logger().info("Ação: Esquerda")
            vl, vr = 0.08, -0.08
        elif action == 2:
            # self.get_logger().info("Ação: Direita")
            vl, vr = 0.08, 0.08
        else:
            vl, vr = 0.0, 0.0

        twist = Twist()
        twist.linear.x = vl
        twist.angular.z = vr
        self.cmd_vel_pub.publish(twist)

    def update(self):


        # Prediz ação
        # stable-baselines3 converte de list -> numpy internamente
        self.get_logger().info(f"last_states: {self.last_states}")
        self.action, _ = self.model.predict(self.last_states)


        # Executa ação
        self.move_robot(self.action)

        # Extrai pose e yaw
        x = self.position.position.x
        y = self.position.position.y
        z = self.position.orientation.z
        w = self.position.orientation.w
        theta = 2.0 * math.atan2(z, w)

        # self.get_logger().info(f"X: {x}, Y: {y}, Theta: {theta}")

        # Dist e alpha
        dist = float(distance_to_goal(x, y, self.goal_x, self.goal_y))
        alpha = float(angle_to_goal(x , y , theta , self.goal_x, self.goal_y))
        # dist = math.sqrt((self.goal_x - x)**2 + (self.goal_y - y)**2)
        # angle_to_goal = math.atan2(self.goal_y - y, self.goal_x - x)
        # alpha = angle_to_goal - theta
        # alpha = math.atan2(math.sin(alpha), math.cos(alpha))
        # alpha = abs(alpha)  # se quiser só positivo

        # Clamps manuais
        clamped_lidar = [float(clamp(val, 0.5, 3.5)) for val in self.lidar_ranges]
        clamped_dist = float(clamp(dist, 1.0, 9.0))
        clamped_alpha = float(clamp(alpha, 0.0, 3.5))
        # clamped_action_oh = [clamp(a, 0.0, 1.0) for a in self.last_action_oh]

        action_one_hot = np.eye(3)[self.action]
        # self.get_logger().info(f"action_one_hot: {action_one_hot}")

        # Monta estado: [5 lidar, dist, alpha, 3 actionOH]
        # raw_state = clamped_lidar + action_one_hot + clamped_dist, clamped_alpha
        # self.get_logger().info(f"clamped_lidar: {clamped_lidar}")
        # self.get_logger().info(f"clamped_dist: {clamped_dist}")
        # self.get_logger().info(f"clamped_alpha: {clamped_alpha}")

        norm_lidar = (np.array(clamped_lidar, dtype=np.float32) - 0.5) / (3.5 - 0.5)
        norm_dist  = np.array(clamped_dist, dtype=np.float32) / 9.0
        norm_alpha = np.array(clamped_alpha, dtype=np.float32) / 3.5

        states = np.concatenate(
            (
                norm_lidar,
                np.array(action_one_hot, dtype=np.int16),
                np.array([norm_dist], dtype=np.float32),
                np.array([norm_alpha], dtype=np.float32),
            )
        )

        # Normaliza manualmente
        # scaled_state = min_max_scale(states, self.data_min, self.data_max)

        # self.get_logger().info(f"Estado normalizado: {scaled_state}")

        self.last_states = states

        # self.get_logger().info(f"Action: {int(self.action)}")


        # Exemplo: se dist <= 1, reset e respawn
        if dist <= 1.0:
            self.reset_sim_and_spawn()


    def reset_sim_and_spawn(self):
        self.get_logger().info("Dist <= 1m, resetando e voltando pra (4.5, 4.5)...")

        # Espera serviços
        while not self.reset_sim_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Esperando /gazebo/reset_simulation...")
        while not self.set_model_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Esperando /gazebo/set_model_state...")

        # Reseta
        reset_req = Empty.Request()
        self.reset_sim_client.call_async(reset_req)

        # Reposiciona
        model_state_msg = ModelState()
        model_state_msg.model_name = "turtlebot3_burger"
        model_state_msg.pose.position.x = 4.5
        model_state_msg.pose.position.y = 4.5
        model_state_msg.pose.orientation.w = 1.0
        model_state_msg.reference_frame = "world"

        spawn_req = SetModelState.Request()
        spawn_req.model_state = model_state_msg
        self.set_model_state_client.call_async(spawn_req)

def main(args=None):
    rclpy.init(args=args)
    navigator = SimpleSensorReader()
    try:
        rclpy.spin_once(navigator)
        while(rclpy.ok):

            rclpy.spin_once(navigator)
            navigator.update()
    except KeyboardInterrupt:
        pass
    finally:
        navigator.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
