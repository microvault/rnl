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

from stable_baselines3 import PPO

def clamp(value, vmin, vmax):
    return max(vmin, min(value, vmax))

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
        s = max(0.0, min(s, 1.0))
        scaled.append(s)
    return scaled

class SimpleSensorReader(Node):
    def __init__(self):
        super().__init__("environment")

        # Monta caminho do modelo (exemplo usando os.path)
        self.position = 0

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
        self.goal_x = 7.0
        self.goal_y = 2.0

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

    def action_to_one_hot(self, action):
        action = int(action)
        oh = [0, 0, 0]
        if 0 <= action < 3:
            oh[action] = 1
        return oh

    def do_action(self, action):
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
        # Extrai pose e yaw
        x = self.position.position.x
        y = self.position.position.y
        z = self.position.orientation.z
        w = self.position.orientation.w
        theta = 2.0 * math.atan2(z, w)

        # Dist e alpha
        dist = math.sqrt((self.goal_x - x)**2 + (self.goal_y - y)**2)
        angle_to_goal = math.atan2(self.goal_y - y, self.goal_x - x)
        alpha = angle_to_goal - theta
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))
        alpha = abs(alpha)  # se quiser só positivo

        # Clamps manuais
        clamped_lidar = [clamp(val, 0.5, 3.5) for val in self.lidar_ranges]
        clamped_dist = clamp(dist, 1.0, 9.0)
        clamped_alpha = clamp(alpha, 0.0, 3.5)
        clamped_action_oh = [clamp(a, 0.0, 1.0) for a in self.last_action_oh]

        # Monta estado: [5 lidar, dist, alpha, 3 actionOH]
        raw_state = clamped_lidar + [clamped_dist, clamped_alpha] + clamped_action_oh

        # Normaliza manualmente
        scaled_state = min_max_scale(raw_state, self.data_min, self.data_max)

        self.get_logger().info(f"Estado normalizado: {scaled_state}")

        # Prediz ação
        # stable-baselines3 converte de list -> numpy internamente
        action, _ = self.model.predict([scaled_state], deterministic=True)

        # Converte pra one-hot e guarda
        self.last_action_oh = self.action_to_one_hot(action)

        # Executa ação
        self.do_action(action)

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
