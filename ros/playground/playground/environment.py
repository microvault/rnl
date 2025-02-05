import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
import numpy as np
import math
from stable_baselines3 import PPO
from sklearn.preprocessing import MinMaxScaler

class SimpleSensorReader(Node):
    def __init__(self):
        super().__init__('environment')

        self.model = PPO.load("turtlebot3_ws/src/playground/models/model")
        qos_profile = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)

        self.goal_x = 0.0
        self.goal_y = 0.0

        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, qos_profile)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile)

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', qos_profile)

        self.lidar_ranges = [0.0]*5

        self.last_action_oh = [0, 0, 0]

        # 1) LiDAR (5 leituras) => 0.5..3.5
        self.LIDAR_MIN, self.LIDAR_MAX = 0.5, 3.5
        # 2) Distância => 1..9
        self.DIST_MIN, self.DIST_MAX = 1.0, 9.0
        # 3) alpha => 0..3.5
        self.ALPHA_MIN, self.ALPHA_MAX = 0.0, 3.5
        # 4) Ação one-hot => 0..1

        # MinMaxScaler c/ colunas na ordem:
        #   [ Lidar1, Lidar2, Lidar3, Lidar4, Lidar5, Dist, Alpha, ActionOH(3) ]
        # total 5 + 1 + 1 + 3 = 10.
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        data_min = [0.5, 0.5, 0.5, 0.5, 0.5,   1.0, 0.0, 0, 0, 0]
        data_max = [3.5, 3.5, 3.5, 3.5, 3.5,   9.0, 3.5, 1, 1, 1]
        self.scaler.fit([data_min, data_max])

    def one_hot_action(self, action):
        one_hot = [0, 0, 0]
        if 0 <= action < 3:
            one_hot[action] = 1
        return one_hot

    def laser_callback(self, msg):
        # Pegamos 5 leituras no FOV de 270. Exemplo básico pegando 5 índices do /scan inteiro.
        ranges = np.array(msg.ranges)
        indices = np.linspace(0, len(ranges)-1, 5, dtype=int)
        selected_ranges = ranges[indices]
        # Tratar inf e nan
        selected_ranges = np.nan_to_num(selected_ranges, nan=msg.range_max, posinf=msg.range_max, neginf=0.0)
        self.lidar_ranges = list(selected_ranges)

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Converte quaternion -> yaw
        z = msg.pose.pose.orientation.z
        w = msg.pose.pose.orientation.w
        theta = 2.0 * math.atan2(z, w)

        # Dist e alpha
        dist_to_goal = self.calc_distance_to_goal(x, y)
        alpha = self.calc_angle_to_goal(x, y, theta)

        # ------- CLAMP DOS ESTADOS -------
        # LiDAR
        clamped_lidar = [max(self.LIDAR_MIN, min(l, self.LIDAR_MAX)) for l in self.lidar_ranges]
        # Dist
        clamped_dist = max(self.DIST_MIN, min(dist_to_goal, self.DIST_MAX))
        # Alpha
        clamped_alpha = max(self.ALPHA_MIN, min(alpha, self.ALPHA_MAX))
        # Última ação (one-hot). Em tese já é [0,1], mas vamos garantir c/ clamp tb.
        clamped_action_oh = [max(0, min(a, 1)) for a in self.last_action_oh]

        # Monta estado bruto: [5 LiDAR, ultima_acao_oh(3), dist, alpha]
        raw_state = clamped_lidar + [clamped_dist, clamped_alpha] + clamped_action_oh
        raw_state = np.array(raw_state).reshape(1, -1)

        # Normaliza
        norm_state = self.scaler.transform(raw_state).flatten()

        # Prediz ação
        action, _ = self.model.predict(norm_state, deterministic=True)

        # Gera one-hot dessa ação
        action_oh = self.action_to_one_hot(action)
        # Atualiza pra usar no próximo loop
        self.last_action_oh = action_oh

        self.do_action(action)

    def calc_distance_to_goal(self, x, y):
        return math.sqrt((self.goal_x - x)**2 + (self.goal_y - y)**2)

    def calc_angle_to_goal(self, x, y, theta):
        angle_to_goal = math.atan2(self.goal_y - y, self.goal_x - x)
        alpha = angle_to_goal - theta
        return math.atan2(math.sin(alpha), math.cos(alpha))

        def do_action(self, action):
            # Se 0 -> frente
            if action == 0:
                self.get_logger().info("Ação: Frente")
                vl = 0.10
                vr = 0.00
            # Se 1 -> virar pra esquerda
            elif action == 1:
                self.get_logger().info("Ação: Esquerda")
                vl = 0.08
                vr = -0.08
            # Se 2 -> virar pra direita
            elif action == 2:
                self.get_logger().info("Ação: Direita")
                vl = 0.08
                vr = 0.08
            else:
                vl = 0.0
                vr = 0.0

            twist = Twist()
            twist.linear.x = vl
            twist.angular.z = vr
            self.cmd_vel_pub.publish(twist)

    def reset_sim_and_spawn(self):
        self.get_logger().info("Dist <= 1m, resetando e voltando pra (4.5, 4.5)...")

        # Espera serviços
        while not self.reset_sim_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Esperando /gazebo/reset_simulation...')
        while not self.set_model_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Esperando /gazebo/set_model_state...')

        # Reseta
        reset_req = Empty.Request()
        self.reset_sim_client.call_async(reset_req)

        # Spawna
        model_state_msg = ModelState()
        model_state_msg.model_name = 'turtlebot3_burger'
        model_state_msg.pose.position.x = 4.5
        model_state_msg.pose.position.y = 4.5
        model_state_msg.pose.orientation.w = 1.0
        model_state_msg.reference_frame = 'world'

        spawn_req = SetModelState.Request()
        spawn_req.model_state = model_state_msg
        self.set_model_state_client.call_async(spawn_req)

def main(args=None):
    rclpy.init(args=args)
    node = SimpleSensorReader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
