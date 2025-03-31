import math
import random

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


def euler_from_quaternion(q):
    """Converte quaternion para yaw (simplificado)"""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw


class AMCL(Node):
    def __init__(self):
        super().__init__("localization")

        # Carrega mapa. Supondo que você tenha salvo em .npy
        self.map_data = np.load("/turtlebot3_ws/src/playground/occupancy_map.npy")

        # Resolução do mapa (igual ao do seu mapper)
        self.resolution = 0.1
        self.origin_x = self.map_data.shape[1] // 2
        self.origin_y = self.map_data.shape[0] // 2

        # Número de partículas
        self.num_particles = 200
        self.particles = self.init_particles()

        # Guardar última odometria e timestamp
        self.last_odom = None
        self.last_odom_stamp = None

        # Subscrever tópicos
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )

        # Publisher da pose estimada
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/amcl_pose", 10
        )

        self.get_logger().info("AMCL node iniciado")

    def init_particles(self):
        """Inicializa partículas aleatórias dentro do mapa"""
        particles = []
        h, w = self.map_data.shape
        for _ in range(self.num_particles):
            # Gera x, y, yaw aleatórios dentro do grid
            x_cell = random.randint(0, w - 1)
            y_cell = random.randint(0, h - 1)
            # Só aceita se o local não for obstáculo
            while self.map_data[y_cell, x_cell] > 100:
                x_cell = random.randint(0, w - 1)
                y_cell = random.randint(0, h - 1)

            # Converte célula pra coordenada em metros
            x = (x_cell - self.origin_x) * self.resolution
            y = (y_cell - self.origin_y) * self.resolution
            yaw = random.uniform(-math.pi, math.pi)
            weight = 1.0 / self.num_particles
            particles.append([x, y, yaw, weight])
        return particles

    def odom_callback(self, msg):
        # Guardar última odometria pra usar na hora do scan_callback
        self.last_odom = msg
        self.last_odom_stamp = msg.header.stamp

    def scan_callback(self, scan_msg):
        if not self.last_odom:
            return

        # 1) Predição (motion model)
        self.predict_particles(self.last_odom)

        # 2) Correção (observation model)
        self.update_weights(scan_msg)

        # 3) Ressampling
        self.resample()

        # Publica pose média
        self.publish_estimated_pose()

    def predict_particles(self, odom_msg):
        """
        Usa a diferença de odometria para deslocar partículas (modelo simplificado).
        A gente não tá guardando a odometria anterior, então isso é minimalista.
        Ideal: armazenar e calcular delta real.
        """
        # V pego do Twist, Yaw do quaternion (mas é bem simplificado)
        linear_vel = odom_msg.twist.twist.linear.x
        angular_vel = odom_msg.twist.twist.angular.z
        dt = 0.1  # Só um chute. Na prática use tempo real entre medições.

        for i, p in enumerate(self.particles):
            x, y, yaw, w = p
            # Modelo de movimento básico: x += v*cos(yaw)*dt
            # Adicione ruído
            x += linear_vel * math.cos(yaw) * dt + random.gauss(0, 0.01)
            y += linear_vel * math.sin(yaw) * dt + random.gauss(0, 0.01)
            yaw += angular_vel * dt + random.gauss(0, 0.005)
            self.particles[i] = [x, y, yaw, w]

    def update_weights(self, scan_msg):
        """
        Observation model bem simples: pra cada partícula,
        pega uns poucos raios do LIDAR e compara com o mapa.
        """
        angles = np.arange(
            scan_msg.angle_min, scan_msg.angle_max, scan_msg.angle_increment
        )
        ranges = np.array(scan_msg.ranges)

        for i, p in enumerate(self.particles):
            x, y, yaw, w = p
            score = 0.0
            n_rays = 20  # pra não usar todas as leituras, diminui tempo de cálculo
            step = max(1, len(ranges) // n_rays)

            for idx in range(0, len(ranges), step):
                r = ranges[idx]
                # Ignora leituras "inválidas"
                if scan_msg.range_min < r < scan_msg.range_max:
                    angle = angles[idx] + yaw
                    mapx = x + r * math.cos(angle)
                    mapy = y + r * math.sin(angle)

                    # Converte pra células
                    cell_x = int(self.origin_x + mapx / self.resolution)
                    cell_y = int(self.origin_y + mapy / self.resolution)

                    # Se dentro do mapa e não é obstáculo, ganha pontos
                    if (
                        0 <= cell_x < self.map_data.shape[1]
                        and 0 <= cell_y < self.map_data.shape[0]
                    ):
                        # Se a célula é escura (livre) no seu map_data
                        if self.map_data[cell_y, cell_x] < 50:
                            score += 1.0

            # Ajusta peso: soma +1 pra evitar zero e normalizar depois
            w_new = score + 1.0
            self.particles[i][3] = w_new

        # Normalizar pesos
        total_weight = sum([p[3] for p in self.particles])
        for i in range(len(self.particles)):
            self.particles[i][3] /= total_weight + 1e-9

    def resample(self):
        """
        Ressampling com método sistemático, por exemplo.
        """
        weights = [p[3] for p in self.particles]
        new_particles = []

        step = 1.0 / self.num_particles
        r = random.random() * step
        c = weights[0]
        i = 0

        for _ in range(self.num_particles):
            U = r + _ * step
            while U > c:
                i += 1
                c += weights[i]
            # Clona partícula i (sem mexer no peso - depois normaliza)
            new_particles.append(self.particles[i][:])
        # Normaliza pesos novamente
        for i in range(self.num_particles):
            new_particles[i][3] = 1.0 / self.num_particles

        self.particles = new_particles

    def publish_estimated_pose(self):
        """
        Faz a média das partículas e publica no tópico /amcl_pose
        """
        x_mean = np.mean([p[0] for p in self.particles])
        y_mean = np.mean([p[1] for p in self.particles])
        yaw_mean = np.mean(
            [p[2] for p in self.particles]
        )  # jeitão simples (nem sempre correto)

        print(f"Estimativa: x={x_mean:.2f} y={y_mean:.2f} yaw={yaw_mean:.2f}")

        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = x_mean
        msg.pose.pose.position.y = y_mean

        # Converte yaw_mean pra quaternion
        half_yaw = yaw_mean / 2.0
        msg.pose.pose.orientation.z = math.sin(half_yaw)
        msg.pose.pose.orientation.w = math.cos(half_yaw)
        # Covariância bem básica
        for i in range(36):
            msg.pose.covariance[i] = 0.0
        # Exemplo: assume alguma incerteza em x,y
        msg.pose.covariance[0] = 0.1
        msg.pose.covariance[7] = 0.1
        msg.pose.covariance[35] = 0.2

        self.pose_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AMCL()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
