import math

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class LidarOdomGridMapper(Node):
    def __init__(self, grid_size=300, resolution=0.1):
        """
        grid_size: tamanho da matriz (grid_size x grid_size)
        resolution: metros por célula
        """
        super().__init__("mapping")

        # Parâmetros do grid
        self.grid_size = grid_size
        self.resolution = resolution
        self.grid = np.zeros((grid_size, grid_size), dtype=np.uint8)

        # Origem do grid no “meio” da matriz: (0,0) do mundo
        self.origin_x = grid_size // 2
        self.origin_y = grid_size // 2

        # Pose atual do robô
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        # Matplotlib em modo interativo
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))

        # Subscreve o tópico /scan (LIDAR)
        self.laser_sub = self.create_subscription(
            LaserScan, "/scan", self.lidar_callback, 10
        )

        # Subscreve o tópico /odom (pose do robô)
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10
        )

    def odom_callback(self, msg):
        """Recebe a pose do robô (x, y, yaw)."""
        # Pega x e y
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        # Converte quaternion -> yaw
        q = msg.pose.pose.orientation
        # Equivalente a tf.transformations.euler_from_quaternion, mas manualmente:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

    def lidar_callback(self, msg):
        """Processa leitura LIDAR, projeta cada raio no mapa global e atualiza grid."""
        # Constrói array de ângulos
        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges)

        for r, angle in zip(ranges, angles):
            if msg.range_min < r < msg.range_max:
                # Converte pra coordenadas globais, somando a pose do robô
                # Robô está em (self.robot_x, self.robot_y) e yaw = self.robot_yaw
                global_x = self.robot_x + r * math.cos(angle + self.robot_yaw)
                global_y = self.robot_y + r * math.sin(angle + self.robot_yaw)

                # Converte global (x,y) -> índices da matriz (x_cell, y_cell)
                x_cell = int(self.origin_x + global_x / self.resolution)
                y_cell = int(self.origin_y + global_y / self.resolution)

                # Atualiza o grid se estiver dentro dos limites
                if 0 <= x_cell < self.grid_size and 0 <= y_cell < self.grid_size:
                    self.grid[y_cell, x_cell] = 255

        # Plot do grid (interativo)
        self.ax.clear()
        self.ax.imshow(
            self.grid,
            cmap="gray",
            origin="lower",
            extent=[
                -self.origin_x * self.resolution,
                (self.grid_size - self.origin_x) * self.resolution,
                -self.origin_y * self.resolution,
                (self.grid_size - self.origin_y) * self.resolution,
            ],
        )
        self.ax.set_title("Mapeamento 2D (usando /odom como ref.)")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.grid(True)

        plt.draw()
        plt.pause(0.001)

    def destroy_node(self):
        # Salva o array em .npy
        np.save("src/playground/occupancy_map.npy", self.grid)
        self.get_logger().info("Mapa salvo em occupancy_map.npy")

        # Salva também em PNG usando o matplotlib
        plt.imsave(
            "./src/playground/occupancy_map.png", self.grid, cmap="gray", origin="lower"
        )
        self.get_logger().info("Mapa salvo em occupancy_map.png")

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LidarOdomGridMapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
