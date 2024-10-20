import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class LidarToGridMap(Node):
    def _init_(self, grid_size=100, resolution=0.5):
        super()._init_("lidar_to_grid_map")
        self.grid_size = grid_size
        self.resolution = resolution
        self.grid = np.zeros((grid_size, grid_size))
        self.origin_x = grid_size // 2
        self.origin_y = grid_size // 2
        self.subscription = self.create_subscription(
            LaserScan, "/scan", self.lidar_callback, 10
        )
        self.subscription

    def lidar_callback(self, msg):
        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges)

        plt.figure(figsize=(8, 8))
        plt.imshow(self.grid, cmap="gray", extent=[-25, 25, -25, 25], origin="lower")

        for r, angle in zip(ranges, angles):
            if msg.range_min <= r <= msg.range_max:
                x = int(self.origin_x + (r * np.cos(angle)) / self.resolution)
                y = int(self.origin_y + (r * np.sin(angle)) / self.resolution)
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    self.grid[y, x] = 1
                    plt.plot(
                        [
                            self.origin_x * self.resolution - 25,
                            x * self.resolution - 25,
                        ],
                        [
                            self.origin_y * self.resolution - 25,
                            y * self.resolution - 25,
                        ],
                        "g-",
                    )

        plt.title("Grid Map de Ocupação com Linhas do LIDAR")
        plt.xlabel("X (metros)")
        plt.ylabel("Y (metros)")
        plt.grid(True)
        plt.show()


def main(args=None):
    rclpy.init(args=args)
    node = LidarToGridMap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
