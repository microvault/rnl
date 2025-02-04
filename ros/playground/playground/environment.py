import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
import numpy as np

class SimpleSensorReader(Node):
    def __init__(self):
        super().__init__('environment')

        # QoS profile for sensors
        qos_profile = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)

        # Create subscriptions
        self.create_subscription(LaserScan, '/scan', self.laser_callback, qos_profile)
        self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile)

        # Initialize variables
        self.pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}

    def laser_callback(self, msg):
        # Converte a lista de distâncias para um array do NumPy
        ranges = np.array(msg.ranges)
        # Filtra os valores finitos
        finite_ranges = ranges[np.isfinite(ranges)]
        # Verifica se há valores finitos
        if finite_ranges.size > 0:
            # Encontra o maior valor finito
            max_range = np.max(finite_ranges)
            print(f"Maior valor do LiDAR (excluindo inf): {max_range}")
        else:
            print("Todos os valores são infinitos.")

    def odom_callback(self, msg):
        # Extract position
        self.pose['x'] = msg.pose.pose.position.x
        self.pose['y'] = msg.pose.pose.position.y
        print(f"X: {self.pose['x']:.3f}m")
        print(f"Y: {self.pose['y']:.3f}m")

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
