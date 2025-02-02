import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from math import pi
# from tf_transformations import euler_from_quaternion

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
        self.lidar_readings = [0.0] * 5  # 5 equidistant readings

    def laser_callback(self, msg):
        # Calculate indices for 5 equidistant readings in 270° FOV
        fov = 270  # degrees
        num_readings = len(msg.ranges)
        fov_indices = num_readings * (fov / 360.0)  # number of indices in FOV
        step = int(fov_indices / 4)  # divide FOV into 4 segments for 5 points
        start_idx = int((num_readings - fov_indices) / 2)  # center the FOV

        # Get 5 equidistant readings
        for i in range(5):
            idx = start_idx + (i * step)
            if idx < len(msg.ranges):
                self.lidar_readings[i] = msg.ranges[idx]

        # Print LiDAR readings
        print("\nLiDAR Readings (5 points across 270° FOV):")
        for i, reading in enumerate(self.lidar_readings):
            angle = -135 + (i * 67.5)  # Calculate approximate angle (-135° to +135°)
            print(f"Point {i+1} ({angle:>6.1f}°): {reading:.3f}m")

    def odom_callback(self, msg):
        # Extract position
        self.pose['x'] = msg.pose.pose.position.x
        self.pose['y'] = msg.pose.pose.position.y

        # Extract orientation (quaternion to euler)
        # orientation_q = msg.pose.pose.orientation
        # orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        # _, _, yaw = euler_from_quaternion(orientation_list)
        # self.pose['theta'] = yaw

        # Print odometry
        print("\nOdometry:")
        print(f"X: {self.pose['x']:.3f}m")
        print(f"Y: {self.pose['y']:.3f}m")
        # print(f"θ: {self.pose['theta']:.3f}rad ({(self.pose['theta'] * 180/pi):.1f}°)")

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
