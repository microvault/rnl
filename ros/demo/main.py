#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class LaserProcessor(Node):
    def __init__(self):
        super().__init__("laser_processor")

        # Create subscribers and publishers
        self.scan_sub = self.create_subscription(
            LaserScan, "scan", self.scan_callback, 10
        )

        # Counter for marker IDs
        self.marker_id = 0

    def scan_callback(self, msg):
        # Calculate angles for 5 equidistant measurements in 270 degrees FOV
        angle_min = -135 * (np.pi / 180)  # -135 degrees in radians
        angle_max = 135 * (np.pi / 180)  # 135 degrees in radians
        angles = np.linspace(angle_min, angle_max, 5)

        # Get corresponding indices in the original scan
        indices = ((angles - msg.angle_min) / msg.angle_increment).astype(int)
        print(indices)


def main(args=None):
    rclpy.init(args=args)
    node = LaserProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
