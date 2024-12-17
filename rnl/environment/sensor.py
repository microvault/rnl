from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from rnl.configs.config import SensorConfig
from rnl.engine.collision import Collision


@dataclass
class SensorRobot:
    def __init__(self, sensor_config: SensorConfig):
        self.collision = Collision()
        self.max_range = sensor_config.max_range
        half_fov = np.radians(sensor_config.fov) / 2
        self.lidar_angle = np.linspace(-half_fov, half_fov, sensor_config.num_rays)

    def sensor(
        self, x: float, y: float, theta: float, segments: List
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform sensor measurements based on the robot's position and environment segments.
        """
        seg = self.collision.filter_segments(segments, x, y, 6)
        intersections = self.collision.lidar_intersection(
            x, y, theta, self.max_range, self.lidar_angle, seg
        )
        measurements = self.collision.lidar_measurement(
            x, y, theta, self.max_range, self.lidar_angle, seg
        )
        return intersections, measurements
