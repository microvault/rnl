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
        self.sensor_config = sensor_config
        half_fov = np.radians(self.sensor_config.fov) / 2
        self.lidar_angle = np.linspace(half_fov, -half_fov, self.sensor_config.num_rays)

    def sensor(
        self, x: float, y: float, theta: float, segments: List, max_range: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform sensor measurements based on the robot's position and environment segments.
        """
        seg = self.collision.filter_segments(segments, x, y, max_range)
        intersections = np.array(
            self.collision.lidar_intersection(
                x, y, theta, self.max_range, self.lidar_angle, seg
            )
        )
        measurements = np.array(
            self.collision.lidar_measurement(
                x, y, theta, self.max_range, self.lidar_angle, seg
            )
        )
        return intersections, measurements

    def random_sensor(self, new_fov: float, new_lidar: int):
        """
        Perform sensor measurements based on the robot's position and environment segments.
        """
        half_fov = np.radians(new_fov) / 2
        self.lidar_angle = np.linspace(half_fov, -half_fov, new_lidar)
