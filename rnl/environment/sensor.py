from dataclasses import dataclass
from typing import Tuple

import numpy as np

from rnl.configs.config import SensorConfig
from rnl.engine.filter import SpatialIndex
from rnl.engine.lidar import lidar_segments


@dataclass
class SensorRobot:
    def __init__(self, sensor_config: SensorConfig, map_segments):
        self.spatial_index = SpatialIndex(map_segments)
        self.max_range = sensor_config.max_range
        self.sensor_config = sensor_config
        half_fov = np.radians(self.sensor_config.fov) / 2
        self.lidar_angle = np.linspace(half_fov, -half_fov, self.sensor_config.num_rays)

    def sensor(
        self, x: float, y: float, theta: float, max_range: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform sensor measurements based on the robot's position and environment segments.
        """

        seg = self.spatial_index.filter_segments(x, y, self.max_range)
        if not seg:
            return np.array([]), np.full(
                self.sensor_config.num_rays, self.sensor_config.max_range
            )

        else:
            inter_all, mea_all = lidar_segments(
                x, y, theta, self.max_range, self.lidar_angle, seg
            )

        return np.array(inter_all), np.array(mea_all)

    def update_map(self, new_map_segments):
        self.map_segments = new_map_segments
        self.spatial_index = SpatialIndex(self.map_segments)
