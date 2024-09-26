from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import pymunk
from pymunk import Vec2d

@dataclass
class Robot:
    """
    A class representing a robot with physical and sensor properties.
    """
    collision: 'Collision'
    robot_radius: float = 0.033
    wheel_base: float = 0.16
    fov: float = 4 * np.pi
    num_rays: int = 20
    max_range: float = 6.0
    min_range: float = 1.0
    mass: float = 1.0
    inertia: float = 0.3

    def __post_init__(self):
        """
        Initialize additional attributes after dataclass initialization.
        """
        self.lidar_angle = np.linspace(0, self.fov, self.num_rays)
        self.body = pymunk.Body(self.mass, self.inertia)

    @staticmethod
    def create_space() -> pymunk.Space:
        """
        Create and return a new pymunk space with no gravity.
        """
        space = pymunk.Space()
        space.gravity = (0.0, 0.0)
        return space

    def create_robot(self, space: pymunk.Space,friction: float = 0.4, damping: float = 0.1) -> pymunk.Body:
        """
        Create and add the robot to the given pymunk space.
        """
        body = pymunk.Body(self.mass, self.inertia)
        body.position = (0, 0)
        shape = pymunk.Circle(body, self.robot_radius)
        shape.friction = 0.4
        shape.damping = 0.1
        space.add(body, shape)
        return body

    def move_robot(self, space: pymunk.Space, robot_body: pymunk.Body, v_linear: float, v_angular: float) -> None:
        """
        Move the robot in the space with given linear and angular velocities.
        """
        direction = Vec2d(np.cos(robot_body.angle), np.sin(robot_body.angle))
        robot_body.velocity = v_linear * direction
        robot_body.angular_velocity = v_angular
        space.step(1 / 60)

    def reset_robot(self, robot_body: pymunk.Body, x: float, y: float) -> None:
        """
        Reset the robot's position and velocity.
        """
        robot_body.position = (x, y)
        robot_body.angle = 0
        robot_body.velocity = (0, 0)
        robot_body.angular_velocity = 0

    def sensor(self, x: float, y: float, segments: List) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform sensor measurements based on the robot's position and environment segments.
        """
        seg = self.collision.filter_segments(segments, x, y, 6)
        intersections = self.collision.lidar_intersection(
            x, y, self.max_range, self.lidar_angle, seg
        )
        measurements = self.collision.lidar_measurement(
            x, y, self.max_range, self.lidar_angle, seg
        )
        return intersections, measurements
