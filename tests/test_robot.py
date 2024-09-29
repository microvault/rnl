import pytest

from rnl.engine.collision import Collision
from rnl.environment.robot import Robot


@pytest.fixture
def robot_instance():
    collision = Collision()
    return Robot(
        collision=collision,
        time=100,
        min_radius=1.0,
        max_radius=3.0,
        max_grid=10,
        wheel_radius=0.3,
        wheel_base=0.3,
        fov=6.28,
        num_rays=10,
        max_range=6.0,
    )
