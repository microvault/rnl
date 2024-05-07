import numpy as np
import pytest
from matplotlib.patches import PathPatch

from microvault.environment.engine.world_generate import generate_maze
from microvault.environment.generate import Generator


@pytest.fixture
def generate_instance():
    return Generator(grid_lenght=10, random=1300)


def test_generate_maze(generate_instance):
    size = 10
    maze = generate_maze(size)
    assert maze.shape == (size, size)


def test_generate_shape_border(generate_instance):
    size = np.zeros((10, 10))
    map = generate_instance._map_border(size)
    assert map.shape == (12, 12)


def test_world_output_type(generate_instance):
    new_map_path, poly, seg = generate_instance.world()
    assert isinstance(new_map_path, PathPatch)
