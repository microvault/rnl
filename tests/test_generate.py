import numpy as np
import pytest
from matplotlib.patches import PathPatch

from microvault.environment.generate import Generator


@pytest.fixture
def generate_instance():
    return Generator(grid_lenght=10, random=1300)


def test_generate_maze(generate_instance):
    size = 10
    maze = generate_instance._generate_maze(size)
    assert maze.shape == (size, size)


def test_generate_map(generate_instance):
    size = 10
    random = 1300

    map = generate_instance._generate_map(size, random)
    assert map.shape == (size, size)


def test_generate_shape_border(generate_instance):
    size = np.zeros((10, 10))
    map = generate_instance._map_border(size)
    assert map.shape == (12, 12)


def test_world_output_type(generate_instance):
    patch = generate_instance.world()
    assert isinstance(patch, PathPatch)
