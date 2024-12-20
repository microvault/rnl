import numpy as np
import pytest
from matplotlib.patches import PathPatch

from rnl.engine.world_generate import GenerateWorld
from rnl.environment.generate_world import Generator


@pytest.fixture
def generate_instance():
    return Generator()


@pytest.fixture
def generate_grid_instance():
    return GenerateWorld()


def test_generate_maze(generate_grid_instance):
    size = 10
    maze = generate_grid_instance.generate_maze(size)
    assert maze.shape == (size, size)


def test_generate_shape_border(generate_instance):
    size = np.zeros((10, 10))
    map = generate_instance._map_border(size)
    assert map.shape == (12, 12)


def test_world_output_type(generate_instance):
    new_map_path, exter, inter, seg = generate_instance.world(3)
    assert isinstance(new_map_path, PathPatch)
