import matplotlib.pyplot as plt
import pytest

from rnl.engine.map2d import Map2D


def test_folder_is_none():
    mapa = Map2D(folder=None, name="nome_do_mapa")
    assert mapa.path is None


def test_name_is_none():
    mapa = Map2D(folder="./data/map", name=None)
    assert mapa.path == "./data/map"


@pytest.fixture
def continuous_instance():
    return Map2D(folder="./data/map", name="map")


def test_grid_map(continuous_instance):
    grid_map = continuous_instance._grid_map()
    assert grid_map.any()


def test_occupancy(continuous_instance):
    grid_map = continuous_instance.occupancy_grid()
    assert grid_map.any()


def test_initial_environment2d(continuous_instance):
    continuous_instance.initial_environment2d(plot=False)
    assert plt.gcf()
