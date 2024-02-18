import matplotlib.pyplot as plt
import pytest

from microvault.environment.continuous import Continuous


@pytest.fixture
def continuous_instance():
    return Continuous(folder="data/map/", name="map")


def test_grid_map(continuous_instance):
    grid_map = continuous_instance._grid_map()
    assert grid_map.any()


def test_plot_initial_environment(continuous_instance):
    continuous_instance.plot_initial_environment(plot=False)
    plt.close()
    assert plt.gcf()
