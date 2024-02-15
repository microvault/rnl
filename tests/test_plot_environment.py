import matplotlib.pyplot as plt
import pytest

from microvault.environment.continuous import Continuous


@pytest.fixture
def continuous_instance():
    return Continuous(plot=False)


def test_plot_initial_environment(continuous_instance):
    continuous_instance.plot_initial_environment()
    plt.pause(2)
    plt.close()
    assert plt.gcf()
