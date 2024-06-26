import matplotlib.pyplot as plt
import pytest

from microvault.environment.continuous import Continuous


@pytest.fixture
def continuous_instance():
    return Continuous(
        time=10,
        size=2,
        fps=100,
        random=300,
        max_speed=1.8,
        min_speed=0.4,
        grid_lenght=10,
    )


def test_environment(continuous_instance):
    continuous_instance.render(plot="")
    assert plt.gcf()
