import matplotlib.pyplot as plt
import pytest

from microvault.environment.continuous import Continuous


@pytest.fixture
def continuous_instance():
    return Continuous()


def test_environment(continuous_instance):
    continuous_instance.trainer(visualize="")
    assert plt.gcf()
