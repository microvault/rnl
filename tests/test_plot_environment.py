import matplotlib.pyplot as plt
import pytest

# from microvault.environment.continuous import Continuous

# @pytest.fixture
# def continuous_instance():
#     return Continuous(n=10, time=10, size=2, speed=1, grid_lenght=10)


# def test_x_direction(continuous_instance):
#     continuous_instance._x_direction(10, 10, 10, 50, np.zeros((20, 0)), np.zeros((20, 0)), 10)
#     assert True


# def test_y_direction(continuous_instance):
#     continuous_instance._y_direction(1, 1, 10, 50, np.zeros((20, 0)), np.zeros((20, 0)), 10)
#     assert True


# def test_environment(continuous_instance):
#     continuous_instance.environment(plot=False)
#     assert plt.gcf()
