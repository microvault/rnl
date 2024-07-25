import pytest

from microvault.environment.environment_navigation import NaviEnv


@pytest.fixture
def continuous_instance():
    return NaviEnv()
