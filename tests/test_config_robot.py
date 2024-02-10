import pytest

from microvault.config_robot import ConfigRobot


def test_config_robot_default_values():
    config = ConfigRobot()
    assert config.radius is None
    assert config.backwards is None
    assert config.typeRobot is None
    assert config.vel_linear is None
    assert config.val_angular is None


def test_config_robot_with_values():
    radius = 2.5
    backwards = True
    type_robot = "onmi"
    vel_linear = 1.5
    val_angular = 0.8

    config = ConfigRobot(
        radius=radius,
        backwards=backwards,
        typeRobot=type_robot,
        vel_linear=vel_linear,
        val_angular=val_angular,
    )

    assert config.radius == radius
    assert config.backwards == backwards
    assert config.typeRobot == type_robot
    assert config.vel_linear == vel_linear
    assert config.val_angular == val_angular


if __name__ == "__main__":
    pytest.main()
