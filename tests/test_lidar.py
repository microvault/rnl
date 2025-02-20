import numpy as np

from rnl.engine.lidar import lidar_segments


def test_lidar_segments_intersection():
    # Cenário: robot no (0,1) e segmento vertical interceptando o feixe horizontal.
    robot_x, robot_y, robot_theta = 0.0, 1.0, 0.0
    lidar_range = 5.0
    lidar_angles = np.array([0.0])
    segments = np.array([[2.0, 0.0, 2.0, 2.0]])

    result = lidar_segments(
        robot_x,
        robot_y,
        robot_theta,
        lidar_range,
        lidar_angles,
        segments,
    )
    points, distances = result

    # Interseção esperada em (2,1) com distância 2.0
    np.testing.assert_almost_equal(points[0], [2.0, 1.0])
    np.testing.assert_almost_equal(distances[0], 2.0)


def test_lidar_segments_no_intersection():
    # Cenário: feixe para cima e segmento não intercepta.
    robot_x, robot_y, robot_theta = 0.0, 0.0, 0.0
    lidar_range = 5.0
    lidar_angles = np.array([np.pi / 2])
    segments = np.array([[2.0, 0.0, 2.0, 2.0]])

    result = lidar_segments(
        robot_x,
        robot_y,
        robot_theta,
        lidar_range,
        lidar_angles,
        segments,
    )
    points, distances = result

    # Sem interseção: ponto deve ser (0,0) e distância igual ao lidar_range
    np.testing.assert_almost_equal(points[0], [0.0, 0.0])
    np.testing.assert_almost_equal(distances[0], lidar_range)
