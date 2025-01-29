import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.lines import Line2D
from shapely.geometry import Point, Polygon

from rnl.engine.collisions import spawn_robot_and_goal
from rnl.environment.world import CreateWorld


def is_inside_polygon(point, polygon: Polygon):
    return polygon.contains(Point(point))


@pytest.fixture
def poly():
    generator = CreateWorld(
        folder="/Users/nicolasalan/microvault/rnl/data/map4",
        name="map4",
    )
    _, _, poly = generator.world()

    return poly


def test_spawn_robot_and_goal(poly, request):
    iterations = 100000

    robot_x = []
    robot_y = []
    goal_x = []
    goal_y = []

    min_distance = float("inf")
    max_distance = 0.0
    distances = []
    for _ in range(iterations):
        robot_pos, goal_pos = spawn_robot_and_goal(
            poly=poly,
            robot_clearance=4.0,
            goal_clearance=2.0,
            min_robot_goal_dist=4.0,
        )

        assert is_inside_polygon(
            goal_pos, poly
        ), f"Goal position {goal_pos} está fora do polígono."

        assert robot_pos != goal_pos, "Robot and goal positions are the same."

        robot_x.append(robot_pos[0])
        robot_y.append(robot_pos[1])
        goal_x.append(goal_pos[0])
        goal_y.append(goal_pos[1])

        distance = np.linalg.norm(np.array(robot_pos) - np.array(goal_pos))
        distances.append(distance)

        if distance < min_distance:
            min_distance = distance
        if distance > max_distance:
            max_distance = distance

    robot_x = np.array(robot_x)
    robot_y = np.array(robot_y)
    goal_x = np.array(goal_x)
    goal_y = np.array(goal_y)
    distances = np.array(distances)

    plt.figure(figsize=(10, 10))

    x_poly, y_poly = poly.exterior.xy
    plt.plot(x_poly, y_poly, color="black", linewidth=2, label="Polígono")

    plt.scatter(robot_x, robot_y, color="red", alpha=0.3, label="Robô", s=5)

    plt.scatter(goal_x, goal_y, color="blue", alpha=0.3, label="Objetivo", s=5)

    plt.xticks([])
    plt.yticks([])

    min_proxy = Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=f"Menor Distância: {min_distance:.2f}",
        markerfacecolor="none",
        markersize=0,
    )
    max_proxy = Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=f"Maior Distância: {max_distance:.2f}",
        markerfacecolor="none",
        markersize=0,
    )

    plt.legend(
        handles=plt.gca().get_legend_handles_labels()[0] + [min_proxy, max_proxy]
    )
    plt.title(f"Posições do Robô e Objetivo no Polígono {poly}")

    plt.show()
