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

    # Inicializar listas para armazenar posições
    robot_x = []
    robot_y = []
    goal_x = []
    goal_y = []

    # Inicializar variáveis para rastrear distâncias
    min_distance = float("inf")
    max_distance = 0.0
    distances = []  # Opcional: armazenar todas as distâncias para análise adicional

    for _ in range(iterations):
        robot_pos, goal_pos = spawn_robot_and_goal(
            poly=poly,
            robot_clearance=4.0,
            goal_clearance=2.0,
            min_robot_goal_dist=2.0,
        )

        # Verificar se o objetivo está dentro do polígono
        assert is_inside_polygon(
            goal_pos, poly
        ), f"Goal position {goal_pos} está fora do polígono."

        # Verificar se robô e objetivo não estão na mesma posição
        assert robot_pos != goal_pos, "Robot and goal positions are the same."

        # Armazenar as posições
        robot_x.append(robot_pos[0])
        robot_y.append(robot_pos[1])
        goal_x.append(goal_pos[0])
        goal_y.append(goal_pos[1])

        # Calcular a distância Euclidiana entre robô e objetivo
        distance = np.linalg.norm(np.array(robot_pos) - np.array(goal_pos))
        distances.append(distance)

        # Atualizar min e max distâncias
        if distance < min_distance:
            min_distance = distance
        if distance > max_distance:
            max_distance = distance

    # Converter listas de posições para arrays numpy para eficiência
    robot_x = np.array(robot_x)
    robot_y = np.array(robot_y)
    goal_x = np.array(goal_x)
    goal_y = np.array(goal_y)
    distances = np.array(distances)

    # Exibir as menores e maiores distâncias
    print(f"Polígono: {poly}")
    print(f"Menor distância entre robô e objetivo: {min_distance:.4f}")
    print(f"Maior distância entre robô e objetivo: {max_distance:.4f}")

    # Opcional: Estatísticas adicionais
    print(f"Distância média: {distances.mean():.4f}")
    print(f"Desvio padrão da distância: {distances.std():.4f}")

    # Plotar os resultados
    plt.figure(figsize=(10, 10))

    # Plotar o polígono
    x_poly, y_poly = poly.exterior.xy
    plt.plot(x_poly, y_poly, color="black", linewidth=2, label="Polígono")

    # Plotar posições do robô
    plt.scatter(robot_x, robot_y, color="red", alpha=0.3, label="Robô", s=5)

    # Plotar posições do objetivo
    plt.scatter(goal_x, goal_y, color="blue", alpha=0.3, label="Objetivo", s=5)

    # Remover os ticks dos eixos
    plt.xticks([])
    plt.yticks([])

    # Criar proxy artists para min e max distâncias
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

    # Adicionar legenda
    # plt.legend()

    # Adicionar título
    plt.title(f"Posições do Robô e Objetivo no Polígono {poly}")

    # Mostrar o plot
    plt.show()
