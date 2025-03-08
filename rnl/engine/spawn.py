import random

import numpy as np
from numba.typed import List


def _point_in_ring(px, py, ring):
    """
    Ray-casting para verificar se o ponto (px, py) está dentro
    de um anel (lista de coords [(x0,y0), (x1,y1), ...]).
    """
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        if ((y1 > py) != (y2 > py)) and (
            px < (x2 - x1) * (py - y1) / (y2 - y1 + 1e-15) + x1
        ):
            inside = not inside
    return inside


def _point_in_polygon(px, py, exterior, holes):
    """
    Verifica se ponto está no polígono (exterior e 0+ buracos).
    """
    if not _point_in_ring(px, py, exterior):
        return False
    if len(holes) == 0:
        return True
    for i in range(len(holes)):
        if _point_in_ring(px, py, holes[i]):
            return False
    return True


def to_python_format(shp):
    """
    Converte o shapely polygon para listas de coordenadas.
    Se for MultiPolygon, seleciona o polígono de maior área.
    """
    if shp.geom_type == "MultiPolygon":
        shp = max(list(shp.geoms), key=lambda a: a.area)
    ext = list(shp.exterior.coords)
    holes = [list(interior.coords) for interior in shp.interiors]
    return ext, holes


def spawn_robot_and_goal(
    poly,
    robot_clearance=3.0,
    goal_clearance=3.0,
    min_robot_goal_dist=2.0,
    max_tries=1000,
    dataset=None,
):
    """
    Gera posições aleatórias para robô e objetivo dentro do 'poly',
    respeitando distância de segurança (clearance) e distância mínima entre eles.
    Utiliza Numba para acelerar o checagem de ponto dentro do polígono.
    """

    safe_poly_robot = poly.buffer(-robot_clearance)
    safe_poly_goal = poly.buffer(-goal_clearance)

    if safe_poly_robot.is_empty or safe_poly_goal.is_empty:
        raise ValueError("Clearance muito grande. Polígono invalido.")

    minx_r, miny_r, maxx_r, maxy_r = safe_poly_robot.bounds
    minx_g, miny_g, maxx_g, maxy_g = safe_poly_goal.bounds

    def to_numba_format(shp):
        if shp.geom_type == "MultiPolygon":
            shp = max(shp.geoms, key=lambda g: g.area)
        ext = np.array(shp.exterior.coords, dtype=np.float64)
        holes_list = List()
        for interior in shp.interiors:
            holes_list.append(np.array(interior.coords, dtype=np.float64))
        return ext, holes_list

    if dataset:
        ext_robot, holes_robot = to_python_format(safe_poly_robot)
        ext_goal, holes_goal = to_python_format(safe_poly_goal)

    else:
        ext_robot, holes_robot = to_numba_format(safe_poly_robot)
        ext_goal, holes_goal = to_numba_format(safe_poly_goal)

    def random_point_in_poly(bounds, ext, holes):
        for _ in range(max_tries):
            rx = random.uniform(bounds[0], bounds[2])
            ry = random.uniform(bounds[1], bounds[3])
            if _point_in_polygon(rx, ry, ext, holes):
                return rx, ry
        return None

    for _ in range(max_tries):
        robo_pos = random_point_in_poly(
            (minx_r, miny_r, maxx_r, maxy_r), ext_robot, holes_robot
        )
        goal_pos = random_point_in_poly(
            (minx_g, miny_g, maxx_g, maxy_g), ext_goal, holes_goal
        )
        if robo_pos and goal_pos:
            dx = robo_pos[0] - goal_pos[0]
            dy = robo_pos[1] - goal_pos[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist >= min_robot_goal_dist:
                return robo_pos, goal_pos

    raise ValueError("Falha ao gerar posicoes validas para robô e objetivo.")
