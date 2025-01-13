import math
import random
from functools import lru_cache

from shapely.geometry import Point, Polygon


class CurriculumTargetPosition:
    """
    Classe para controlar o "currículo" de spawn do objetivo:
      - Começa spawn perto do robô (fração pequena).
      - Conforme atinge recompensas, vai aumentando a fração.
      - Quando chega ao máximo, significa qualquer lugar no mapa (não apenas longe).
    """

    def __init__(
        self,
        total_steps: int = 40_000_000,
        min_fraction: float = 0.01,
        max_fraction: float = 1.0,
        increase_smoothness: float = 1.0,
    ):
        """
        :param total_steps: total de steps (ex: 40 milhões) para chegar no max_fraction
        :param min_fraction: fração inicial do raio
        :param max_fraction: fração máxima do raio (1.0 significa mapa todo)
        :param increase_smoothness: controla a curva de crescimento (1.0 = linear)
        """
        self.total_steps = total_steps
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction
        self.increase_smoothness = increase_smoothness

    def get_fraction(self, current_step: int) -> float:
        """
        Retorna a fração de spawn do objetivo de acordo com o step atual.
        Aqui faço um crescimento suave entre min_fraction e max_fraction.
        """
        progress = min(current_step / self.total_steps, 1.0)
        progress_curved = progress**self.increase_smoothness

        fraction = (
            self.min_fraction
            + (self.max_fraction - self.min_fraction) * progress_curved
        )
        return fraction


@lru_cache(maxsize=None)
def get_polygon_bounds(poly):
    """Função cacheada (exemplo) para obter bounds e evitar chamar .bounds várias vezes."""
    return poly.bounds


def spawn_robot_and_goal_curriculum(
    poly: Polygon,
    fraction: float,
    robot_clearance: float = 3.0,
    goal_clearance: float = 3.0,
    min_robot_goal_dist: float = 2.0,
    max_tries: int = 1000,
):
    """
    Gera posições aleatórias para robô e objetivo dentro do 'poly',
    mas leva em conta a fração de raio de spawn para o objetivo.
      - Se fraction < 1.0, o objetivo só pode spawnar até (fraction * raio_máximo) de distância do robô.
      - Se fraction = 1.0, pode spawnar em qualquer lugar do polígono.
    """

    safe_poly_robot = poly.buffer(-robot_clearance)
    safe_poly_goal = poly.buffer(-goal_clearance)

    if safe_poly_robot.is_empty or safe_poly_goal.is_empty:
        raise ValueError("Clearance muito grande. Polígono invalido.")

    minx_r, miny_r, maxx_r, maxy_r = get_polygon_bounds(safe_poly_robot)
    minx_g, miny_g, maxx_g, maxy_g = get_polygon_bounds(safe_poly_goal)

    def random_point_in_poly(shp, minx, miny, maxx, maxy, tries=1000):
        for _ in range(tries):
            rx = random.uniform(minx, maxx)
            ry = random.uniform(miny, maxy)
            if shp.contains(Point(rx, ry)):
                return (rx, ry)
        return None

    for _ in range(max_tries):
        robo_pos = random_point_in_poly(
            safe_poly_robot, minx_r, miny_r, maxx_r, maxy_r, tries=50
        )
        if not robo_pos:
            continue

        if fraction >= 0.9999:
            goal_pos = random_point_in_poly(
                safe_poly_goal, minx_g, miny_g, maxx_g, maxy_g, tries=50
            )
        else:
            center_robot = Point(robo_pos)
            dist_max = center_robot.distance(
                Point(
                    get_polygon_bounds(safe_poly_goal)[2],
                    get_polygon_bounds(safe_poly_goal)[3],
                )
            )
            max_dist = dist_max * fraction

            goal_pos = None
            for _ in range(50):
                candidate = random_point_in_poly(
                    safe_poly_goal, minx_g, miny_g, maxx_g, maxy_g, tries=1
                )
                if candidate is not None:
                    dist_to_robot = math.hypot(
                        candidate[0] - robo_pos[0], candidate[1] - robo_pos[1]
                    )
                    if dist_to_robot <= max_dist:
                        goal_pos = candidate
                        break

        if robo_pos and goal_pos:
            dx = robo_pos[0] - goal_pos[0]
            dy = robo_pos[1] - goal_pos[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist >= min_robot_goal_dist:
                return robo_pos, goal_pos

    raise ValueError(
        "Falha ao gerar posições válidas para robô e objetivo com a fração atual."
    )


def max_distance_in_polygon_with_holes(poly: Polygon) -> float:
    """
    Retorna a maior distância (diâmetro) entre dois pontos do polígono
    (considerando sua fronteira externa e buracos).
    """
    boundary = poly.boundary  # Pode ser LineString ou MultiLineString
    coords = []

    if boundary.geom_type == "MultiLineString":
        # boundary é um MultiLineString: iteramos em boundary.geoms
        for linestring in boundary.geoms:
            coords.extend(linestring.coords)
    elif boundary.geom_type == "LineString":
        # boundary é uma só: polígono sem furos
        coords.extend(boundary.coords)
    else:
        raise ValueError(f"Geometria inesperada para boundary: {boundary.geom_type}")

    max_dist = 0.0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dist = math.dist(coords[i], coords[j])  # requer Python 3.8+
            if dist > max_dist:
                max_dist = dist

    return max_dist


def min_robot_goal_spawn_distance(poly, clearance_robot, clearance_goal):
    """
    Retorna a distância mínima entre as regiões válidas de spawn
    para o robô e para o objetivo dentro do polígono 'poly'.
    """
    # Polígono "seguro" para o robô
    safe_poly_robot = poly.buffer(-clearance_robot)
    # Polígono "seguro" para o objetivo
    safe_poly_goal = poly.buffer(-clearance_goal)

    # Distância mínima entre safe_poly_robot e safe_poly_goal
    dist_min = safe_poly_robot.distance(safe_poly_goal)
    return dist_min
