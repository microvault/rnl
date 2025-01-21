import random
from typing import List, Tuple

import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def lidar_segments(
    robot_x: float,
    robot_y: float,
    robot_theta: float,
    lidar_range: float,
    lidar_angles: np.ndarray,
    segments: np.ndarray,
):
    """
    Returns two things for each angle:
    1) Intersection point (x, y) or (0.0, 0.0) if none
    2) Distance to that intersection or lidar_range if none
    """
    n = lidar_angles.shape[0]
    points = np.zeros((n, 2), dtype=np.float64)
    distances = np.zeros(n, dtype=np.float64)

    def cross2d(a, b):
        return a[0] * b[1] - a[1] * b[0]

    p_init = np.array([robot_x, robot_y])

    for i in range(n):
        ang = lidar_angles[i] + robot_theta
        r = np.array([lidar_range * np.cos(ang), lidar_range * np.sin(ang)])

        found = False
        best_dist = 1e12
        best_pt = np.zeros(2)

        for seg in segments:
            q = np.array([seg[0], seg[1]])
            s = np.array([seg[2] - seg[0], seg[3] - seg[1]])
            direction_vec = q - p_init

            cross_rs = cross2d(r, s)
            if abs(cross_rs) < 1e-12:
                continue

            t = cross2d(direction_vec, s) / cross_rs
            u = cross2d(direction_vec, r) / cross_rs

            if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
                ip = p_init + t * r
                dx = ip[0] - p_init[0]
                dy = ip[1] - p_init[1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < best_dist:
                    best_dist = dist
                    best_pt = ip
                    found = True

        if found:
            points[i, 0] = best_pt[0]
            points[i, 1] = best_pt[1]
            distances[i] = best_dist
        else:
            points[i, 0] = 0.0
            points[i, 1] = 0.0
            distances[i] = lidar_range

    return points, distances


@njit(fastmath=True, cache=True)
def filter_list_segment(
    segs: List[Tuple[float, float, float, float]], x: float, y: float, max_range: float
) -> List:
    """
    Filters segments based on proximity to point (x, y).

    Parameters:
    - segs: List of segments (x1, y1, x2, y2).
    - x, y: Coordinates of center point.
    - max_range: Maximum range to filter segments.

    Returns:
    - List of filtered segments.
    """
    segments_inside = []
    region_center = np.array([x, y])

    for x1, y1, x2, y2 in segs:
        seg_ends = np.array([[x1, y1], [x2, y2]])
        distances = np.sqrt(np.sum((seg_ends - region_center) ** 2, axis=1))
        if np.all(distances <= max_range):
            segments_inside.append((x1, y1, x2, y2))

    return segments_inside


@njit(fastmath=True, cache=True)
def convert_to_segments(polygon):
    segments = []
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        segments.append((x1, y1, x2, y2))
    return segments


@njit(fastmath=True, cache=True)
def is_counter_clockwise(polygon: List[Tuple[float, float]]) -> bool:

    sum = 0.0
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        sum += (x2 - x1) * (y2 + y1)
    return sum < 0


@njit(fastmath=True, cache=True)
def extract_segment_from_polygon(stack: List) -> List:
    """
    Extracts line segments from a stack of polygons.

    Parameters:
    stack (List): List of polygons, each represented by a list of points.

    Returns:
    List: List of line segments extracted from the polygons.
    """

    assert len(stack) >= 1, "Stack must have at least 1 polygon."

    total_segments = []
    for polygon in stack:
        segments = connect_polygon_points(polygon)
        total_segments.extend(segments)

    return convert_to_line_segments(total_segments)


@njit(fastmath=True, cache=True)
def convert_to_line_segments(total_segments: List) -> List:
    """
    Converts a list of line segments into formed (x1, y1, x2, y2) representing start point, end point

    Parameters:
    total_segments (List): List of line segments.

    Returns:
    List: List of line segments converted into (x1, y1, x2, y2) format.
    """

    assert len(total_segments) >= 3, "Polygon must have at least 3 points."

    line_segments = []
    for segment in total_segments:
        line_segments.append(
            (segment[0][0], segment[0][1], segment[1][0], segment[1][1])
        )

    return line_segments


@njit(fastmath=True, cache=True)
def connect_polygon_points(polygon: np.ndarray) -> List:
    """
    Connects the points of a polygon to form line segments.

    Parameters:
    polygon (np.ndarray): The polygon represented by a list of points.

    Returns:
    List: List of line segments connecting the points of the polygon.
    """

    #                segment 1
    # (point 1) +-----------------+ (point 2)
    #           |                 |
    # segment 4 |                 | segment 2
    #           |                 |
    # (point 4) +-----------------+ (point 3)
    #                segment 3

    assert len(polygon) >= 3, "Polygon must have at least 3 points."

    num_points = len(polygon)
    segments = []
    for i in range(num_points):
        current_point = polygon[i]
        # wrap-around to close the polygon
        next_point = polygon[(i + 1) % num_points]
        segment = (current_point, next_point)
        segments.append(segment)

    return segments


@njit
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


@njit
def _point_in_polygon(px, py, exterior, holes):
    """
    Verifica se ponto está no polígono (exterior e 0+ buracos).
    """
    if not _point_in_ring(px, py, exterior):
        return False
    for hole in holes:
        if _point_in_ring(px, py, hole):
            return False
    return True


def spawn_robot_and_goal(
    poly,
    robot_clearance=3.0,
    goal_clearance=3.0,
    min_robot_goal_dist=2.0,
    max_tries=1000,
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
        ext = np.array(shp.exterior.coords, dtype=np.float64)
        holes = [np.array(i.coords, dtype=np.float64) for i in shp.interiors]
        return ext, holes


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
