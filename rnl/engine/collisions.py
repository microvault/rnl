from typing import List, Tuple

import numpy as np
from numba import njit
from numba.typed import List as TypedList


@njit(fastmath=True, cache=True)
def filter_list_segment(segs, x, y, max_range):
    """
    Filters segments based on proximity to point (x, y).

    Parameters:
    - segs: List of segments (x1, y1, x2, y2).
    - x, y: Coordinates of center point.
    - max_range: Maximum range to filter segments.

    Returns:
    - List of filtered segments.
    """
    segs_arr = np.array(segs)
    d1 = np.sqrt((segs_arr[:, 0] - x) ** 2 + (segs_arr[:, 1] - y) ** 2)
    d2 = np.sqrt((segs_arr[:, 2] - x) ** 2 + (segs_arr[:, 3] - y) ** 2)
    mask = (d1 <= max_range) & (d2 <= max_range)
    arr = segs_arr[mask]
    result = TypedList()
    for i in range(arr.shape[0]):
        result.append((arr[i, 0], arr[i, 1], arr[i, 2], arr[i, 3]))
    return result


@njit(fastmath=True, cache=True)
def is_counter_clockwise(polygon: List[Tuple[float, float]]) -> bool:

    sum = 0.0
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        sum += (x2 - x1) * (y2 + y1)
    return sum < 0


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


@njit
def convert_to_segments(polygon):
    segments = []
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        segments.append((x1, y1, x2, y2))
    return segments


@njit
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


@njit
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
