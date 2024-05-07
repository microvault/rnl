from functools import lru_cache
from typing import List, Tuple

import numpy as np
from numba import njit

lru_cache(maxsize=None)


def range_seg_poly(segment: list, poly: list) -> Tuple[bool, float, float]:
    inter = []

    for segments in poly:
        p = segment[0]
        r = segment[1] - segment[0]
        q = segments[0]
        s = segments[1] - segments[0]

        temp1 = cross_product(r, s)

        if temp1 != 0:
            direction_vec1 = q - p

            cross_product1 = cross_product(direction_vec1, s)
            if cross_product1 != 0:
                t = cross_product1 / cross_product(r, s)
            else:
                t = 1

            direction_vec2 = q - p

            cross_product2 = cross_product(direction_vec2, r)

            if cross_product2 != 0:
                u = cross_product2 / cross_product(r, s)
            else:
                u = 1

            if t >= 0 and t <= 1 and u >= 0 and u <= 1:
                int_point = p + t * r
                lrange = np.sqrt(np.sum((int_point - p) ** 2))
                inter.append([True, int_point, lrange])
    if inter:
        min_lrange_index = np.argmin([point[2] for point in inter])
        return inter[min_lrange_index]
    else:
        return False, 0.0, 0.0


@njit
def filter_segment(segs, x: float, y: float, max_range: int) -> List:

    segments_trans = [
        [np.array(segments[:2]), np.array(segments[2:])] for segments in segs
    ]

    segments_inside = []

    for segment in segments_trans:
        x1, y1 = segment[0]
        x2, y2 = segment[1]
        segment_endpoints = np.array([[x1, y1], [x2, y2]])
        region_center = np.array([x, y])
        distances = np.sqrt(np.sum((segment_endpoints - region_center) ** 2, axis=1))

        if np.all(distances <= 6):
            segments_inside.append(segment)

    return segments_inside


@njit
def extract_segment(stack: List) -> List:
    total_segments = []
    for polygon in stack:
        segments = []
        num_points = len(polygon)
        for i in range(num_points):
            current_point = polygon[i]
            next_point = polygon[(i + 1) % num_points]
            segment = (current_point, next_point)
            segments.append(segment)
        total_segments.extend(segments)

    all_obs = []
    for segment in total_segments:
        all_obs.append((segment[0][0], segment[0][1], segment[1][0], segment[1][1]))

    return all_obs


@njit(fastmath=True)
def cross_product(a, b) -> float:
    """
    Calculate the cross product of two 2D vectors.

    :param a: First vector [a1, a2]
    :param b: Second vector [b1, b2]
    :return: Cross product of the two vectors
    """
    return a[0] * b[1] - a[1] * b[0]
