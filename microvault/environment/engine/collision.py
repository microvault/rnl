from typing import List, Tuple

import numpy as np
from numba import njit


def range_seg_poly(segment: list, poly: list) -> Tuple[bool, float, float]:
    inter = []

    for segments in poly:
        p = segment[0]
        r = segment[1] - segment[0]
        q = segments[0]
        s = segments[1] - segments[0]

        temp1 = np.cross(r, s)

        if temp1 != 0:

            t = np.cross(q - p, s) / (np.cross(r, s) if np.cross(r, s) != 0 else 1)
            u = np.cross(q - p, r) / (np.cross(r, s) if np.cross(r, s) != 0 else 1)

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
