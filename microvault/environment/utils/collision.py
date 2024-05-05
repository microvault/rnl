from typing import Tuple

import numpy as np


def range_seg_poly(segment: list, poly: list) -> Tuple[bool, float, float]:
    inter = []

    for segments in poly:
        p = segment[0]
        r = segment[1] - segment[0]
        q = segments[0]
        s = segments[1] - segments[0]

        temp1 = np.cross(r, s)
        temp2 = np.cross(q - p, r)

        if temp1 == 0 and temp2 == 0:
            # collinear
            t0 = (q - p) @ r / (r @ r if r @ r != 0 else 1)
            t1 = t0 + s @ r / (r @ r if r @ r != 0 else 1)

            if max(t0, t1) >= 0 and min(t0, t1) < 0:
                int_point = p
                lrange = 0
                inter.append([1.0, int_point, lrange])

            elif min(t0, t1) >= 0 and min(t0, t1) <= 1:
                int_point = p + min(t0, t1) * r
                lrange = np.linalg.norm(int_point - p)
                inter.append([1.0, int_point, lrange])

        elif temp1 != 0:
            t = np.cross(q - p, s) / (np.cross(r, s) if np.cross(r, s) != 0 else 1)
            u = np.cross(q - p, r) / (np.cross(r, s) if np.cross(r, s) != 0 else 1)

            if t >= 0 and t <= 1 and u >= 0 and u <= 1:
                int_point = p + t * r
                lrange = np.linalg.norm(int_point - p)
                inter.append([True, int_point, lrange])
    if inter:
        min_lrange_index = np.argmin([point[2] for point in inter])
        return inter[min_lrange_index]
    else:
        return False, 0.0, 0.0
