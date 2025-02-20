import numpy as np
from numba import njit, prange


@njit(fastmath=True, parallel=True, cache=True)
def lidar_segments(
    robot_x: float,
    robot_y: float,
    robot_theta: float,
    lidar_range: float,
    lidar_angles: np.ndarray,
    segments,  # Lista de UniTuples (x1, y1, x2, y2)
):
    n = lidar_angles.shape[0]
    points = np.zeros((n, 2), dtype=np.float64)
    distances = np.empty(n, dtype=np.float64)

    for i in prange(n):
        ang = lidar_angles[i] + robot_theta
        # Calcula as componentes do vetor do raio
        r0 = lidar_range * np.cos(ang)
        r1 = lidar_range * np.sin(ang)

        best_dist = 1e12
        best_x = 0.0
        best_y = 0.0
        found = False

        # Itera sobre a lista de segmentos
        for j in range(len(segments)):
            seg = segments[j]
            seg0 = seg[0]
            seg1 = seg[1]
            seg2 = seg[2]
            seg3 = seg[3]

            # Vetor do segmento
            s0 = seg2 - seg0
            s1 = seg3 - seg1
            # Vetor do robô até o início do segmento
            d0 = seg0 - robot_x
            d1 = seg1 - robot_y

            # Produto vetorial entre o raio e o segmento
            cross_rs = r0 * s1 - r1 * s0
            if abs(cross_rs) < 1e-12:
                continue

            t = (d0 * s1 - d1 * s0) / cross_rs
            u = (d0 * r1 - d1 * r0) / cross_rs

            if t >= 0.0 and t <= 1.0 and u >= 0.0 and u <= 1.0:
                ipx = robot_x + t * r0
                ipy = robot_y + t * r1
                dx = ipx - robot_x
                dy = ipy - robot_y
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_x = ipx
                    best_y = ipy
                    found = True

        if found:
            points[i, 0] = best_x
            points[i, 1] = best_y
            distances[i] = best_dist
        else:
            points[i, 0] = 0.0
            points[i, 1] = 0.0
            distances[i] = lidar_range

    return points, distances
