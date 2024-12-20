import numpy as np
from typing import List, Tuple
from rtree import index
from numba import njit

class SpatialIndex:
    def __init__(self, segments: List[Tuple[float,float,float,float]]):
        p = index.Property()
        p.dimension = 2
        self.idx = index.Index(properties=p)
        for i, seg in enumerate(segments):
            x1, y1, x2, y2 = seg
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            self.idx.insert(i, (xmin, ymin, xmax, ymax))
        self.segments = segments

    def query(self, x: float, y: float, max_range: float):
        return [self.segments[i] for i in self.idx.intersection((x - max_range, y - max_range, x + max_range, y + max_range))]

@njit
def is_distance_within_range(distance: np.ndarray, lidar_range: float) -> bool:
    return np.all(distance <= lidar_range)

@njit
def calculate_distances(seg_ends: np.ndarray, center: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((seg_ends - center) ** 2, axis=1))


def filter_list_segment(segs: List[Tuple[float,float,float,float]], x: float, y: float, max_range: float) -> List:
    segments_inside = []
    region_center = np.array([x, y])
    for (x1, y1, x2, y2) in segs:
        seg_ends = np.array([[x1, y1], [x2, y2]])
        distances = calculate_distances(seg_ends, region_center)
        if is_distance_within_range(distances, max_range):
            segments_inside.append((x1, y1, x2, y2))
    return segments_inside
