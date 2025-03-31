from typing import List, Tuple

from rnl.engine.collisions import filter_list_segment
from rnl.engine.utils import Index, Property


class SpatialIndex:
    def __init__(self, segments: List[Tuple[float, float, float, float]]):
        """
        Initializes the spatial index with the given segments.

        Parameters:
        - segments: List of map segments, each defined by (x1, y1, x2, y2).
        """
        p = Property()
        p.dimension = 2
        self.idx = Index(properties=p)
        for i, seg in enumerate(segments):
            x1, y1, x2, y2 = seg
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            self.idx.insert(i, (xmin, ymin, xmax, ymax))
        self.segments = segments

    def query(
        self, x: float, y: float, max_range: float
    ) -> List[Tuple[float, float, float, float]]:
        """
        Query the segments that are within the maximum range from the position (x, y).

        Parameters:
        - x, y: Robot coordinates.
        - max_range: Maximum range to filter the segments.

        Returns:
        - List of filtered segments.
        """
        return [
            self.segments[i]
            for i in self.idx.intersection(
                (x - max_range, y - max_range, x + max_range, y + max_range)
            )
        ]

    def filter_segments(
        self, x: float, y: float, max_range: float
    ) -> List[Tuple[float, float, float, float]]:
        """
        Filters segments based on robot position and maximum range.

        Parameters:
        - x, y: Robot coordinates.
        - max_range: Maximum range to filter segments.

        Returns:
        - List of filtered segments.
        """
        segs_proximos = self.query(x, y, max_range)
        if not segs_proximos:
            return []
        segs_filtrados = filter_list_segment(segs_proximos, x, y, max_range)
        return segs_filtrados
