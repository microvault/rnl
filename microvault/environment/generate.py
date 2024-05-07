from typing import List, Tuple

import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from shapely.geometry import LineString, Polygon
from skimage import measure

from .engine.collision import extract_segment
from .engine.world_generate import generate_maze


class Generator:
    def __init__(
        self,
        grid_lenght: int = 10,
        random: int = 1300,
    ):
        self.grid_lenght = grid_lenght
        self.random = random

    def _map_border(self, m: np.ndarray) -> np.ndarray:
        rows, columns = m.shape

        new = np.zeros((rows + 2, columns + 2))

        new[1:-1, 1:-1] = m

        return new

    def line_to_np_stack(self, line: LineString) -> np.ndarray:
        """
        Convert LineString to a numpy stack of points.

        :param line: LineString object
        :return: numpy stack of points as (X, Y)
        """
        coords = np.array(line.coords)

        return np.vstack((coords[:, 0], coords[:, 1])).T

    def world(self) -> Tuple[PathPatch, Polygon, List]:
        m = generate_maze(
            map_size=self.grid_lenght,
            decimation=0.0,
            min_blocks=10,
            num_cells_togo=self.random,
        )

        border = self._map_border(m)
        map_grid = 1 - border

        contours = measure.find_contours(map_grid, 0.5)

        exterior = [
            (border.shape[1] - 1, border.shape[0] - 1),
            (0, border.shape[0] - 1),
            (0, 0),
            (border.shape[1] - 1, 0),
        ]
        interiors = []
        segments = []

        for n, contour in enumerate(contours):
            poly = []
            for idx, vertex in enumerate(contour):
                poly.append((vertex[1], vertex[0]))

            interiors.append(poly)

            interior_segment = LineString(poly)
            segments.append(interior_segment)

        exterior_segment = LineString(exterior + [exterior[0]])
        segments.insert(0, exterior_segment)

        stacks = [self.line_to_np_stack(line) for line in segments]

        segment = extract_segment(stacks)

        poly = Polygon(exterior, holes=interiors)

        if not poly.is_valid:
            poly = poly.buffer(0)
            print("invalid")

        path = Path.make_compound_path(
            Path(np.asarray(poly.exterior.coords)[:, :2]),
            *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors]
        )

        path_patch = PathPatch(
            path, edgecolor=(0.1, 0.2, 0.5, 0.15), facecolor=(0.1, 0.2, 0.5, 0.15)
        )

        return path_patch, poly, segment
