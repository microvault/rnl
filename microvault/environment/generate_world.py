from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from shapely.geometry import LineString, Polygon
from skimage import measure

from microvault.engine.collision import Collision
from microvault.engine.world_generate import GenerateWorld


@dataclass
class Generator:
    def __init__(
        self,
        collision: Collision,
        generate: GenerateWorld,
        grid_lenght: int = 10,
        random: int = 1300,
    ):
        self.grid_lenght = grid_lenght
        self.random = random
        self.collision = collision
        self.generate = generate

    @staticmethod
    def _map_border(m: np.ndarray) -> np.ndarray:
        """
        Adds a border around the given map array.

        Parameters:
        m (np.ndarray): The map array to add a border to.

        Returns:
        np.ndarray: The map array with a border added.
        """
        rows, columns = m.shape

        new = np.zeros((rows + 2, columns + 2))

        new[1:-1, 1:-1] = m

        return new

    @staticmethod
    def line_to_np_stack(line: LineString) -> np.ndarray:
        """
        Converts a LineString object to a numpy array stack of points.

        Parameters:
        line (LineString): The LineString object to convert.

        Returns:
        np.ndarray: The numpy array stack of points representing the LineString.
        """
        coords = np.array(line.coords)

        return np.vstack((coords[:, 0], coords[:, 1])).T

    def upscale_map(self, original_map, resolution):
        # TODO
        new_shape = (
            original_map.shape[0] * int(1 / resolution),
            original_map.shape[1] * int(1 / resolution),
        )

        new_map = np.zeros(new_shape)

        for i in range(new_shape[0]):
            for j in range(new_shape[1]):
                original_value = original_map[int(i * resolution), int(j * resolution)]
                new_map[i, j] = original_value

        return new_map

    def world(self) -> Tuple[PathPatch, Polygon, List]:
        """
        Generates a maze world.

        Returns:
        Tuple[PathPatch, Polygon, List]: A tuple containing:
        - PathPatch: The PathPatch object representing the maze.
        - Polygon: The Polygon object representing the maze boundaries.
        - List: List of LineString segments representing the maze segments.
        """
        m = self.generate.generate_maze(
            map_size=self.grid_lenght,
            decimation=0.0,
            min_blocks=0,
            num_cells_togo=self.random,
        )

        border = self._map_border(m)
        map_grid = 1 - border

        contours = measure.find_contours(map_grid, 0.5)

        height, width = map_grid.shape
        exterior = []

        """
        #---------1---------#
        |                   |
        |                   |
        4                   2
        |                   |
        |                   |
        #---------3---------#
        """

        # 1
        for x in range(width):
            exterior.append((x, height - 1))

        # 2
        for y in range(height - 2, -1, -1):
            exterior.append((width - 1, y))

        # 3
        for x in range(width - 2, -1, -1):
            exterior.append((x, 0))

        # 4
        for y in range(1, height - 1):
            exterior.append((0, y))

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

        segment = self.collision.extract_seg_from_polygon(stacks)

        poly = Polygon(exterior, holes=interiors).buffer(0)

        path = Path.make_compound_path(
            Path(np.asarray(poly.exterior.coords)[:, :2]),
            *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors]
        )

        path_patch = PathPatch(
            path, edgecolor=(0.1, 0.2, 0.5, 0.15), facecolor=(0.1, 0.2, 0.5, 0.15)
        )

        return path_patch, poly, segment
