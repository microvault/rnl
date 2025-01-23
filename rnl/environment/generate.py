from dataclasses import dataclass
from typing import List

import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from numba import njit
from shapely.geometry import LineString, Polygon
from skimage import measure

from rnl.engine.world import GenerateWorld


@dataclass
class Generator:
    def __init__(self, mode: str):
        self.mode = mode
        self.generate = GenerateWorld()

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

    def world(self, grid_lenght):
        """
        Generates a maze world.

        Returns:
        Tuple[PathPatch, Polygon, List]: A tuple containing:
        - PathPatch: The PathPatch object representing the maze.
        - Polygon: The Polygon object representing the maze boundaries.
        - List: List of LineString segments representing the maze segments.
        """
        if self.mode == "easy-01":
            # Tamanho do grid
            width, height = grid_lenght, grid_lenght

            exterior = []
            # 1 (topo)
            for x in range(width):
                exterior.append((x, height - 1))
            # 2 (lado direito)
            for y in range(height - 2, -1, -1):
                exterior.append((width - 1, y))
            # 3 (base)
            for x in range(width - 2, -1, -1):
                exterior.append((x, 0))
            # 4 (lado esquerdo)
            for y in range(1, height - 1):
                exterior.append((0, y))

            # Nenhum interior (sem buracos)
            interiors = []

            # Cria o polígono
            poly = Polygon(exterior, holes=interiors).buffer(0)
            if not poly.is_valid:
                poly = poly.buffer(0)
                if not poly.is_valid:
                    raise ValueError("Polígono inválido.")

            polygon = np.array(exterior + [exterior[0]], dtype=np.float32)

            stack = [polygon]

            # Agora geramos os segmentos (x1, y1, x2, y2)
            segments = extract_segment_from_polygon(stack)

            path = Path.make_compound_path(
                Path(np.asarray(poly.exterior.coords)[:, :2]),
                *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors]
            )

            # Cria o patch
            path_patch = PathPatch(
                path, edgecolor=(0.1, 0.2, 0.5, 0.15), facecolor=(0.1, 0.2, 0.5, 0.15)
            )

            return path_patch, segments, poly

        elif self.mode == "medium":
            m = self.generate.generate_maze(
                map_size=grid_lenght,
                decimation=0.0,
                min_blocks=0,
                num_cells_togo=100,
                no_mut=True,
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

            segment = extract_segment_from_polygon(stacks)

            poly = Polygon(exterior, holes=interiors).buffer(0)

            if not poly.is_valid:
                poly = poly.buffer(0)
                if not poly.is_valid:
                    raise ValueError("The polygon is not valid.")

            path = Path.make_compound_path(
                Path(np.asarray(poly.exterior.coords)[:, :2]),
                *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors]
            )

            path_patch = PathPatch(
                path, edgecolor=(0.1, 0.2, 0.5, 0.15), facecolor=(0.1, 0.2, 0.5, 0.15)
            )

            return path_patch, segment, poly


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
