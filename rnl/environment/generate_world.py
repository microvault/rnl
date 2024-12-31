from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from shapely.geometry import LineString, Polygon
from skimage import measure
from numba import njit
from rnl.engine.world_generate import GenerateWorld
from rnl.engine.map2d import Map2D


@dataclass
class Generator:
    def __init__(
        self,
        random: float,
        mode: str,
        folder: str,
        name: str,
    ):
        self.random = random
        # self.collision = Collision()
        self.generate = GenerateWorld()
        if folder != "None":
            self.map2d = Map2D(folder=folder, name=name, silent=False)
            self.use_map = True
        else:
            self.use_map = False


        self.debug = True

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

    # TODO
    def upscale_map(self, original_map, resolution):
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

    def world(self, grid_lenght):
        """
        Generates a maze world.

        Returns:
        Tuple[PathPatch, Polygon, List]: A tuple containing:
        - PathPatch: The PathPatch object representing the maze.
        - Polygon: The Polygon object representing the maze boundaries.
        - List: List of LineString segments representing the maze segments.
        """
        m = np.empty((0, 0))

        if self.use_map:
            contour_mask = self.map2d.initial_environment2d(plot=False)
            if contour_mask is None:
                raise ValueError("Failed to generate the contour mask.")
            m = contour_mask

            if m is None or m.size == 0:
                raise ValueError("A máscara 'm' está vazia ou não foi gerada corretamente.")

            contours = measure.find_contours(m, 0.5)

            all_x = []
            all_y = []

            for contour in contours:
                for point in contour:
                    y, x = point
                    all_x.append(x)
                    all_y.append(y)

            exteriors = []
            interiors = []
            segments = []

            for contour in contours:
                poly = [(vertex[1], vertex[0]) for vertex in contour]

                if is_counter_clockwise(poly):
                    exteriors.append(poly)
                else:
                    interiors.append(poly)

            if not exteriors:
                raise ValueError("Nenhum contorno exterior encontrado.")

            exteriors_sorted = sorted(exteriors, key=lambda p: Polygon(p).area, reverse=True)
            exterior = exteriors_sorted[0]

            for extra_exterior in exteriors_sorted[1:]:
                interiors.append(extra_exterior)

            exterior_segment = LineString(exterior + [exterior[0]])
            segments.append(exterior_segment)

            int_segments = []
            for interior in interiors:
                interior_segment = LineString(interior + [interior[0]])
                segments.append(interior_segment)
                int_segments.append(interior_segment)

            ext_segments = convert_to_segments(exterior)
            int_segments = [convert_to_segments(interior) for interior in interiors]

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

            return path_patch, ext_segments, int_segments, segment, contour_mask, poly

        else:
            contour_mask = self.generate.generate_maze(
                map_size=grid_lenght,
                decimation=0.0,
                min_blocks=5,
                num_cells_togo=1300,
            )

            border = self._map_border(contour_mask)
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

            ext_segments = convert_to_segments(exterior)
            int_segments = [convert_to_segments(interior) for interior in interiors]

            stacks = [self.line_to_np_stack(line) for line in segments]

            segment = extract_segment_from_polygon(stacks)

            poly = Polygon(exterior, holes=interiors).buffer(0)

            path = Path.make_compound_path(
                Path(np.asarray(poly.exterior.coords)[:, :2]),
                *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors]
            )

            path_patch = PathPatch(
                path, edgecolor=(0.1, 0.2, 0.5, 0.15), facecolor=(0.1, 0.2, 0.5, 0.15)
            )

            return path_patch, ext_segments, int_segments, segment, contour_mask, poly

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
def is_counter_clockwise(polygon: List[Tuple[float, float]]) -> bool:

    sum = 0.0
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        sum += (x2 - x1) * (y2 + y1)
    return sum < 0


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
