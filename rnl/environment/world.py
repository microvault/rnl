from dataclasses import dataclass

import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from shapely.geometry import LineString, Polygon

from rnl.engine.collisions import (
    convert_to_segments,
    extract_segment_from_polygon,
    is_counter_clockwise,
)
from rnl.engine.map2d import Map2D
from rnl.engine.polygons import find_contour, process


@dataclass
class CreateWorld:
    def __init__(
        self,
        folder: str,
        name: str,
    ):
        self.map2d = Map2D(folder=folder, name=name)

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

    def world(self, mode: str):
        """
        Generates a maze world.

        Returns:
        Tuple[PathPatch, Polygon, List]: A tuple containing:
        - PathPatch: The PathPatch object representing the maze.
        - Polygon: The Polygon object representing the maze boundaries.
        - List: List of LineString segments representing the maze segments.
        """
        m = np.empty((0, 0))

        contour_mask = self.map2d.initial_environment2d(plot=False, mode=mode)
        if contour_mask is None:
            raise ValueError("Failed to generate the contour mask.")
        m = contour_mask

        if m is None or m.size == 0:
            raise ValueError("A máscara 'm' está vazia ou não foi gerada corretamente.")

        conts = find_contour(m, 0.5)
        contours = process(conts)

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

        exteriors_sorted = sorted(
            exteriors, key=lambda p: Polygon(p).area, reverse=True
        )
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
            *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors],
        )

        path_patch = PathPatch(
            path, edgecolor=(0.1, 0.2, 0.5, 0.15), facecolor=(0.1, 0.2, 0.5, 0.15)
        )

        return path_patch, segment, poly
