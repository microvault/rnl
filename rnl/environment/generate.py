from dataclasses import dataclass

import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from shapely.geometry import LineString, Polygon
import os, json, random

from rnl.engine.collisions import extract_segment_from_polygon
from rnl.engine.polygons import find_contours, process
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

    def world(self, grid_length: float, resolution: float = 0.01):
        """
        Generates a maze world.

        Returns:
        Tuple[PathPatch, Polygon, List]: A tuple containing:
        - PathPatch: The PathPatch object representing the maze.
        - Polygon: The Polygon object representing the maze boundaries.
        - List: List of LineString segments representing the maze segments.
        """
        if self.mode in ("easy-00"):
            width = int(grid_length) + 1
            height = int(grid_length) + 1

            exterior = []
            # Borda superior
            for x in range(width):
                exterior.append((x, (height - 1)))
            # Borda direita
            for y in range(height - 2, -1, -1):
                exterior.append(((width - 1), y))
            # Borda inferior
            for x in range(width - 2, -1, -1):
                exterior.append((x, 0))
            # Borda esquerda
            for y in range(1, height - 1):
                exterior.append((0, y))

            poly = Polygon(exterior, holes=[]).buffer(0)
            if not poly.is_valid:
                poly = poly.buffer(0)
                if not poly.is_valid:
                    raise ValueError("Polígono inválido.")

            polygon = np.array(exterior + [exterior[0]], dtype=np.float32)
            stack = [polygon]
            segments = extract_segment_from_polygon(stack)
            path = Path.make_compound_path(
                Path(np.asarray(poly.exterior.coords)[:, :2]),
                *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors],
            )
            path_patch = PathPatch(
                path, edgecolor=(0.1, 0.2, 0.5, 0.15), facecolor=(0.1, 0.2, 0.5, 0.15)
            )
            return path_patch, segments, poly

        if self.mode in ("easy-01", "easy-02"):
            width = int(grid_length / resolution) + 1
            height = int(grid_length / resolution) + 1

            exterior = []
            # Borda superior
            for x in range(width):
                exterior.append((x * resolution, (height - 1) * resolution))
            # Borda direita
            for y in range(height - 2, -1, -1):
                exterior.append(((width - 1) * resolution, y * resolution))
            # Borda inferior
            for x in range(width - 2, -1, -1):
                exterior.append((x * resolution, 0))
            # Borda esquerda
            for y in range(1, height - 1):
                exterior.append((0, y * resolution))

            poly = Polygon(exterior, holes=[]).buffer(0)
            if not poly.is_valid:
                poly = poly.buffer(0)
                if not poly.is_valid:
                    raise ValueError("Polígono inválido.")

            polygon = np.array(exterior + [exterior[0]], dtype=np.float32)
            stack = [polygon]
            segments = extract_segment_from_polygon(stack)
            path = Path.make_compound_path(
                Path(np.asarray(poly.exterior.coords)[:, :2]),
                *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors],
            )
            path_patch = PathPatch(
                path, edgecolor=(0.1, 0.2, 0.5, 0.15), facecolor=(0.1, 0.2, 0.5, 0.15)
            )
            return path_patch, segments, poly

        elif self.mode == "easy-03":
            m = self.generate.generate_maze(
                map_size=grid_length,
                decimation=0.0,
                min_blocks=0,
                num_cells_togo=100,
                no_mut=True,
            )

            border = self._map_border(m)
            map_grid = 1 - border

            conts = find_contours(map_grid, 0.5)
            contours = process(conts)

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
                *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors],
            )

            path_patch = PathPatch(
                path, edgecolor=(0.1, 0.2, 0.5, 0.15), facecolor=(0.1, 0.2, 0.5, 0.15)
            )

            return path_patch, segment, poly

        elif self.mode == "hard":

            json_dir = "/Users/nicolasalan/microvault/rnl/dataset/json"
            files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
            if not files:
                raise ValueError("No JSON files found in the dataset directory.")

            random_file = random.choice(files)
            file_path = os.path.join(json_dir, random_file)
            print(f"Selected JSON file: {file_path}")

            data = None
            with open(file_path, "r") as f:
                data = json.load(f)

            if "verts" not in data:
                raise ValueError("The selected JSON does not contain 'verts'.")

            verts = np.array(data["verts"], dtype=np.float32)
            if not np.allclose(verts[0], verts[-1]):
                verts = np.vstack([verts, verts[0]])

            poly = Polygon(verts).buffer(0)
            if not poly.is_valid:
                poly = poly.buffer(0)
                if not poly.is_valid:
                    raise ValueError("Invalid polygon generated from vertices.")

            if poly.geom_type == "MultiPolygon":
                poly = max(poly, key=lambda a: a.area)

            processed_verts = np.array(poly.exterior.coords, dtype=np.float32)
            stack = [processed_verts]
            segments = extract_segment_from_polygon(stack)

            path = Path.make_compound_path(
                Path(np.asarray(poly.exterior.coords)[:, :2]),
                *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors]
            )
            path_patch = PathPatch(
                path, edgecolor=(0.1, 0.2, 0.5, 0.15), facecolor=(0.1, 0.2, 0.5, 0.15)
            )
            return path_patch, segments, poly
        else:
            raise ValueError(f"Modo {self.mode} não implementado.")
