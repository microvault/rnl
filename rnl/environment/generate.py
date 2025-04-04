import json
import os
import random
from dataclasses import dataclass

import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from shapely.geometry import LineString, Polygon

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

    def world(
        self,
        grid_length: float,
        resolution: float = 0.05,
        porcentage_obstacle: float = 40.0,
    ):
        """
        Generates a maze world.

        Returns:
        Tuple[PathPatch, Polygon, List]: A tuple containing:
        - PathPatch: The PathPatch object representing the maze.
        - Polygon: The Polygon object representing the maze boundaries.
        - List: List of LineString segments representing the maze segments.
        """
        if self.mode in ("easy-10"):

            # map_grid = np.array([[0, 0, 0],
            #                      [0, 1, 0],
            #                      [0, 0, 0]])

            # conts = find_contours(map_grid, 0.5)
            # contours = process(conts)

            # height, width = map_grid.shape
            # exterior = []

            # # 1
            # for x in range(width):
            #     exterior.append((x, height))

            # # 2
            # for y in range(height, -1, -1):
            #     exterior.append((width, y))

            # # 3
            # for x in range(width, -1, -1):
            #     exterior.append((x, 0))

            # # 4
            # for y in range(1, height):
            #     exterior.append((0, y))

            # interiors = []
            # segments = []

            # for n, contour in enumerate(contours):
            #     poly = []
            #     for idx, vertex in enumerate(contour):
            #         poly.append((vertex[1], vertex[0]))

            #     interiors.append(poly)

            #     interior_segment = LineString(poly)
            #     segments.append(interior_segment)

            # exterior_segment = LineString(exterior + [exterior[0]])
            # segments.insert(0, exterior_segment)

            # stacks = [self.line_to_np_stack(line) for line in segments]

            # segment = extract_segment_from_polygon(stacks)

            # Defina as coordenadas do quadrado externo
            exterior = [(0, 0), (3, 0), (3, 3), (0, 3)]

            # Defina as coordenadas do quadrado interno
            interior = [(1, 1), (2, 1), (2, 2), (1, 2)]

            # Crie as LineStrings (exterior + interior)
            segments = []
            exterior_segment = LineString(exterior + [exterior[0]])
            segments.append(exterior_segment)

            interior_segment = LineString(interior + [interior[0]])
            segments.append(interior_segment)

            # Exemplo de como converter cada segment em seu formato (caso precise)
            stacks = [self.line_to_np_stack(line) for line in segments]
            segment = extract_segment_from_polygon(stacks)

            # Construa seu polígono final se precisar
            poly = Polygon(exterior, holes=[interior])

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

        if self.mode in ("easy-00"):
            width = int(grid_length) + 1
            height = int(grid_length) + 1

            exterior = []
            for x in range(width):
                exterior.append((x, (height - 1)))
            for y in range(height - 2, -1, -1):
                exterior.append(((width - 1), y))
            for x in range(width - 2, -1, -1):
                exterior.append((x, 0))
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

        if self.mode in ("easy-01", "easy-02", "easy-03"):
            width = int(grid_length / resolution) + 1
            height = int(grid_length / resolution) + 1

            exterior = []
            for x in range(width):
                exterior.append((x * resolution, (height - 1) * resolution))
            for y in range(height - 2, -1, -1):
                exterior.append(((width - 1) * resolution, y * resolution))
            for x in range(width - 2, -1, -1):
                exterior.append((x * resolution, 0))
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

        elif self.mode == "easy-04" or self.mode == "easy-05":
            m = self.generate.generate_maze(
                map_size=int(grid_length),
                decimation=1000.0,  # 1000.0
                min_blocks=0,
                no_mut=True,
                porcentage_obstacle=porcentage_obstacle,
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

        elif self.mode == "visualize":
            m = self.generate.generate_maze(
                map_size=int(grid_length),
                decimation=0.0,
                min_blocks=0,
                no_mut=True,
                porcentage_obstacle=porcentage_obstacle,
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

        elif self.mode == "train-mode":
            m = self.generate.generate_maze(
                map_size=int(grid_length),
                decimation=100.0,
                min_blocks=0,
                no_mut=True,
                porcentage_obstacle=porcentage_obstacle,
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
                poly_corrigido = poly.buffer(0)
                if (
                    poly_corrigido.is_valid
                    and poly_corrigido.geom_type != "MultiPolygon"
                ):
                    poly = poly_corrigido
                else:
                    # Se continuar sendo MultiPolygon, seleciona o maior polígono válido
                    valid_polys = [p for p in poly.geoms if p.is_valid]
                    if valid_polys:
                        poly = max(valid_polys, key=lambda a: a.area)
                    else:
                        raise ValueError(
                            "Nenhum polígono válido foi gerado dos vértices."
                        )

            processed_verts = np.array(poly.exterior.coords, dtype=np.float32)
            stack = [processed_verts]
            segments = extract_segment_from_polygon(stack)

            path = Path.make_compound_path(
                Path(np.asarray(poly.exterior.coords)[:, :2]),
                *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors],
            )
            path_patch = PathPatch(
                path, edgecolor=(0.1, 0.2, 0.5, 0.15), facecolor=(0.1, 0.2, 0.5, 0.15)
            )
            return path_patch, segments, poly
        else:
            raise ValueError(f"Modo {self.mode} não implementado.")
