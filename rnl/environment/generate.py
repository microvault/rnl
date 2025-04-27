import json
import os
import random
from dataclasses import dataclass

import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely import affinity
from rnl.engine.collisions import extract_segment_from_polygon
from rnl.engine.polygons import find_contour, process
from rnl.engine.world import GenerateWorld
from rnl.engine.utils import load_pgm
from skimage.measure import find_contours
import yaml
from shapely.ops import unary_union

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

    def cell_to_world(self, i, j, res):
        x = (i + 0.5) * res
        y = (j + 0.5) * res
        return x, y

    def world(
        self,
        grid_length: float,
        resolution: float = 0.05,
        grid_length_x: float = 0,
        grid_length_y: float = 0,
        porcentage_obstacle: float = 20.0,
    ):
        """
        Generates a maze world.

        Returns:
        Tuple[PathPatch, Polygon, List]: A tuple containing:
        - PathPatch: The PathPatch object representing the maze.
        - Polygon: The Polygon object representing the maze boundaries.
        - List: List of LineString segments representing the maze segments.
        """
        if self.mode in ("avoid"):

            exterior = [(0, 0), (3, 0), (3, 3), (0, 3)]

            interior = [(1, 1), (1, 2), (2, 2), (2, 1)]

            segments = [
                LineString(exterior + [exterior[0]]),
                LineString(interior + [interior[0]])
            ]
            stacks   = [self.line_to_np_stack(line) for line in segments]
            segment  = extract_segment_from_polygon(stacks)

            poly = Polygon(exterior, holes=[interior])
            if not poly.is_valid:
                poly = poly.buffer(0)

            path = Path.make_compound_path(
                Path(np.asarray(exterior)[:, :2]),
                Path(np.asarray(interior)[:, :2])
            )

            path_patch = PathPatch(
                path,
                facecolor=(0.1, 0.2, 0.5, 0.15),
                edgecolor=(0.1, 0.2, 0.5, 0.15)
            )

            return path_patch, segment, poly

        elif self.mode in ("turn"):
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

        elif self.mode in ("long"):
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

        elif "custom" in self.mode:
            thresh = 0.65

            yaml_path = "/Users/nicolasalan/microvault/rnl/data/map6/map6.yaml"
            with open(yaml_path) as f:
                    info = yaml.safe_load(f)
            res = float(info["resolution"])
            ox, oy, oyaw = info["origin"]
            pgm_path = os.path.join(os.path.dirname(yaml_path), info["image"])
            img = load_pgm(pgm_path)
            if info.get("negate", 0):
                img = 255 - img
            occ = img < int(thresh * 255)
            h, w = occ.shape
            gx, gy   = w*res, h*res

            contours = find_contours(occ.astype(float), 0.5)
            polys = []
            for c in contours:
                pts = np.stack([
                    ox + c[:,1]*res,
                    oy + (h - c[:,0])*res
                ], axis=1)
                polys.append(Polygon(pts).buffer(0))
            poly = unary_union(polys).buffer(0)

            if oyaw:
                cx, cy = ox + gx/2, oy + gy/2
                poly = affinity.rotate(poly,
                                       np.degrees(oyaw),
                                       origin=(cx, cy),
                                       use_radians=False)

            stack = []
            if isinstance(poly, MultiPolygon):
                for p in poly.geoms:
                    stack.append(np.asarray(p.exterior.coords, dtype=np.float64))
            else:
                stack.append(np.asarray(poly.exterior.coords, dtype=np.float64))

            segments = extract_segment_from_polygon(stack)

            paths = []
            if isinstance(poly, MultiPolygon):
                for p in poly.geoms:
                    coords = np.asarray(p.exterior.coords, np.float64)
                    paths.append(Path(coords[:, :2]))
            else:
                coords = stack[0]
                paths.append(Path(coords[:, :2]))

            patch = PathPatch(
                Path.make_compound_path(*paths),
                edgecolor=(0.1, 0.2, 0.5, 0.15),
                facecolor=(0.1, 0.2, 0.5, 0.15),
            )

            return patch, segments, poly

        elif self.mode in ("easy-01", "easy-02", "easy-03"):
            width  = int(round(grid_length / resolution))
            height = int(round(grid_length / resolution))

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

            cx = (width  - 1) * resolution / 2
            cy = (height - 1) * resolution / 2
            poly = affinity.rotate(poly, -90, origin=(cx, cy), use_radians=False)

            polygon  = np.array(poly.exterior.coords, dtype=np.float32)
            stack    = [polygon]
            segments = extract_segment_from_polygon(stack)
            path     = Path.make_compound_path(
                Path(polygon[:, :2]),
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

            conts = find_contour(map_grid, 0.5)
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

            conts = find_contour(map_grid, 0.5)
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

            conts = find_contour(map_grid, 0.5)
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
