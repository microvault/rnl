import json
import os
import random
from dataclasses import dataclass

import numpy as np
import yaml
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from shapely import affinity
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid
from skimage.measure import find_contours

from rnl.engine.collisions import extract_segment_from_polygon
from rnl.engine.polygons import find_contour, process
from rnl.engine.utils import load_pgm
from rnl.engine.world import GenerateWorld


@dataclass
class Generator:
    def __init__(self, render: bool = False, folder: str = "", name: str = ""):
        self.folder = folder
        self.name = name
        self.render = render
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
        grid_length: float = 0,
        resolution: float = 0.05,
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
        thresh = 0.65
        yaml_path = self.folder + "/" + self.name + ".yaml"

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
        gx, gy = w * res, h * res

        # ------------------------------------------------------------------------
        contours = find_contours(occ.astype(float), 0.5)

        stacks = []
        paths = []
        all_poly = []
        for c in contours:
            pts = np.stack(
                [ox + c[:, 1] * res, oy + (h - c[:, 0]) * res],
                axis=1,
            )

            closed_pts = np.vstack([pts, pts[0]])
            codes = [Path.MOVETO] + [Path.LINETO] * (len(pts) - 1) + [Path.CLOSEPOLY]

            paths.append(Path(closed_pts, codes))
            p = Polygon(closed_pts)
            all_poly.append(p)
            stacks.append(closed_pts.astype(np.float64))

        poly = unary_union(all_poly)

        if oyaw:
            cx, cy = ox + gx / 2, oy + gy / 2
            poly = affinity.rotate(
                poly, np.degrees(oyaw), origin=(cx, cy), use_radians=False
            )

        segments = []
        for stk in stacks:
            segments.extend(extract_segment_from_polygon([stk]))

        patch = PathPatch(
            Path.make_compound_path(*paths),
            edgecolor=(0.1, 0.2, 0.5, 0.15),
            facecolor=(0.1, 0.2, 0.5, 0.15),
            linewidth=1.0,
        )

        return patch, segments, poly
