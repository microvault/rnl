from dataclasses import dataclass
from typing import List, Tuple

# import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from shapely.geometry import LineString, Polygon
from skimage import measure

from .engine.collision import extract_segment
from .engine.world_generate import generate_maze


@dataclass
class Generator:
    def __init__(
        self,
        grid_lenght: int = 10,
        random: int = 1300,
    ):
        self.grid_lenght = grid_lenght
        self.random = random

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
        # Calcula as dimensões do novo mapa
        new_shape = (
            original_map.shape[0] * int(1 / resolution),
            original_map.shape[1] * int(1 / resolution),
        )

        # Cria um novo mapa com a resolução desejada
        new_map = np.zeros(new_shape)

        # Itera sobre cada célula do novo mapa
        for i in range(new_shape[0]):
            for j in range(new_shape[1]):
                # Verifica o valor correspondente no mapa original
                original_value = original_map[int(i * resolution), int(j * resolution)]
                # Define o valor no novo mapa
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
        m = generate_maze(
            map_size=self.grid_lenght,
            decimation=0.0,
            min_blocks=10,
            num_cells_togo=self.random,
        )

        border = self._map_border(m)
        map_grid = 1 - border

        # print(map_grid)

        # new_resolution = 0.1
        # new_map = self.upscale_map(map_grid, new_resolution)

        # print(new_map)

        # map_grid = new_map

        # plt.imshow(new_map, cmap='binary', origin='lower')
        # plt.colorbar()
        # plt.show()

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

        # x, y = segment.xy  # Extrai coordenadas x e y do segmento

        # for seg in segment:
        #     x_values = [seg[0], seg[2]]
        #     y_values = [seg[1], seg[3]]
        #     plt.plot(x_values, y_values, color='blue')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('Line Segments')
        # plt.grid(True)
        # plt.show()

        # for i in range(len(stacks)):
        #     x, y = stacks[i].T
        #     plt.plot(x, y, color='black')

        # plt.plot(x, y)  # Plota o segmento
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('Segmento')
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.show()

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


# env = Generator()
# path_patch, poly, segment = env.world()
