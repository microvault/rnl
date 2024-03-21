import itertools

import numpy as np
import numpy.random as npr
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from skimage import measure


class Generator:
    def __init__(
        self,
        type="robot",
        size=10,
        grid_lenght=10,
        random=1300,
    ):
        self.type = type
        self.size = size
        self.grid_lenght = grid_lenght
        self.random = random

    def _generate_maze(self, maze_size, decimation=0.0) -> np.ndarray:
        """
        Generates a maze using Kruskal's algorithm
        """
        m = (maze_size - 1) // 2
        n = (maze_size - 1) // 2
        maze = np.ones((maze_size, maze_size))
        for i, j in list(itertools.product(range(m), range(n))):
            maze[2 * i + 1, 2 * j + 1] = 0
        m = m - 1
        L = np.arange(n + 1)
        R = np.arange(n)
        L[n] = n - 1

        while m > 0:
            for i in range(n):
                j = L[i + 1]
                if i != j and npr.randint(3) != 0:
                    R[j] = R[i]
                    L[R[j]] = j
                    R[i] = i + 1
                    L[R[i]] = i
                    maze[2 * (n - m) - 1, 2 * i + 2] = 0
                if i != L[i] and npr.randint(3) != 0:
                    L[R[i]] = L[i]
                    R[L[i]] = R[i]
                    L[i] = i
                    R[i] = i
                else:
                    maze[2 * (n - m), 2 * i + 1] = 0

            m -= 1

        for i in range(n):
            j = L[i + 1]
            if i != j and (i == L[i] or npr.randint(3) != 0):
                R[j] = R[i]
                L[R[j]] = j
                R[i] = i + 1
                L[R[i]] = i
                maze[2 * (n - m) - 1, 2 * i + 2] = 0

            L[R[i]] = L[i]
            R[L[i]] = R[i]
            L[i] = i
            R[i] = i

        return maze

    def _generate_map(
        self, map_size, num_cells_togo, save_boundary=True, min_blocks=10
    ) -> np.ndarray:

        maze = self._generate_maze(map_size)

        if save_boundary:
            maze = maze[1:-1, 1:-1]
            map_size -= 2

        index_ones = np.arange(map_size * map_size)[maze.flatten() == 1]

        reserve = min(index_ones.size, min_blocks)
        num_cells_togo = min(num_cells_togo, index_ones.size - reserve)

        if num_cells_togo > 0:
            blocks_remove = npr.choice(index_ones, num_cells_togo, replace=False)
            maze[blocks_remove // map_size, blocks_remove % map_size] = 0

        if save_boundary:
            map_size += 2
            maze2 = np.ones((map_size, map_size))
            maze2[1:-1, 1:-1] = maze
            return maze2
        else:
            return maze

    def _map_border(self, m) -> np.ndarray:
        rows, columns = m.shape

        new = np.zeros((rows + 2, columns + 2))

        new[1:-1, 1:-1] = m

        return new

    def world(self):

        m = self._generate_map(self.grid_lenght, self.random, save_boundary=False)

        border = self._map_border(m)
        map_grid = 1 - border

        contornos = measure.find_contours(map_grid, 0.5)

        codes = []
        path_data = []

        grid = [
            (map_grid.shape[1] - 1, map_grid.shape[0] - 1),
            (0, map_grid.shape[0] - 1),
            (0, 0),
            (map_grid.shape[1] - 1, 0),
            (0, 0),
        ]
        code_grid = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]

        for i in range(len(grid)):
            codes.append(code_grid[i])
            path_data.append(grid[i])

        for n, contour in enumerate(contornos):
            for idx, vertex in enumerate(contour):
                if idx == 0:
                    codes.append(Path.MOVETO)
                    path_data.append((vertex[1], vertex[0]))
                elif idx > 0:
                    if idx == len(contour) - 1:
                        codes.append(Path.LINETO)
                        path_data.append((vertex[1], vertex[0]))
                        codes.append(Path.CLOSEPOLY)
                        path_data.append((0, 0))
                    else:
                        codes.append(Path.LINETO)
                        path_data.append((vertex[1], vertex[0]))

        path = Path(path_data, codes)

        path_patch = PathPatch(
            path, edgecolor=(0.1, 0.2, 0.5, 0.15), facecolor=(0.1, 0.2, 0.5, 0.15)
        )

        return path_patch