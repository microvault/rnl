import functools
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import imread
from yaml import SafeLoader, load


class Map2D:
    def __init__(
        self,
        folder=None,
        name=None,
        silent=False,
    ):
        self.path = folder

        if folder is None or name is None:
            return

        folder = os.path.expanduser(folder)
        yaml_file = os.path.join(folder, name + ".yaml")

        if not silent:
            print(f"Loading map definition from {yaml_file}")

        with open(yaml_file) as stream:
            mapparams = load(stream, Loader=SafeLoader)
        map_file = os.path.join(folder, mapparams["image"])

        if not silent:
            print(f"Map definition found. Loading map from {map_file}")

        mapimage = imread(map_file)
        temp = (1.0 - mapimage.T[:, ::-1] / 254.0).astype(np.float32)
        mapimage = np.ascontiguousarray(temp)
        self._occupancy = mapimage
        self.occupancy_shape0 = mapimage.shape[0]
        self.occupancy_shape1 = mapimage.shape[1]
        self.resolution_ = mapparams["resolution"]
        self.origin = np.array(mapparams["origin"][:2]).astype(np.float32)

        if mapparams["origin"][2] != 0:
            raise ValueError("Map origin z coordinate must be 0")

        self._thresh_occupied = mapparams["occupied_thresh"]
        self.thresh_free = mapparams["free_thresh"]
        self.HUGE_ = 100 * self.occupancy_shape0 * self.occupancy_shape1

        if self.resolution_ == 0:
            raise ValueError("resolution can not be 0")

    def occupancy_grid(self) -> np.ndarray:
        """return the gridmap without filter

        Returns:
            np.ndarray: occupancy grid
        """
        occ = np.array(self._occupancy)
        return occ

    @functools.lru_cache(maxsize=None)
    def _grid_map(self) -> np.ndarray:
        """This function receives the grid map and filters only the region of the map

        Returns:
            np.ndarray: grid map
        """
        data = self.occupancy_grid()

        data = np.where(data < 0, 0, data)
        data = np.where(data != 0, 1, data)

        idx = np.where(data == 0)

        min_x = np.min(idx[1])
        max_x = np.max(idx[1])
        min_y = np.min(idx[0])
        max_y = np.max(idx[0])

        dist_x = (max_x - min_x) + 1
        dist_y = (max_y - min_y) + 1

        if (max_y - min_y) != (max_x - min_x):
            dist_y = max_y - min_y
            dist_x = max_x - min_x

            diff = round(abs(dist_y - dist_x) / 2)

            # distance y > distance x
            if dist_y > dist_x:
                min_x = int(min_x - diff)
                max_x = int(max_x + diff)

            # distance y < distance x
            if dist_y < dist_x:
                min_y = int(min_y - diff)
                max_y = int(max_y + diff)

        diff_x = max_x - min_x
        diff_y = max_y - min_y

        # TODO: remove this
        if abs((diff_y) - (diff_x)) == 1:

            if diff_y < diff_x:
                max_y = max_y + 1

            if diff_y > diff_x:
                max_x = max_x + 1

        if min(min_x, max_x, min_y, max_y) < 0:
            min_x_adjusted = min_x + abs(min_x)
            max_x_adjusted = max_x + abs(min_x)
            min_y_adjusted = min_y + abs(min_y)
            max_y_adjusted = max_y + abs(min_y)

            map_record = data[
                min_y_adjusted : max_y_adjusted + 1, min_x_adjusted : max_x_adjusted + 1
            ]

        else:
            map_record = data[min_y : max_y + 1, min_x : max_x + 1]

        new_map_grid = np.zeros_like(map_record)
        new_map_grid[map_record == 0] = 1

        return new_map_grid

    def plot_initial_environment2d(self, plot=True) -> None:
        new_map_grid = self._grid_map()

        print(new_map_grid.shape)

        idx = np.where(new_map_grid.sum(axis=0) > 0)[0]

        min_idx = np.min(idx)
        max_idx = np.max(idx)

        subgrid = new_map_grid[:, min_idx : max_idx + 1]

        plt.imshow(subgrid, cmap="gray", interpolation="nearest")
        plt.axis("off")

        if plot:
            plt.show()

    def plot_initial_environment3d(self, plot=True) -> None:
        """generate environment from map"""

        new_map_grid = self._grid_map()

        idx = np.where(new_map_grid.sum(axis=0) > 0)[0]

        min_idx = int(np.min(idx))
        max_idx = int(np.max(idx))

        # print(new_map_grid)

        # all_edges = []

        # for i in tqdm(range(min_idx, max_idx), desc="Plotting environment"):
        #     for j in range(min_idx, max_idx):
        #         if new_map_grid[i, j] == 1:
        #             polygon = [(j, i), (j + 1, i), (j + 1, i + 1), (j, i + 1)]
        #             poly = Polygon(polygon, color=(0.1, 0.2, 0.5, 0.15))

        #             vert = poly.get_xy()
        #             edges = [
        #                 (vert[k], vert[(k + 1) % len(vert)]) for k in range(len(vert))
        #             ]

        #             all_edges.extend(edges)
