import functools
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.pyplot import imread
from mpl_toolkits.mplot3d import art3d
from tqdm import tqdm
from yaml import SafeLoader, load


class Continuous:
    def __init__(
        self,
        folder=None,
        name=None,
        silent=False,
        fig_width=8,
        fig_height=8,
    ):
        self.fig_width = fig_width
        self.fig_height = fig_height
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

    def occupancy(self):
        occ = np.array(self._occupancy)
        return occ

    @functools.lru_cache(maxsize=None)
    def _grid_map(self) -> np.ndarray:
        """This function receives the grid map and filters only the region of the map

        Returns:
            np.ndarray: grid map
        """
        data = self.occupancy()

        data = np.where(data < 0, 0, data)
        data = np.where(data != 0, 1, data)

        idx = np.where(data == 0)

        min_x = np.min(idx[1])
        max_x = np.max(idx[1])
        min_y = np.min(idx[0])
        max_y = np.max(idx[0])

        dist_x = (max_x - min_x) + 1
        dist_y = (max_y - min_y) + 1

        dist_max = max(dist_x, dist_y)
        dist_min = min(dist_x, dist_y)

        dist = (dist_max - dist_min) / 2

        if dist % 2 != 0:
            dist = int(dist) + 1

        new_min_y = min_y - dist
        new_max_y = max_y + dist

        if (new_max_y - new_min_y) != (max_x - min_x):
            diff_y = abs(new_max_y - new_min_y)
            diff_x = abs(max_x - min_x)

            diff = diff_y - diff_x

            # TODO: if x is less than y
            if diff_y < diff_x:
                new_max_y = new_max_y + diff

            if diff_y > diff_x:
                max_x = max_x + diff

        map_record = data[new_min_y : new_max_y + 1, min_x : max_x + 1]

        new_map_grid = np.zeros_like(map_record)
        new_map_grid[map_record == 0] = 1

        return new_map_grid

    def plot_initial_environment(self, plot=True) -> None:
        """generate environment from map"""

        new_map_grid = self._grid_map()

        idx = np.where(new_map_grid.sum(axis=0) > 0)[0]

        min_idx = np.min(idx)
        max_idx = np.max(idx)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.remove()
        ax = fig.add_subplot(1, 1, 1, projection="3d")

        ax.set_xlim(min_idx, max_idx * 0.6)
        ax.set_ylim(min_idx, max_idx * 0.6)
        ax.set_zlim(-0.01, 0.01)

        idx = np.where(new_map_grid.sum(axis=0) > 0)[0]

        min_idx = np.min(idx)
        max_idx = np.max(idx)

        for i in tqdm(range(min_idx, max_idx), desc="Plotting environment"):
            for j in range(min_idx, max_idx):
                if new_map_grid[i, j] == 1:
                    polygon = [(j, i), (j + 1, i), (j + 1, i + 1), (j, i + 1)]
                    poly = Polygon(polygon, color=(0, 0, 0, 1))
                    ax.add_patch(poly)
                    art3d.pathpatch_2d_to_3d(poly, z=0, zdir="z")

        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # Hide grid lines
        ax.grid(False)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Hide axes
        ax.set_axis_off()

        # Set camera
        ax.elev = 20
        ax.azim = -155
        ax.dist = 1

        fig.subplots_adjust(left=0, right=1.1, bottom=-0.2, top=1)

        label = ax.text(
            0,
            0,
            0.02,
            "Continuous Tag\n".lower(),
        )

        label.set_fontsize(14)
        label.set_fontweight("normal")
        label.set_color("#666666")

        if plot == True:
            plt.show()
        else:
            return fig
