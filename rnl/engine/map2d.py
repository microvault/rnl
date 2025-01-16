import functools
import os
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import imread
from yaml import SafeLoader, load


class Map2D:
    def __init__(
        self,
        folder: str,
        name: str,
    ):
        self.path = folder

        if folder is None or name is None:
            return

        folder = os.path.expanduser(folder)
        yaml_file = os.path.join(folder, name + ".yaml")

        with open(yaml_file) as stream:
            mapparams = load(stream, Loader=SafeLoader)
        map_file = os.path.join(folder, mapparams["image"])

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

    def divide_map_into_quadrants(self, map_grid: np.ndarray, quadrant: int) -> np.ndarray:
        """
        Returns one of the four quadrants of the map_grid.

        Parameters:
            map_grid (np.ndarray): The occupancy grid map.
            quadrant (int): The quadrant number (1, 2, 3, or 4).

        Returns:
            np.ndarray: The specified quadrant of the map.
        """
        height, width = map_grid.shape
        mid_height, mid_width = height // 2, width // 2

        if quadrant == 1:
            return map_grid[:mid_height, :mid_width]
        elif quadrant == 2:
            return map_grid[:mid_height, mid_width:]
        elif quadrant == 3:
            return map_grid[mid_height:, :mid_width]
        elif quadrant == 4:
            return map_grid[mid_height:, mid_width:]
        else:
            raise ValueError("Quadrant must be 1, 2, 3, or 4.")

    def initial_environment2d(
        self,
        plot: bool = False,
        kernel_size: Tuple[int, int] = (3, 3),
        morph_iterations: int = 1,
        approx_epsilon_factor: float = 0.01,
        contour_retrieval_mode: int = cv2.RETR_TREE,
        contour_approx_method: int = cv2.CHAIN_APPROX_SIMPLE,
    ) -> Optional[np.ndarray]:
        # _map_grid = self._grid_map()
        new_map_grid = self._grid_map()

        # new_map_grid = self.divide_map_into_quadrants(_map_grid, 1)

        idx = np.where(new_map_grid.sum(axis=0) > 0)[0]
        if idx.size == 0:
            return None

        min_idx = np.min(idx)
        max_idx = np.max(idx)

        subgrid = new_map_grid[:, min_idx : max_idx + 1]

        subgrid_uint8 = (subgrid * 255).astype(np.uint8)

        border_size = 10
        subgrid_uint8 = cv2.copyMakeBorder(
            subgrid_uint8,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )

        kernel = np.ones(kernel_size, np.uint8)
        eroded = cv2.erode(subgrid_uint8, kernel, iterations=morph_iterations)
        dilated = cv2.dilate(eroded, kernel, iterations=morph_iterations)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            dilated, connectivity=8
        )
        if num_labels <= 1:
            return None

        largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.zeros_like(dilated, dtype=np.uint8)
        mask[labels == largest_component] = 255

        mask_closed = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations
        )

        contours, hierarchy = cv2.findContours(
            mask_closed, contour_retrieval_mode, contour_approx_method
        )
        if not contours:
            return None

        contour_mask = np.zeros_like(mask_closed, dtype=np.uint8)
        for i, contour in enumerate(contours):
            contour_int = contour.astype(np.int32)
            epsilon = approx_epsilon_factor * cv2.arcLength(contour_int, True)
            approx = cv2.approxPolyDP(contour_int, epsilon, True)
            approx_int = approx.astype(np.int32)

            if hierarchy[0][i][3] == -1:
                cv2.fillPoly(contour_mask, [approx_int], 255)
            else:
                cv2.fillPoly(contour_mask, [approx_int], 0)

        kernel_smooth = np.ones((1, 1), np.uint8)
        contour_mask = cv2.morphologyEx(contour_mask, cv2.MORPH_OPEN, kernel_smooth)

        if plot:
            plt.figure(figsize=(10, 10))
            plt.imshow(contour_mask, cmap="gray")
            plt.axis("off")
            plt.savefig("contour.png", bbox_inches='tight', pad_inches=0)  # Salva a imagem antes de mostrar
            plt.show()

        return contour_mask


if __name__ == "__main__":
    map = Map2D("/Users/nicolasalan/microvault/rnl/data/map4", "map4")
    map.initial_environment2d(plot=True)
