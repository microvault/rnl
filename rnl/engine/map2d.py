import functools
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import imread
from yaml import SafeLoader, load
import random

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

    def get_subgrid_from_map(self, map_grid, border=10):
        """
        Extrai uma subgrade da grid onde há valores positivos e adiciona uma borda.

        Parâmetros:
          - map_grid: grid original (valores entre 0 e 1)
          - border: tamanho da borda a ser adicionada

        Retorna:
          - subgrid_uint8: subgrade em uint8 com valores escalados (0-255) e borda.
        """
        # Encontra as colunas com qualquer valor > 0
        idx = np.where(map_grid.sum(axis=0) > 0)[0]
        if idx.size == 0:
            return None
        min_idx = np.min(idx)
        max_idx = np.max(idx)
        subgrid = map_grid[:, min_idx : max_idx + 1]
        # Converte para uint8 (0 a 255)
        subgrid_uint8 = (subgrid * 255).astype(np.uint8)
        # Adiciona uma borda de tamanho fixo
        subgrid_uint8 = np.pad(
            subgrid_uint8,
            pad_width=((border, border), (border, border)),
            mode="constant",
            constant_values=0,
        )
        return subgrid_uint8

    def divide_map_into_quadrants(
        self, map_grid: np.ndarray, quadrant: int
    ) -> np.ndarray:
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

    def erode(self, img, kernel, iterations=1):
        """
        Realiza erosão morfológica em uma imagem em escala de cinza.

        A erosão diminui os objetos claros.
        """
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        out = img.copy()
        for _ in range(iterations):
            padded = np.pad(
                out,
                ((pad_h, pad_h), (pad_w, pad_w)),
                mode="constant",
                constant_values=255,
            )
            # Cria uma "sliding window view" da imagem
            windows = np.lib.stride_tricks.sliding_window_view(padded, (k_h, k_w))
            out = np.min(windows, axis=(-2, -1))
        return out

    def dilate(self, img, kernel, iterations=1):
        """
        Realiza dilatação morfológica em uma imagem em escala de cinza.

        A dilatação expande os objetos claros.
        """
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        out = img.copy()
        for _ in range(iterations):
            padded = np.pad(
                out,
                ((pad_h, pad_h), (pad_w, pad_w)),
                mode="constant",
                constant_values=0,
            )
            windows = np.lib.stride_tricks.sliding_window_view(padded, (k_h, k_w))
            out = np.max(windows, axis=(-2, -1))
        return out

    def connected_components(self, binary_img):
        """
        Rótula componentes conectados (8-conectividade) em uma imagem binária.

        Retorna:
          - labels: matriz com os rótulos dos componentes
          - num_labels: número total de componentes encontrados
        """
        h, w = binary_img.shape
        labels = np.zeros((h, w), dtype=np.int32)
        label_counter = 1
        for i in range(h):
            for j in range(w):
                if binary_img[i, j] > 0 and labels[i, j] == 0:
                    stack = [(i, j)]
                    labels[i, j] = label_counter
                    while stack:
                        x, y = stack.pop()
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < h and 0 <= ny < w:
                                    if binary_img[nx, ny] > 0 and labels[nx, ny] == 0:
                                        labels[nx, ny] = label_counter
                                        stack.append((nx, ny))
                    label_counter += 1
        return labels, label_counter - 1

    def get_largest_component(self, binary_img):
        """
        Retorna a máscara da maior componente conectada na imagem binária.
        """
        labels, num_labels = self.connected_components(binary_img)
        if num_labels < 1:
            return None
        # Calcula a área de cada componente
        areas = np.array([np.sum(labels == l) for l in range(1, num_labels + 1)])
        largest_component = np.argmax(areas) + 1  # Rótulos começam em 1
        mask = np.zeros_like(binary_img, dtype=np.uint8)
        mask[labels == largest_component] = 255
        return mask

    def closing_morphology(self, img, kernel, iterations=1):
        """
        Aplica fechamento morfológico: dilatação seguida de erosão.
        """
        dilated = self.dilate(img, kernel, iterations=iterations)
        closed = self.erode(dilated, kernel, iterations=iterations)
        return closed

    def smooth_mask(self, img):
        """
        Realiza uma suavização simples na máscara usando operações morfológicas
        com um kernel 1x1 (não altera muito, mas pode remover ruídos pequenos).
        """
        kernel_smooth = np.ones((1, 1), dtype=np.uint8)
        opened = self.erode(img, kernel_smooth, iterations=1)
        smoothed = self.dilate(opened, kernel_smooth, iterations=1)
        return smoothed

    # --- Função principal ---
    def initial_environment2d(
        self,
        plot=False,
        kernel_size=(3, 3),
        morph_iterations=1,
        approx_epsilon_factor=0.001,
        mode="medium-00",
    ):
        """
        Prepara o ambiente 2D a partir da grid do mapa.

        Passos:
        1. Extrai a subgrade da grid e adiciona borda.
        2. Aplica operações morfológicas (erosão e dilatação) para remover ruídos.
        3. Detecta a maior componente conectada.
        4. Aplica fechamento morfológico e suavização.
        5. (Opcional) Plota o resultado.

        Retorna:
        - A máscara final processada (contour_mask) ou None se não houver região.
        """
        # 1. Obter grid e extrair subgrid com borda
        if mode == "medium-00":
            num = random.randint(1, 4)
            new_map_grid = self.divide_map_into_quadrants(self._grid_map(), num)
        elif mode == "medium-01":
            new_map_grid = self._grid_map()
        else:
            new_map_grid = self._grid_map()

        subgrid_uint8 = self.get_subgrid_from_map(new_map_grid, border=10)
        if subgrid_uint8 is None:
            return None

        # 2. Definir kernel para morfologia e aplicar erosão/dilatação
        kernel = np.ones(kernel_size, dtype=np.uint8)
        eroded = self.erode(subgrid_uint8, kernel, iterations=morph_iterations)
        dilated = self.dilate(eroded, kernel, iterations=morph_iterations)

        # 3. Extrair a maior componente conectada (imagem binária: valores > 0)
        mask_component = self.get_largest_component(dilated)
        if mask_component is None:
            return None

        # 4. Aplicar fechamento morfológico e suavização
        mask_closed = self.closing_morphology(
            mask_component, kernel, iterations=morph_iterations
        )
        mask_smoothed = self.smooth_mask(mask_closed)

        # 5. Plot se solicitado
        if plot:
            plt.figure(figsize=(10, 10))
            plt.imshow(mask_smoothed, cmap="gray")
            plt.axis("off")
            plt.show()

        return mask_smoothed
