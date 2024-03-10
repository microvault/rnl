import itertools

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from mpl_toolkits.mplot3d import art3d
from skimage import measure
from tqdm import tqdm


class Continuous:
    def __init__(
        self,
        n=10,
        time=100,
        size=2,
        speed=1.8,
        grid_lenght=50,
    ):
        self.num_agents = n
        self.time = time
        self.size = size
        self.speed = speed
        self.frame = 100

        self.grid_lenght = grid_lenght

        self.xmax = grid_lenght
        self.ymax = grid_lenght

        self.x = np.zeros((self.num_agents, self.time))  # position
        self.y = np.zeros((self.num_agents, self.time))  # position
        self.sp = np.zeros((self.num_agents, self.time))  # speed
        self.theta = np.zeros((self.num_agents, self.time))  # angle
        self.vx = np.zeros((self.num_agents, self.time))  # velocity
        self.vy = np.zeros((self.num_agents, self.time))  # velocity

        self.target_x = np.zeros((self.num_agents, self.time))  # position
        self.target_y = np.zeros((self.num_agents, self.time))  # position

        self.agents = [None for _ in range(self.num_agents)]
        self.targets = [None for _ in range(self.num_agents)]

        self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 6))

        self.path_patch = None

        self.random = 1300

    def _generate_maze(self, maze_size, decimation=0.0):
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
    ):

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

    def _x_direction(self, agents, i, num_agents, xmax, x, vx, time) -> None:
        for a in range(0, num_agents):
            if (time - 1) != i:
                if x[a, i] + vx[a, i] >= xmax or x[a, i] + vx[a, i] <= 0:
                    x[a, i + 1] = x[a, i] - vx[a, i]
                    vx[a, i + 1] = -vx[a, i]
                else:
                    x[a, i + 1] = x[a, i] + vx[a, i]
                    vx[a, i + 1] = vx[a, i]
            else:
                if x[a, i] + vx[a, i] >= xmax or x[a, i] + vx[a, i] <= 0:
                    x[a, i] = x[a, i] - vx[a, i]
                    vx[a, i] = -vx[a, i]
                else:
                    x[a, i] = x[a, i] + vx[a, i]
                    vx[a, i] = vx[a, i]

    def _y_direction(self, agents, i, num_agents, ymax, y, vy, time) -> None:
        for a in range(0, num_agents):
            if (time - 1) != i:
                if y[a, i] + vy[a, i] >= ymax or y[a, i] + vy[a, i] <= 0:
                    y[a, i + 1] = y[a, i] - vy[a, i]
                    vy[a, i + 1] = -vy[a, i]
                else:
                    y[a, i + 1] = y[a, i] + vy[a, i]
                    vy[a, i + 1] = vy[a, i]
            else:
                if y[a, i] + vy[a, i] >= ymax or y[a, i] + vy[a, i] <= 0:
                    y[a, i] = y[a, i] - vy[a, i]
                    vy[a, i] = -vy[a, i]
                else:
                    y[a, i] = y[a, i] + vy[a, i]
                    vy[a, i] = vy[a, i]

    def _map_border(self, m) -> np.ndarray:
        rows, columns = m.shape

        new = np.zeros((rows + 2, columns + 2))

        new[1:-1, 1:-1] = m

        return new

    def _get_label(self, timestep):
        line1 = "Environment\n"
        line2 = "Time Step:".ljust(14) + f"{timestep:4.0f}\n"
        return line1 + line2

    def _generate_sim(self):

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

    def environment(self, plot=False):

        self.ax.remove()
        self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")

        self.ax.set_xlim(0, self.grid_lenght)
        self.ax.set_ylim(0, self.grid_lenght)

        path = self._generate_sim()

        self.ax.add_patch(path)

        art3d.pathpatch_2d_to_3d(path, z=0, zdir="z")

        self.ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # # Hide grid lines
        self.ax.grid(False)

        # Hide axes ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])

        # Hide axes
        self.ax.set_axis_off()

        # Set camera
        self.ax.elev = 20
        self.ax.azim = -155
        self.ax.dist = 1

        label = self.ax.text(
            0,
            0,
            0.6,
            self._get_label(0),
        )

        label.set_fontsize(14)
        label.set_fontweight("normal")
        label.set_color("#666666")

        self.fig.subplots_adjust(left=0, right=1, bottom=0.1, top=1)

        for a in tqdm(range(0, self.num_agents), desc="Initializing environment"):
            self.agents[a] = self.ax.plot3D(
                np.random.uniform(0, self.xmax),
                np.random.uniform(0, self.ymax),
                0,
                marker="o",
                markersize=self.size,
            )[0]

            self.targets[a] = self.ax.plot3D(
                np.random.uniform(0, self.xmax),
                np.random.uniform(0, self.ymax),
                0,
                marker="x",
                markersize=self.size,
            )[0]

        def init():
            for a in range(self.num_agents):

                self.target_x[a, 0] = np.random.uniform(0, self.xmax)
                self.target_y[a, 0] = np.random.uniform(0, self.ymax)

                self.x[a, 0] = np.random.uniform(0, self.xmax)
                self.y[a, 0] = np.random.uniform(0, self.ymax)
                self.sp[a, 0] = np.random.uniform(0, self.speed)
                self.theta[a, :] = np.random.uniform(0, 2 * np.pi)
                self.vx[a, 0] = self.sp[a, 0] * np.cos(self.theta[a, 0])
                self.vy[a, 0] = self.sp[a, 0] * np.sin(self.theta[a, 0])

            for patch in self.ax.patches:
                patch.remove()

            new_map_path = self._generate_sim()
            self.ax.add_patch(new_map_path)
            art3d.pathpatch_2d_to_3d(new_map_path, z=0, zdir="z")

        def animate(i):
            for a, (agent, target) in enumerate(zip(self.agents, self.targets)):
                self._x_direction(
                    a, i, self.num_agents, self.xmax, self.x, self.vx, self.time
                )
                self._y_direction(
                    a, i, self.num_agents, self.ymax, self.y, self.vy, self.time
                )

                agent.set_data_3d(
                    [self.x[a, i]],
                    [self.y[a, i]],
                    [0],
                )

                target.set_data_3d(
                    [self.target_x[a, 0]],
                    [self.target_y[a, 0]],
                    [0],
                )

            label.set_text(self._get_label(i))

        if plot == True:
            ani = animation.FuncAnimation(
                self.fig,
                animate,
                init_func=init,
                blit=False,
                frames=self.time,
                interval=self.frame,
            )
            plt.show()


# engine = Continuous().environment(plot=True)
