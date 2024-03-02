import functools

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import art3d
from tqdm import tqdm


class Continuous:
    def __init__(
        self,
        n=100,
        time=100,
        size=5,
        speed=100,
        grid_lenght=50,
    ):
        self.num_agents = n
        self.time = time
        self.size = size
        self.speed = speed

        self.grid_lenght = grid_lenght

        self.xmax = grid_lenght
        self.ymax = grid_lenght

        self.x = np.zeros((self.num_agents, self.time))
        self.y = np.zeros((self.num_agents, self.time))
        self.sp = np.zeros((self.num_agents, self.time))
        self.theta = np.zeros((self.num_agents, self.time))
        self.vx = np.zeros((self.num_agents, self.time))
        self.vy = np.zeros((self.num_agents, self.time))

    def _x_direction(self, a, i, num_agents, xmax, x, vx) -> None:
        for a in range(0, num_agents):
            try:
                if x[a, i] + vx[a, i] >= xmax or x[a, i] + vx[a, i] <= 0:
                    x[a, i + 1] = x[a, i] - vx[a, i]
                    vx[a, i + 1] = -vx[a, i]
                else:
                    x[a, i + 1] = x[a, i] + vx[a, i]
                    vx[a, i + 1] = vx[a, i]
            except IndexError:
                pass

    def _y_direction(self, a, i, num_agents, ymax, y, vy) -> None:
        for a in range(0, num_agents):
            try:
                if y[a, i] + vy[a, i] >= ymax or y[a, i] + vy[a, i] <= 0:
                    y[a, i + 1] = y[a, i] - vy[a, i]
                    vy[a, i + 1] = -vy[a, i]
                else:
                    y[a, i + 1] = y[a, i] + vy[a, i]
                    vy[a, i + 1] = vy[a, i]
            except IndexError:
                pass

    def _get_label(self, timestep):
        line1 = "Environment\n"
        line2 = "Time Step:".ljust(14) + f"{timestep:4.0f}\n"
        return line1 + line2

    @functools.lru_cache(maxsize=None)
    def environment(self, plot=False):

        # TODO:
        # new_map_grid = np.ones((50, 50), dtype=int)

        # min_idx, max_idx = 0, len(new_map_grid)

        # x = np.arange(min_idx, max_idx, 1)
        # y = np.arange(min_idx, max_idx, 1)
        # x, y = np.meshgrid(x, y)

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.remove()
        ax = fig.add_subplot(1, 1, 1, projection="3d")

        ax.set_xlim(0, 50)
        ax.set_ylim(0, 50)

        corner_points = [
            (0, 0),
            (0, self.grid_lenght),
            (self.grid_lenght, self.grid_lenght),
            (self.grid_lenght, 0),
        ]
        poly = Polygon(corner_points, color=(0.1, 0.2, 0.5, 0.15))
        ax.add_patch(poly)
        art3d.pathpatch_2d_to_3d(poly, z=0, zdir="z")

        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # # Hide grid lines
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

        label = ax.text(
            0,
            0,
            0.6,
            self._get_label(0),
        )

        label.set_fontsize(14)
        label.set_fontweight("normal")
        label.set_color("#666666")

        fig.subplots_adjust(left=0, right=1, bottom=0.1, top=1)

        lines = [None for _ in range(self.num_agents)]

        for a in tqdm(range(0, self.num_agents), desc="Plot environment"):
            lines[a] = ax.plot3D(
                np.random.uniform(0, self.xmax),
                np.random.uniform(0, self.ymax),
                0,
                marker="o",
                markersize=self.size,
            )[0]
            self.sp[a, 0] = np.random.uniform(0, self.speed)
            self.theta[a, :] = np.random.uniform(0, 2 * np.pi)
            self.vx[a, 0] = self.sp[a, 0] * np.cos(self.theta[a, 0])
            self.vy[a, 0] = self.sp[a, 0] * np.sin(self.theta[a, 0])

        def animate(i):
            for a, line in enumerate(lines):
                self._x_direction(a, i, self.num_agents, self.xmax, self.x, self.vx)
                self._y_direction(a, i, self.num_agents, self.ymax, self.y, self.vy)
                line.set_data_3d(
                    [self.x[a, i]],
                    [self.y[a, i]],
                    [0],
                )
            label.set_text(self._get_label(i))

        if plot == True:
            ani = animation.FuncAnimation(
                fig, animate, blit=False, frames=self.time, interval=100
            )
            plt.show()
