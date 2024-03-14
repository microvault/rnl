import itertools

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
from generate import Generator
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from mpl_toolkits.mplot3d import art3d
from robot import Robot
from skimage import measure


class Continuous:
    def __init__(
        self,
        n=1,
        time=100,
        size=3,
        speed=1.8,
        grid_lenght=10,
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

        self.generator = Generator(grid_lenght=grid_lenght, random=self.random)
        self.robot = Robot()

    def _get_label(self, timestep):
        line1 = "Environment\n"
        line2 = "Time Step:".ljust(14) + f"{timestep:4.0f}\n"
        return line1 + line2

    def environment(self, plot=False):

        self.ax.remove()
        self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")

        self.ax.set_xlim(0, self.grid_lenght)
        self.ax.set_ylim(0, self.grid_lenght)

        path = self.generator.world()

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

        for a in range(0, self.num_agents):
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

            new_map_path = self.generator.world()
            self.ax.add_patch(new_map_path)
            art3d.pathpatch_2d_to_3d(new_map_path, z=0, zdir="z")

        def animate(i):
            for a, (agent, target) in enumerate(zip(self.agents, self.targets)):
                self.robot.x_direction(
                    a, i, self.num_agents, self.xmax, self.x, self.vx, self.time
                )
                self.robot.y_direction(
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


engine = Continuous().environment(plot=True)
