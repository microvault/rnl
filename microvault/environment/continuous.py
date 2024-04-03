import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import art3d
from shapely.geometry import Point
from typing_extensions import ParamSpecArgs

from .generate import Generator
from .robot import Robot


class Continuous:
    def __init__(
        self,
        n=5,
        time=10,
        size=3,
        frame=100,
        random=300,
        max_speed=0.5,
        min_speed=0.4,
        grid_lenght=5,  # TODO: error < 5
    ):
        self.num_agents = n
        self.time = time
        self.size = size
        self.frame = frame
        self.max_speed = max_speed
        self.min_speed = min_speed

        self.random = random

        self.grid_lenght = grid_lenght

        self.xmax = grid_lenght
        self.ymax = grid_lenght

        self.segments = None

        # TODO: remove the team and remove in array format
        self.target_x = np.zeros((self.num_agents, self.time))
        self.target_y = np.zeros((self.num_agents, self.time))

        # self.agents = [None for _ in range(self.num_agents)]
        self.targets = [None for _ in range(self.num_agents)]

        self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 6))

        self.generator = Generator(grid_lenght=grid_lenght, random=self.random)
        self.robots = Robot(
            self.num_agents, self.time, 1, 3, self.grid_lenght, self.grid_lenght
        )

        self.ax.remove()
        self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")

        (
            self.x,
            self.y,
            self.sp,
            self.theta,
            self.vx,
            self.vy,
            self.agents,
            self.radius,
        ) = self.robots.init_agent(self.ax)

        self.init_animation(self.ax)

    def init_animation(self, ax):
        ax.set_xlim(0, self.grid_lenght)
        ax.set_ylim(0, self.grid_lenght)

        # ------ Create wordld ------ #

        path, poly, seg = self.generator.world()

        ax.add_patch(path)

        art3d.pathpatch_2d_to_3d(path, z=0, zdir="z")

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

        self.label = self.ax.text(
            0,
            0,
            0.6,
            self._get_label(0),
        )

        self.label.set_fontsize(14)
        self.label.set_fontweight("normal")
        self.label.set_color("#666666")

        self.fig.subplots_adjust(left=0, right=1, bottom=0.1, top=1)

        for a in range(0, self.num_agents):
            self.targets[a] = self.ax.plot3D(
                np.random.uniform(0, self.xmax),
                np.random.uniform(0, self.ymax),
                0,
                marker="x",
                markersize=self.size,
            )[0]

    def _ray_casting(self, poly, x, y) -> bool:
        return poly.contains(Point(x, y))

    def change_advance(self):
        # Aqui você pode implementar a lógica para mudar a direção do robô após a colisão
        # Por exemplo, você pode inverter a direção ou aplicar outra lógica adequada ao seu problema
        new_vx = -self.vx
        new_vy = -self.vy
        return new_vx, new_vy

    def reset(self):

        for patch in self.ax.patches:
            patch.remove()

        new_map_path, poly, seg = self.generator.world()
        self.segments = seg

        self.ax.add_patch(new_map_path)
        art3d.pathpatch_2d_to_3d(new_map_path, z=0, zdir="z")

        for a in range(self.num_agents):

            self.target_x[a, 0] = np.random.uniform(0, self.xmax)
            self.target_y[a, 0] = np.random.uniform(0, self.ymax)

            self.x[a, 0] = np.random.uniform(0, self.xmax)
            self.y[a, 0] = np.random.uniform(0, self.ymax)

            target_inside = False

            while not target_inside:
                self.target_x[a, 0] = np.random.uniform(0, self.xmax)
                self.target_y[a, 0] = np.random.uniform(0, self.ymax)

                self.x[a, 0] = np.random.uniform(0, self.xmax)
                self.y[a, 0] = np.random.uniform(0, self.ymax)

                if self._ray_casting(
                    poly, self.target_x[a, 0], self.target_y[a, 0]
                ) and self._ray_casting(poly, self.x[a, 0], self.y[a, 0]):
                    target_inside = True

            self.sp[a, 0] = np.random.uniform(self.min_speed, self.max_speed)
            self.theta[a, :] = np.random.uniform(0, 2 * np.pi)
            self.vx[a, 0] = self.sp[a, 0] * np.cos(self.theta[a, 0])
            self.vy[a, 0] = self.sp[a, 0] * np.sin(self.theta[a, 0])

    def step(self, i):
        for a, (agent, target) in enumerate(zip(self.agents, self.targets)):
            self.robots.x_advance(a, i, self.x, self.vx)
            self.robots.x_advance(a, i, self.y, self.vy)

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

        self.label.set_text(self._get_label(i))

    def _get_label(self, timestep):
        line1 = "Environment\n"
        line2 = "Time Step:".ljust(14) + f"{timestep:4.0f}\n"
        return line1 + line2

    def show(self, plot=False):

        if plot == True:

            ani = animation.FuncAnimation(
                self.fig,
                self.step,
                init_func=self.reset,
                blit=False,
                frames=self.time,
                interval=self.frame,
            )
            plt.show()


# env = Continuous()
# env.show(plot=True)
