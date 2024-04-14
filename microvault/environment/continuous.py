import timeit

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import art3d
from shapely.geometry import Point

from .generate import Generator
from .robot import Robot


class Continuous:
    def __init__(
        self,
        time=1000,
        size=3,
        frame=1,
        random=4000,
        max_speed=0.5,
        min_speed=0.4,
        grid_lenght: int = 15,  # TODO: error < 5
    ):
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

        self.fov = -90 * np.pi / 180

        # TODO: remove the team and remove in array format
        self.target_x = 0
        self.target_y = 0

        self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 6))

        self.generator = Generator(grid_lenght=grid_lenght, random=self.random)
        self.robot = Robot(self.time, 1, 3, self.grid_lenght, self.grid_lenght)

        self.ax.remove()
        self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")

        (
            self.x,
            self.y,
            self.sp,
            self.theta,
            self.vx,
            self.vy,
            self.radius,
        ) = self.robot.init_agent(self.ax)

        self.target = self.ax.plot3D(
            np.random.uniform(0, self.xmax),
            np.random.uniform(0, self.ymax),
            0,
            marker="x",
            markersize=self.size,
        )[0]

        self.agent = self.ax.plot3D(
            self.x, self.y, 0, marker="o", markersize=self.radius
        )[0]

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
            0.05,
            self._get_label(0),
        )

        self.label.set_fontsize(14)
        self.label.set_fontweight("normal")
        self.label.set_color("#666666")

        self.fig.subplots_adjust(left=0, right=1, bottom=0.1, top=1)

    def _ray_casting(self, poly, x, y) -> bool:
        return poly.contains(Point(x, y))

    def change_advance(self):
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

        self.target_x = np.random.uniform(0, self.xmax)
        self.target_y = np.random.uniform(0, self.ymax)

        self.x[0] = np.random.uniform(0, self.xmax)
        self.y[0] = np.random.uniform(0, self.ymax)

        target_inside = False

        while not target_inside:
            self.target_x = np.random.uniform(0, self.xmax)
            self.target_y = np.random.uniform(0, self.ymax)

            self.x[0] = np.random.uniform(0, self.xmax)
            self.y[0] = np.random.uniform(0, self.ymax)

            if self._ray_casting(
                poly, self.target_x, self.target_y
            ) and self._ray_casting(poly, self.x[0], self.y[0]):
                target_inside = True

        self.sp[0] = np.random.uniform(self.min_speed, self.max_speed)
        self.theta[:] = np.random.uniform(0, 2 * np.pi)
        self.vx[0] = self.sp[0] * np.cos(self.theta[0])
        self.vy[0] = self.sp[0] * np.sin(self.theta[0])

    def step(self, i):
        if hasattr(self, "laser_scatters"):
            for scatter in self.laser_scatters:
                scatter.remove()
            del self.laser_scatters

        self.robot.x_advance(i, self.x, self.vx)
        self.robot.y_advance(i, self.y, self.vy)

        segments_trans = [
            [np.array(segment[:2]), np.array(segment[2:])] for segment in self.segments
        ]

        lidar_range = 6
        num_rays = 20
        lidar_angles = np.linspace(0, 2 * np.pi, num_rays)

        intersections = self.robot.lidar_intersections(
            self.x[i], self.y[i], lidar_range, lidar_angles, segments_trans
        )

        # Plotar as novas leituras do laser
        self.laser_scatters = []
        for angle, intersection in zip(lidar_angles, intersections):
            if intersection is not None:
                scatter = plt.scatter(
                    intersection[0], intersection[1], color="g", s=0.5
                )  # Usando scatter para os raios do LiDAR
                self.laser_scatters.append(scatter)

        self.agent.set_data_3d(
            [self.x[i]],
            [self.y[i]],
            [0],
        )

        self.target.set_data_3d(
            [self.target_x],
            [self.target_y],
            [0],
        )

        self.label.set_text(self._get_label(i))

    def timer(self):
        tempo_decorrido = timeit.timeit(self.reset, number=1)
        print("Tempo decorrido:", tempo_decorrido, "segundos")

    def _get_label(self, timestep):
        line1 = "Environment\n"
        line2 = "Time Step:".ljust(14) + f"{timestep:4.0f}\n"
        return line1 + line2

    def show(self, plot=False):

        if plot:

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
