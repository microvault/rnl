import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import art3d
from shapely.geometry import Point, Polygon

from .engine.collision import filter_segment, lidar_intersections
from .generate import Generator
from .robot import Robot

# car racing
# - 8 m/s
# - 0 m/s
# - radius = 56 cm
# - peso = 2.64 kg


class Continuous:
    def __init__(
        self,
        time: int = 100,  # max step
        size: float = 3.0,  # size robot
        fps: int = 10,  # 10 frames per second
        random: float = 1e20,  # 100 random points
        max_speed: float = 0.6,  # 0.2 m/s
        min_speed: float = 0.5,  # 0.1 m/s
        num_rays: int = 10,  # num range lidar
        max_range: int = 6,  # max range
        grid_lenght: int = 20,  # TODO: error < 5 -> [5 - 15]
    ):
        self.time = time
        self.size = size
        self.fps = fps
        self.num_rays = num_rays
        self.max_range = max_range
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

    def init_animation(self, ax) -> None:
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

    def _ray_casting(self, poly: Polygon, x: float, y: float) -> bool:
        return poly.contains(Point(x, y))

    def _get_label(self, timestep: int) -> str:
        line1 = "Environment\n"
        line2 = "Time Step:".ljust(14) + f"{timestep:4.0f}\n"
        return line1 + line2

    def reset(self) -> None:
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

    def step(self, i: int) -> None:
        self.robot.x_advance(i, self.x, self.vx)
        self.robot.y_advance(i, self.y, self.vy)

        seg = filter_segment(self.segments, self.x[i], self.y[i], 6)

        if hasattr(self, "laser_scatters"):
            for scatter in self.laser_scatters:
                scatter.remove()
            del self.laser_scatters

        lidar_angles = np.linspace(0, 2 * np.pi, self.num_rays)
        intersections, measurements = lidar_intersections(
            self.x[i], self.y[i], self.max_range, lidar_angles, seg
        )

        self.laser_scatters = []
        for angle, intersection in zip(lidar_angles, intersections):
            if intersection is not None:
                scatter = plt.scatter(
                    intersection[0], intersection[1], color="g", s=0.5
                )
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

    def show(self, plot: bool = False) -> None:

        if plot:

            # TODO
            # fig2 = plt.figure()
            # ax2 = fig2.add_subplot(111)
            # ax2.plot([1, 2, 3, 4], [1, 4, 9, 16])

            ani = animation.FuncAnimation(
                self.fig,
                self.step,
                init_func=self.reset,
                blit=False,
                frames=self.time,
                interval=self.fps,
            )
            plt.show()


# env = Continuous()
# env.show(plot=True)
