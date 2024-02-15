import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import art3d


class Continuous:
    def __init__(self, plot=True, fig_width=6, fig_height=6):
        self.fig_width = fig_width
        self.fig_height = fig_height
        self.plot = plot

    def plot_initial_environment(self):

        fig, ax = plt.subplots(1, 1, figsize=(self.fig_width, self.fig_height))
        ax.remove()
        ax = fig.add_subplot(1, 1, 1, projection="3d")

        # Bounds
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(-0.01, 0.01)

        # Surface
        corner_points = [(0, 0), (0, 1), (1, 1), (1, 0)]

        poly = Polygon(corner_points, color=(0.1, 0.2, 0.5, 0.15))
        ax.add_patch(poly)
        art3d.pathpatch_2d_to_3d(poly, z=0, zdir="z")

        # "Hide" side panes
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
        ax.elev = 40
        ax.azim = -55
        ax.dist = 10

        # Try to reduce whitespace
        fig.subplots_adjust(left=0, right=1, bottom=-0.2, top=1)

        if self.plot:
            plt.show()
        else:
            return fig
