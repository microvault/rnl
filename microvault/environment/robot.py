class Robot:
    def __init__(
        self,
        type="robot",
        size=10,
        speed=1,
        position_x=0,
        position_y=0,
        theta=0,
        vx=0,
        vy=0,
        acceleration=0,
        fov=90,
        control_hz=10,
        laser_max=10,
        laser_min=0,
        torque=0,
    ):
        self.type = type
        self.size = size
        self.speed = speed
        self.position_x = position_x
        self.position_y = position_y
        self.theta = theta
        self.vx = vx
        self.vy = vy
        self.acceleration = acceleration
        self.fov = fov
        self.control_hz = control_hz
        self.laser_max = laser_max
        self.laser_min = laser_min
        self.torque = torque

    def x_direction(self, agents, i, num_agents, xmax, x, vx, time) -> None:
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

    def y_direction(self, agents, i, num_agents, ymax, y, vy, time) -> None:
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

    def random_robot() -> None:
        pass
