import rnl as vault


def main():
    # 1.step -> config robot
    param_robot = vault.robot(
        base_radius=0.105,
        vel_linear=[0.0, 0.22],
        vel_angular=[1.0, 2.84],
        wheel_distance=0.16,
        weight=1.0,
        threshold=0.3,  # 4
        collision=0.075,  # 2
        path_model="",
    )

    # 2.step -> config sensors [for now only lidar sensor!!]
    param_sensor = vault.sensor(
        fov=270,
        num_rays=5,  # min 5 max 20
        min_range=0.0,
        max_range=3.5,  # 3.5
    )

    # 3.step -> config env
    param_env = vault.make(
        scalar=50,
        folder_map="",  # ./data/map4
        name_map="",  # map4
        max_timestep=1000,
    )

    # 4.step -> config render
    param_render = vault.render(controller=False, debug=True)

    # 5.step -> config sim robot
    sim = vault.Simulation(param_robot, param_sensor, param_env, param_render)
    # 6.step -> run robot
    sim.run()


if __name__ == "__main__":
    main()
