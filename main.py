import argparse
import multiprocessing as mp
import os
import rnl as vault


def main(arg):
    wandb_key = os.environ.get("WANDB_API_KEY")
    algo = str(os.environ.get("ALGO"))
    # 1.step -> config robot
    param_robot = vault.robot(
        base_radius=0.105,
        vel_linear=[0.0, 0.22],
        vel_angular=[1.0, 2.84],
        wheel_distance=0.16,
        weight=1.0,
        threshold=1.0,  # 4
        collision=0.5,  # 2
        path_model="",
    )

    # 2.step -> config sensors [for now only lidar sensor!!]
    param_sensor = vault.sensor(
        fov=270,
        num_rays=5,
        min_range=1.0,
        max_range=12.0,
    )

    # 3.step -> config env
    param_env = vault.make(
        folder_map="./data/map4",
        name_map="map4",
        max_timestep=10000,
        mode="easy-01",  # easy-01, medium
    )

    # 4.step -> config render
    param_render = vault.render(controller=False, debug=True, plot=False)

    if args.mode == "learn":
        # 5.step -> config train robot
        model = vault.Trainer(
            param_robot,
            param_sensor,
            param_env,
            param_render,
            pretrained_model=False,
        )

        # 6.step -> train robot
        model.learn(
            algorithms=algo,
            max_timestep_global=5000000,
            seed=42,
            num_envs=2,
            device="cpu",
            checkpoint=100000,
            checkpoint_path="./checkpoints/model",
            population_size=5,
            use_wandb=True,
            wandb_api_key=str(wandb_key),
        )

    elif args.mode == "sim":
        # 5.step -> config train robot
        model = vault.Simulation(
            param_robot, param_sensor, param_env, param_render, pretrained_model=False
        )
        # 6.step -> run robot
        model.run()

    elif args.mode == "run":
        model = vault.Probe(
            csv_file="./data/debugging.csv",
            num_envs=2,
            max_steps=100000,
            robot_config=param_robot,
            sensor_config=param_sensor,
            env_config=param_env,
            render_config=param_render,
            pretrained_model=False,
        )

        model.execute()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    parser = argparse.ArgumentParser(description="Train or setup environment.")
    parser.add_argument(
        "mode", choices=["learn", "sim", "run"], help="Mode to run: 'train' or 'run'"
    )

    args = parser.parse_args()
    main(args)
