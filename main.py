import argparse
import os

import rnl as vault


def main(arg):
    wandb_key = os.environ.get("WANDB_API_KEY")
    # algo = str(os.environ.get("ALGO"))
    # 1.step -> config robot
    param_robot = vault.robot(
        base_radius=0.105,
        vel_linear=[0.0, 0.22],
        vel_angular=[1.0, 2.84],
        wheel_distance=0.16,
        weight=1.0,
        threshold=1.0,  # 4
        collision=0.5,  # 2
        path_model="None",
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
        folder_map="None",  # ./data/map4
        name_map="None",
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
        )

        # 6.step -> train robot
        model.learn(
            algorithm="PPO",
            max_timestep_global=1000000,
            seed=42,
            buffer_size=1000000,
            hidden_size=[256, 256],
            activation="LeakyReLU",  # LeakyReLU, ReLU
            batch_size=1024,
            num_envs=4,
            device="cuda",
            checkpoint_path="model_29_01_2025",
            use_wandb=True,
            wandb_api_key=str(wandb_key),
            lr=0.0003,
            learn_step=512,
            gae_lambda=0.95,
            action_std_init=0.6,
            clip_coef=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            update_epochs=10,
            name="rnl_29_01_2025",
        )

    elif args.mode == "sim":
        # 5.step -> config train robot
        model = vault.Simulation(param_robot, param_sensor, param_env, param_render)
        # 6.step -> run robot
        model.run()

    elif args.mode == "run":
        model = vault.Probe(
            num_envs=4,
            max_steps=1000,
            robot_config=param_robot,
            sensor_config=param_sensor,
            env_config=param_env,
            render_config=param_render,
        )

        model.execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or setup environment.")
    parser.add_argument(
        "mode", choices=["learn", "sim", "run"], help="Mode to run: 'train' or 'run'"
    )

    args = parser.parse_args()
    main(args)
