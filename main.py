import argparse
import os

import rnl as vault


def main(arg):
    wandb_key = os.environ.get("WANDB_API_KEY")
    # 1.step -> config robot
    param_robot = vault.robot(
        base_radius=0.105,
        vel_linear=[0.0, 0.22],
        vel_angular=[1.0, 2.84],
        wheel_distance=0.16,
        weight=1.0,
        threshold=0.3,  # 4
        collision=0.075,  # 2
        path_model="None",
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
        scalar=arg.scalar,
        grid_length=2,
        folder_map="",  # ./data/map4
        name_map="",  # map4
        max_timestep=1000,
        mode="easy-01",  # easy-00, easy-01, easy-02, medium
        reward_function=args.type_reward,  # [time, distance, orientation, obstacle, all, any, distance_orientation, distance_time, orientation_time, distance_orientation_time, distance_obstacle, orientation_obstacle]
    )

    # 4.step -> config render
    param_render = vault.render(controller=arg.controller, debug=arg.debug, plot=False)

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
            max_timestep_global=args.max_timestep_global,
            seed=args.seed,
            hidden_size=list(map(int, args.hidden_size.split(","))),
            activation=args.activation,
            batch_size=args.batch_size,
            num_envs=args.num_envs,
            device=args.device,
            checkpoint=args.checkpoint,
            use_wandb=True,
            wandb_api_key=str(wandb_key),
            lr=args.lr,
            learn_step=args.learn_step,
            gae_lambda=args.gae_lambda,
            action_std_init=args.action_std_init,
            clip_coef=args.clip_coef,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            update_epochs=args.update_epochs,
            name=args.name,
        )

    elif args.mode == "sim":
        # 5.step -> config train robot
        model = vault.Simulation(param_robot, param_sensor, param_env, param_render)
        # 6.step -> run robot
        model.run()

    elif args.mode == "run":
        model = vault.Probe(
            num_envs=10,
            max_steps=1000,
            robot_config=param_robot,
            sensor_config=param_sensor,
            env_config=param_env,
            render_config=param_render,
        )

        model.execute()

    elif args.mode == "training":
        model = vault.Probe(
            num_envs=10,
            max_steps=100,
            robot_config=param_robot,
            sensor_config=param_sensor,
            env_config=param_env,
            render_config=param_render,
        )

        model.training()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or setup environment.")
    parser.add_argument(
        "mode", choices=["learn", "sim", "run", "training"], help="Mode"
    )

    parser.add_argument(
        "--max_timestep_global",
        type=int,
    )

    parser.add_argument(
        "--seed",
        type=int,
    )

    parser.add_argument(
        "--hidden_size",
        type=str,
    )

    parser.add_argument(
        "--activation",
        type=str,
        choices=["LeakyReLU", "ReLU"],
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--num_envs",
        type=int,
    )

    parser.add_argument(
        "--device",
        type=str,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
    )

    parser.add_argument(
        "--lr",
        type=float,
    )

    parser.add_argument(
        "--learn_step",
        type=int,
    )

    parser.add_argument(
        "--gae_lambda",
        type=float,
    )

    parser.add_argument(
        "--action_std_init",
        type=float,
    )

    parser.add_argument(
        "--clip_coef",
        type=float,
    )

    parser.add_argument(
        "--ent_coef",
        type=float,
    )

    parser.add_argument(
        "--vf_coef",
        type=float,
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
    )

    parser.add_argument(
        "--update_epochs",
        type=int,
    )

    parser.add_argument(
        "--name",
        type=str,
    )

    parser.add_argument(
        "--controller",
        type=str2bool,
    )

    parser.add_argument(
        "--debug",
        type=str2bool,
    )

    parser.add_argument(
        "--scalar",
        type=int,
    )

    parser.add_argument(
        "--type_reward",
        type=str,
        choices=[
            "time",
            "distance",
            "orientation",
            "obstacle",
            "all",
            "any",
            "distance_orientation",
            "distance_time",
            "orientation_time",
            "distance_orientation_time",
            "distance_obstacle",
            "orientation_obstacle",
        ],
    )

    args = parser.parse_args()
    main(args)
