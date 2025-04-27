import argparse
import os

import rnl as vault


def main(arg):
    wandb_key = os.environ.get("WANDB_API_KEY")
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    # 1.step -> config robot
    param_robot = vault.robot(
        base_radius=0.105,
        vel_linear=[0.0, 0.22],
        vel_angular=[1.0, 2.84],
        wheel_distance=0.16,
        weight=1.0,
        threshold=0.10,  # 4 # 0.03
        collision=0.10,  # 2 # 0.075
        noise=True,
        path_model="",
    )

    # 2.step -> config sensors [for now only lidar sensor!!]
    param_sensor = vault.sensor(
        fov=270,
        num_rays=5,  # min 5 max 20
        min_range=0.0,
        max_range=20.5,  # 3.5
    )

    # 3.step -> config env
    param_env = vault.make(
        scalar=arg.scalar,
        folder_map="./data/map6",
        name_map="map6",
        max_timestep=1000000,  # 1000
        type=args.type,
        grid_size=[2.2, 2.15]
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
            pretrained=args.pretrained,
            use_agents=args.agents,
            max_timestep_global=args.max_timestep_global,
            seed=args.seed,
            hidden_size=list(map(int, args.hidden_size.split(","))),
            type_model=args.type_model,
            activation=args.activation,
            batch_size=args.batch_size,
            num_envs=args.num_envs,
            device=args.device,
            checkpoint=args.checkpoint,
            checkpoint_path=args.checkpoint_path,
            use_wandb=args.use_wandb,
            wandb_api_key=str(wandb_key),
            llm_api_key=str(gemini_api_key),
            lr=args.lr,
            learn_step=args.learn_step,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            update_epochs=args.update_epochs,
            name=args.name,
            verbose=args.verbose,
            policy_type=args.policy_type,
        )

    elif args.mode == "sim":
        # 5.step -> config train robot
        model = vault.Simulation(param_robot, param_sensor, param_env, param_render, type=args.type)
        # 6.step -> run robot
        model.run()

    elif args.mode == "run":
        model = vault.Probe(
            num_envs=args.num_envs,
            max_steps=args.max_timestep_global,
            robot_config=param_robot,
            sensor_config=param_sensor,
            env_config=param_env,
            render_config=param_render,
            seed=args.seed,
        )

        model.execute()


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
    parser.add_argument("mode", choices=["learn", "sim", "run"], help="Mode")

    parser.add_argument(
        "--pretrained",
        type=str,
    )

    parser.add_argument(
        "--agents",
        type=str2bool,
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
        type=int,
    )

    parser.add_argument(
        "--checkpoint_path",
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
        "--verbose",
        type=str2bool,
    )

    parser.add_argument(
        "--scalar",
        type=int,
    )

    parser.add_argument(
        "--use_wandb",
        type=str2bool,
    )

    parser.add_argument(
        "--type",
        type=str,
    )

    parser.add_argument(
        "--obstacle_percentage",
        type=float,
    )

    parser.add_argument(
        "--map_size",
        type=float,
    )

    parser.add_argument(
        "--policy_type",
        type=str,
    )

    parser.add_argument(
        "--type_model",
        type=str,
    )

    args = parser.parse_args()
    main(args)
