import argparse
import multiprocessing as mp
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
        threshold=1.0, # 4
        collision=1.0, # 2
        path_model="",
    )

    # 2.step -> config sensors [for now only lidar sensor!!]
    param_sensor = vault.sensor(
        fov=270,
        num_rays=5,
        min_range=1.0,
        max_range=90.0,
    )

    # 3.step -> config env
    param_env = vault.make(
        folder_map="./data/map4",
        name_map="map4",
        max_timestep=1000,
        mode="easy-01" # easy-01, medium
    )

    # 4.step -> config render
    param_render = vault.render(controller=False, debug=False)

    if args.mode == "learn":
        # 5.step -> config train robot
        model = vault.Trainer(
            param_robot,
            param_sensor,
            param_env,
            param_render,
            pretrained_model=False,
            train_docker=True,
            probe=False,
        )

        # 6.step -> train robot
        model.learn(
            max_timestep_global=1000000,
            gamma=0.99,
            batch_size=1024,
            lr=0.0001,
            num_envs=100,
            device="cuda",
            learn_step=1024,
            checkpoint=100000,
            checkpoint_path="./checkpoints/model",
            overwrite_checkpoints=False,
            use_mutation=False,
            population_size=10,
            no_mutation=0.4,
            arch_mutation=0.2,
            new_layer=0.2,
            param_mutation=0.2,
            active_mutation=0.2,
            hp_mutation=0.2,
            hp_mutation_selection=["lr", "batch_size", "learn_step"],
            mutation_strength=0.1,
            save_elite=True,
            elite_path="./checkpoints/elite",
            tourn_size=2,
            elitism=True,
            hidden_size=[128, 128],
            use_wandb=True,
            wandb_api_key=str(wandb_key),
            min_lr=0.0001,
            max_lr=0.01,
            min_learn_step=256,
            max_learn_step=8192,
            min_batch_size=128,
            max_batch_size=1024,
            evo_steps=10000,
            eval_steps=None,
            eval_loop=3,
            mutate_elite=True,
            rand_seed=42,
            activation=["ReLU", "ELU", "GELU"],
            mlp_activation="ReLU",
            mlp_output_activation="ReLU",
            min_hidden_layers=1,
            max_hidden_layers=4,
            min_mlp_nodes=32,
            max_mlp_nodes=256,
            gae_lambda=0.95,
            action_std_init=0.6,
            clip_coef=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            update_epochs=4,
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
            csv_file="./data/debugging.csv", # ./data/debugging.csv
            num_envs=20,
            max_steps=1000,
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
