import argparse
import multiprocessing as mp

import numpy as np

import rnl as vault


def main(arg):
    # 1.step -> config robot
    param_robot = vault.robot(
        base_radius=0.033,  # (centimeters) # TODO: RANDOMIZE
        vel_linear=[0.0, 2.0],  # [min, max] # TODO: RANDOMIZE
        vel_angular=[1.0, 2.0],  # [min, max] # TODO: RANDOMIZE
        wheel_distance=0.16,  # (centimeters) # TODO: RANDOMIZE
        weight=1.0,  # (kilograms) # TODO: RANDOMIZE
        threshold=0.01,  # (centimeters) # TODO: RANDOMIZE
    )

    # 2.step -> config sensors [for now only lidar sensor!!]
    param_sensor = vault.sensor(
        fov=4 * np.pi,  # TODO: RANDOMIZE
        num_rays=20,  # TODO: RANDOMIZE
        min_range=0.0,  # TODO: RANDOMIZE
        max_range=6.0,  # TODO: RANDOMIZE
    )

    # 3.step -> config env
    param_env = vault.make(
        map_file="None",  # TODO: RANDOMIZE
        random_mode="normal",  # hard, normal
        timestep=1000,  # TODO: RANDOMIZE
        grid_dimension=3,  # TODO: RANDOMIZE
        friction=0.4,  # TODO: RANDOMIZE
        porcentage_obstacles=0.1,  # TODO: RANDOMIZE
        max_step=1000,  # TODO: RANDOMIZE
    )

    if args.mode == "train":
        # 4.step -> config train robot
        model = vault.Trainer(
            param_robot, param_sensor, param_env, pretrained_model=False
        )
        # 5.step -> train robot
        model.learn(
            max_timestep=800000,
            memory_size=1000000,
            gamma=0.99,
            n_step=3,
            alpha=0.6,
            beta=0.4,
            tau=0.001,
            prior_eps=0.000001,
            num_atoms=51,
            v_min=-200,
            v_max=200,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.995,
            batch_size=64,
            lr=0.0001,
            seed=1,
            num_envs=2,
            device="cpu",
            learn_step=10,
            target_score=200,
            max_steps=1000000,
            evaluation_steps=10000,
            evaluation_loop=1,
            learning_delay=0,
            n_step_memory=1,
            checkpoint=1000,
            checkpoint_path="checkpoints",
            overwrite_checkpoints=False,
            use_wandb=False,
            wandb_api_key="",
            accelerator=False,
            use_mutation=True,
            freq_evolution=10000,
            population_size=6,
            no_mutation=0.4,
            arch_mutation=0.2,
            new_layer=0.2,
            param_mutation=0.2,
            active_mutation=0.0,
            hp_mutation=0.2,
            hp_mutation_selection=["lr", "batch_size"],
            mutation_strength=0.1,
            evolution_steps=10000,
            save_elite=False,
            elite_path="elite",
            tourn_size=2,
            elitism=True,
            hidden_size=[800, 600],
        )
    else:
        # 4.step -> config render
        param_render = vault.render(fps=1, controller=False, rgb_array=True)

        # 5.step -> config train robot
        model = vault.Trainer(
            param_robot, param_sensor, param_env, param_render, pretrained_model=False
        )
        # 6.step -> run robot
        model.run()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    parser = argparse.ArgumentParser(description="Train or setup environment.")
    parser.add_argument(
        "mode", choices=["train", "run"], help="Mode to run: 'train' or 'run'"
    )

    # Parse arguments
    args = parser.parse_args()
    main(args)
