import argparse
import multiprocessing as mp
import rnl as vault
import os


def main(arg):
    wandb_key = os.environ.get("WANDB_API_KEY")
    # 1.step -> config robot
    param_robot = vault.robot(
        base_radius=20.0,  # (centimeters)
        vel_linear=[0.0, 2.0],  # [min, max]
        vel_angular=[1.0, 2.0],  # [min, max]
        wheel_distance=0.16,  # (centimeters)
        weight=1.0,  # (kilograms)
        threshold=0.05,  # (centimeters)
        path_model="./",
    )

    # 2.step -> config sensors [for now only lidar sensor!!]
    param_sensor = vault.sensor(
        fov=180,
        num_rays=40,
        min_range=1.0,
        max_range=20.0,
    )

    # 3.step -> config env
    param_env = vault.make(
        folder_map="None", #"/Users/nicolasalan/microvault/rnl/data/map",
        name_map="None",
        random_mode="hard",  # hard, normal
        max_timestep=1000,
        grid_dimension=5,
        friction=0.4,
        porcentage_obstacles=0.1,
        randomization_interval=100,
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
            num_envs=16,
            device="cpu",
            learn_step=10,
            target_score=200,
            max_steps=1000000,
            evaluation_steps=10000,
            evaluation_loop=1,
            learning_delay=2,
            n_step_memory=3,
            checkpoint=100000,
            checkpoint_path="checkpoints",
            overwrite_checkpoints=False,
            use_mutation=True,
            freq_evolution=10000,
            population_size=4,
            no_mutation=0.4,
            arch_mutation=0.2,
            new_layer=0.2,
            param_mutation=0.2,
            active_mutation=0.0,
            hp_mutation=0.2,
            hp_mutation_selection=["lr", "batch_size", "learn_step"],
            mutation_strength=0.1,
            evolution_steps=10000,
            save_elite=False,
            elite_path="elite",
            tourn_size=2,
            elitism=True,
            hidden_size=[800, 600],
            save=True,
            use_wandb=True,
            wandb_api_key=str(wandb_key)
        )


    else:
        # 4.step -> config render
        param_render = vault.render(
            fps=1, controller=True, rgb_array=True, data_colletion=False
        )

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
