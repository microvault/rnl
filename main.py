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
        threshold=4.0,
        collision=2.0,
        path_model="",
    )

    # 2.step -> config sensors [for now only lidar sensor!!]
    param_sensor = vault.sensor(
        fov=270,
        num_rays=5,
        min_range=1.0,
        max_range=60.0,
    )

    # 3.step -> config env
    param_env = vault.make(
        folder_map="/Users/nicolasalan/microvault/rnl/data/map4",
        name_map="map4",
        max_timestep=2000,
    )

    if args.mode == "train":
        # 4.step -> config train robot
        model = vault.Trainer(
            param_robot, param_sensor, param_env, pretrained_model=False
        )

        # 5.step -> train robot
        model.learn(
            max_timestep=1000,  #
            memory_size=1000,
            gamma=0.99,
            n_step=1,
            alpha=0.6,
            beta=0.4,
            tau=0.001,
            prior_eps=0.000001,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.995,
            batch_size=4,
            lr=0.0001,
            seed=1,
            num_envs=20,
            device="cpu",
            learn_step=10,
            target_score=200,
            max_steps=1000,
            evaluation_steps=100,
            evaluation_loop=1,
            learning_delay=2,
            n_step_memory=3,
            checkpoint=1,
            checkpoint_path="model",
            overwrite_checkpoints=False,
            use_mutation=True,
            freq_evolution=100,
            population_size=2,
            no_mutation=0.4,
            arch_mutation=0.2,
            new_layer=0.2,
            param_mutation=0.2,
            active_mutation=0.0,
            hp_mutation=0.2,
            hp_mutation_selection=["lr", "batch_size", "learn_step"],
            mutation_strength=0.1,
            evolution_steps=100,
            save_elite=False,
            elite_path="elite",
            tourn_size=2,
            elitism=True,
            hidden_size=[32, 32],
            use_wandb=False,
            wandb_api_key=str(wandb_key),
            eps_start=1.0,
            eps_end=0.1,
            eps_decay=0.995,
            noise_std=0.5,
            per=True,
            min_lr=0.001,
            max_lr=0.01,
            min_learn_step=1,
            max_learn_step=120,
            min_batch_size=4,
            max_batch_size=1024,
            evo_steps=5000,
            eval_steps=10,
            eval_loop=10,
            mutate_elite=True,
            rand_seed=1,
            activation=["ReLU"],
            mlp_activation="ReLU",
            mlp_output_activation="ReLU",
            min_hidden_layers=2,
            max_hidden_layers=4,
            min_mlp_nodes=64,
            max_mlp_nodes=800,
        )

    else:
        # 4.step -> config render
        param_render = vault.render(controller=False)

        # 5.step -> config train robot
        model = vault.Simulation(
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

    args = parser.parse_args()
    main(args)
