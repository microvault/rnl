import numpy as np
import pytest

import rnl as vault


@pytest.fixture
def setup_parameters():
    param_robot = vault.robot(
        base_radius=0.033,
        vel_linear=[0.0, 2.0],
        vel_angular=[1.0, 2.0],
        wheel_distance=0.16,
        weight=1.0,
        threshold=0.01,
    )

    param_sensor = vault.sensor(
        fov=4 * np.pi,
        num_rays=20,
        min_range=0.0,
        max_range=6.0,
    )

    param_env = vault.make(
        map_file="None",
        random_mode="normal",
        timestep=1000,
        grid_dimension=3,
        friction=0.4,
        porcentage_obstacles=0.1,
        max_step=1000,
    )

    return param_robot, param_sensor, param_env


def test_robot_parameters(setup_parameters):
    param_robot, _, _ = setup_parameters
    assert param_robot.base_radius == 0.033
    assert param_robot.vel_linear == [0.0, 2.0]
    assert param_robot.vel_angular == [1.0, 2.0]
    assert param_robot.wheel_distance == 0.16
    assert param_robot.weight == 1.0
    assert param_robot.threshold == 0.01


def test_sensor_parameters(setup_parameters):
    _, param_sensor, _ = setup_parameters
    assert param_sensor.fov == 4 * np.pi
    assert param_sensor.num_rays == 20
    assert param_sensor.min_range == 0.0
    assert param_sensor.max_range == 6.0


def test_environment_parameters(setup_parameters):
    _, _, param_env = setup_parameters
    assert param_env.map_file == "None"
    assert param_env.random_mode == "normal"
    assert param_env.timestep == 1000
    assert param_env.grid_dimension == 3
    assert param_env.friction == 0.4
    assert param_env.porcentage_obstacles == 0.1
    assert param_env.max_step == 1000


def test_trainer_learning(setup_parameters):
    param_robot, param_sensor, param_env = setup_parameters
    model = vault.Trainer(param_robot, param_sensor, param_env, pretrained_model=False)

    result = model.learn(
        max_timestep=10,
        memory_size=1000,
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
        num_envs=1,
        device="cpu",
        learn_step=10,
        target_score=200,
        max_steps=10,
        evaluation_steps=10,
        evaluation_loop=1,
        learning_delay=0,
        n_step_memory=1,
        checkpoint=100,
        checkpoint_path="checkpoints",
        overwrite_checkpoints=False,
        use_wandb=False,
        wandb_api_key="",
        accelerator=False,
        use_mutation=True,
        freq_evolution=10,
        population_size=2,
        no_mutation=0.4,
        arch_mutation=0.2,
        new_layer=0.2,
        param_mutation=0.2,
        active_mutation=0.0,
        hp_mutation=0.2,
        hp_mutation_selection=["lr", "batch_size"],
        mutation_strength=0.1,
        evolution_steps=10,
        save_elite=False,
        elite_path="elite",
        tourn_size=1,
        elitism=True,
        hidden_size=[60, 40],
    )

    assert result is None


if __name__ == "__main__":
    pytest.main()
