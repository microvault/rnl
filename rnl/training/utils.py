import gymnasium as gym
import numpy as np

from rnl.algorithms.rainbow import RainbowDQN
from rnl.configs.config import (
    AgentConfig,
    EnvConfig,
    RenderConfig,
    RobotConfig,
    SensorConfig,
    TrainerConfig,
)
from rnl.environment.environment_navigation import NaviEnv


def make_vect_envs(
    num_envs: int,
    robot_config: RobotConfig,
    sensor_config: SensorConfig,
    env_config: EnvConfig,
    render_config: RenderConfig,
    pretrained_model: bool,
):
    """Returns async-vectorized gym environments with custom parameters.

    :param env_name: Gym environment name or custom environment class
    :type env_name: str or type
    :param num_envs: Number of vectorized environments, defaults to 1
    :type num_envs: int, optional
    :param env_kwargs: Additional keyword arguments for the environment
    :type env_kwargs: dict
    """

    def make_env():
        return NaviEnv(
            robot_config, sensor_config, env_config, render_config, pretrained_model
        )

    return gym.vector.AsyncVectorEnv([make_env for _ in range(num_envs)])


def create_population(
    state_dim: int,
    action_dim: int,
    net_config: dict,
    agent_config: AgentConfig,
    trainer_config: TrainerConfig,
    population_size: int,
    num_envs: int,
    device: str,
):
    """Returns population of identical agents.

    :param state_dim: State observation dimension
    :type state_dim: int
    :param action_dim: Action dimension
    :type action_dim: int
    :param population_size: Number of agents in population, defaults to 1
    :type population_size: int, optional
    :param num_envs: Number of vectorized environments, defaults to 1
    :type num_envs: int, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    """
    population = []

    for idx in range(population_size):
        agent = RainbowDQN(
            state_dim=state_dim,
            action_dim=action_dim,
            index=idx,
            net_config=net_config,
            batch_size=trainer_config.batch_size,
            lr=trainer_config.lr,
            learn_step=trainer_config.learn_step,
            gamma=agent_config.gamma,
            tau=agent_config.tau,
            beta=agent_config.beta,
            prior_eps=agent_config.prior_eps,
            num_atoms=agent_config.num_atoms,
            v_min=agent_config.v_min,
            v_max=agent_config.v_max,
            n_step=agent_config.n_step,
            noise_std=agent_config.noise_std,
            device=device,
        )
        population.append(agent)

    return population


def print_hyperparams(pop):
    """Prints current hyperparameters of agents in a population and their fitnesses.

    :param pop: Population of agents
    :type pop: list[object]
    """

    for agent in pop:
        print(
            "Agent ID: {}    Mean 5 Fitness: {:.2f}    Attributes: {}".format(
                agent.index, np.mean(agent.fitness[-5:]), agent.inspect_attributes()
            )
        )
