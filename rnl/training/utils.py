import gymnasium as gym
import numpy as np
import torch
from tqdm import trange

from rnl.algorithms.ppo import PPO
from rnl.configs.config import EnvConfig, RenderConfig, RobotConfig, SensorConfig
from rnl.environment.env import NaviEnv


def make_vect_envs(
    num_envs: int,
    robot_config: RobotConfig,
    sensor_config: SensorConfig,
    env_config: EnvConfig,
    render_config: RenderConfig,
    use_render: bool,
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
            robot_config, sensor_config, env_config, render_config, use_render
        )

    return gym.vector.AsyncVectorEnv([make_env for _ in range(num_envs)])


def create_population(
    env_navigation,
    algo,
    state_dim,
    action_dim,
    one_hot,
    net_config,
    INIT_HP,
    actor_network=None,
    critic_network=None,
    population_size=1,
    num_envs=1,
    device="cpu",
):
    """Returns population of identical agents.

    :param algo: RL algorithm
    :type algo: str
    :param state_dim: State observation dimension
    :type state_dim: int
    :param action_dim: Action dimension
    :type action_dim: int
    :param one_hot: One-hot encoding
    :type one_hot: bool
    :param net_config: Network configuration
    :type net_config: dict or None
    :param INIT_HP: Initial hyperparameters
    :type INIT_HP: dict
    :param actor_network: Custom actor network, defaults to None
    :type actor_network: nn.Module, optional
    :param critic_network: Custom critic network, defaults to None
    :type critic_network: nn.Module, optional
    :param population_size: Number of agents in population, defaults to 1
    :type population_size: int, optional
    :param num_envs: Number of vectorized environments, defaults to 1
    :type num_envs: int, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    :param torch_compiler:
    :type torch_compiler:
    """
    population = []

    for idx in range(population_size):
        agent = PPO(
            env_navigation=env_navigation,
            state_dim=state_dim,
            action_dim=action_dim,
            one_hot=one_hot,
            discrete_actions=INIT_HP["DISCRETE_ACTIONS"],
            index=idx,
            net_config=net_config,
            batch_size=INIT_HP["BATCH_SIZE"],
            lr=INIT_HP["LR"],
            learn_step=INIT_HP["LEARN_STEP"],
            gamma=INIT_HP["GAMMA"],
            gae_lambda=INIT_HP["GAE_LAMBDA"],
            action_std_init=INIT_HP["ACTION_STD_INIT"],
            clip_coef=INIT_HP["CLIP_COEF"],
            ent_coef=INIT_HP["ENT_COEF"],
            vf_coef=INIT_HP["VF_COEF"],
            max_grad_norm=INIT_HP["MAX_GRAD_NORM"],
            target_kl=INIT_HP["TARGET_KL"],
            update_epochs=INIT_HP["UPDATE_EPOCHS"],
            actor_network=actor_network,
            critic_network=critic_network,
            device=device,
            accelerator=None,
        )
        population.append(agent)

    return population


def check_policy_on_policy_with_probe_env(env, agent, learn_steps=5000, device="cpu"):
    print(f"Probe environment: {type(env).__name__}")

    for _ in trange(learn_steps):
        state, _ = env.reset()
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []
        truncs = []

        for _ in range(100):
            action, log_prob, _, value = agent.get_action(np.expand_dims(state, 0))
            action = action[0]
            log_prob = log_prob[0]
            value = value[0]
            next_state, reward, done, trunc, _ = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            values.append(value)
            truncs.append(trunc)

            state = next_state
            if done:
                state, _ = env.reset()

        experiences = (
            states,
            actions,
            log_probs,
            rewards,
            dones,
            values,
            next_state,
        )
        agent.learn(experiences)

    for sample_obs, v_values in zip(env.sample_obs, env.v_values):
        state = torch.tensor(sample_obs).float().to(device)
        if v_values is not None:
            predicted_v_values = agent.critic(state).detach().cpu().numpy()[0]
            # print("---")
            print("v", v_values, predicted_v_values)
            assert np.allclose(v_values, predicted_v_values, atol=0.1)

    if hasattr(env, "sample_actions"):
        for sample_action, policy_values in zip(env.sample_actions, env.policy_values):
            action = torch.tensor(sample_action).float().to(device)
            if policy_values is not None:
                predicted_policy_values = (
                    agent.actor(sample_obs).detach().cpu().numpy()[0]
                )
                print("pol", policy_values, predicted_policy_values)
                assert np.allclose(policy_values, predicted_policy_values, atol=0.1)
