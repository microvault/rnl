import gymnasium as gym
import numpy as np

from rnl.configs.config import EnvConfig, RenderConfig, RobotConfig, SensorConfig
from rnl.configs.rewards import RewardConfig
from rnl.environment.env import NaviEnv


def _safe_plot(ax, y, color, label):
    if len(y) == 0:
        ax.set_visible(False)
        return
    x = range(1, len(y) + 1)
    ax.plot(x, y, color=color, label=label, linewidth=1.5)
    ax.set_ylabel(label, fontsize=8)
    ax.legend(fontsize=6)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.tick_params(axis="x", labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    ax.text(
        0.5,
        -0.25,
        f"µ {np.mean(y):.4f} | min {np.min(y):.4f} | max {np.max(y):.4f}",
        transform=ax.transAxes,
        ha="center",
        fontsize=6,
    )


class VectorEnv:
    """
    Simple vectorized environment wrapper for multiple parallel environments.
    """

    def __init__(self, envs):
        self.envs = envs
        self.num_envs = len(envs)
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space

    def reset(self):
        """Reset all environments."""
        observations = []
        for env in self.envs:
            obs, _ = env.reset()
            observations.append(obs)
        return np.array(observations)

    def step(self, actions):
        """Step all environments with given actions."""
        observations = []
        rewards = []
        dones = []
        truncated_list = []
        infos = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, done, truncated, info = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            truncated_list.append(truncated)
            infos.append(info)

        return (
            np.array(observations),
            np.array(rewards),
            np.array(dones),
            np.array(truncated_list),
            infos,
        )

    def env_method(self, method_name, *args, indices=None, **kwargs):
        """Call a method on one or more environments."""
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]

        results = []
        for i in indices:
            method = getattr(self.envs[i], method_name)
            results.append(method(*args, **kwargs))
        return results


def make_vec_env_custom(
    robot_config: RobotConfig,
    sensor_config: SensorConfig,
    env_config: EnvConfig,
    render_config: RenderConfig,
    reward_config: RewardConfig,
    num_envs: int = 1,
):
    """
    Create a vectorized environment.
    """
    envs = []
    for i in range(num_envs):
        env = NaviEnv(
            robot_config,
            sensor_config,
            env_config,
            render_config,
            use_render=False,
            type_reward=reward_config,
        )
        env.reset(seed=13 + i)
        envs.append(env)

    return VectorEnv(envs)
