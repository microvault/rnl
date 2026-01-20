import os
import time

import numpy as np

from rnl.agents.evaluate import evaluate_agent, statistics
from rnl.configs.config import EnvConfig, RenderConfig, RobotConfig, SensorConfig
from rnl.configs.rewards import RewardConfig
from rnl.training.utils import make_environemnt


class DynamicTrainingCallback:
    """
    Custom callback for PPO training.
    """

    def __init__(
        self,
        check_freq: int,
        wandb_run,
        save_checkpoint: int,
        model_save_path: str,
        run_id: str,
        robot_config: RobotConfig,
        sensor_config: SensorConfig,
        env_config: EnvConfig,
        render_config: RenderConfig,
        type_reward: RewardConfig,
    ):
        self.check_freq = check_freq
        self.wandb_run = wandb_run
        self.save_checkpoint = save_checkpoint
        self.model_save_path = model_save_path
        self.run_id = run_id
        self.robot_config = robot_config
        self.sensor_config = sensor_config
        self.env_config = env_config
        self.render_config = render_config
        self.type_reward = type_reward

        self.start_time = None
        self.episode_rewards = []
        self.episode_lengths = []
        self.n_calls = 0
        self.model = None
        self.training_env = None

    def init_callback(self, model, training_env):
        """Initialize callback with model and training environment."""
        self.model = model
        self.training_env = training_env
        self.start_time = time.time()
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

    def on_step(self) -> bool:
        """Called after each training step."""
        self.n_calls += 1

        # Note: Episode info collection would need to be implemented
        # depending on how the environment tracking is set up

        if self.n_calls % self.check_freq == 0:
            eval_env = make_environemnt(
                self.robot_config,
                self.sensor_config,
                self.env_config,
                self.render_config,
                self.type_reward,
            )
            (
                sucess_rate,
                percentage_unsafe,
                percentage_angular,
                ep_mean_length,
                avg_collision_steps,
                avg_goal_steps,
            ) = evaluate_agent(self.model, eval_env)

            infos_list = []
            try:
                if self.training_env is not None and hasattr(self.training_env, 'num_envs'):
                    for i in range(self.training_env.num_envs):
                        if hasattr(self.training_env, 'env_method'):
                            env_info = self.training_env.env_method("get_infos", indices=i)[0]
                            if env_info:
                                infos_list.extend(env_info)
            except Exception as e:
                print(f"Warning: Could not retrieve environment infos: {e}")

            stats = {}
            for campo in [
                "obstacle_score",
                "orientation_score",
                "progress_score",
                "time_score",
            ]:
                if any(campo in info for info in infos_list):
                    media, _, _, _ = statistics(infos_list, campo)
                    stats[campo + "_mean"] = media

            mean_metrics = {
                "success_rate_mean": sucess_rate,
                "percentage_unsafe_mean": percentage_unsafe,
                "percentage_angular_mean": percentage_angular,
                "avg_collision_steps_mean": avg_collision_steps,
                "avg_goal_steps_mean": avg_goal_steps,
                **{
                    campo + "_mean": stats.get(campo + "_mean", 0.0)
                    for campo in [
                        "time_score",
                        "progress_score",
                        "orientation_score",
                        "obstacle_score",
                    ]
                },
                "ep_rew_mean": (
                    float(np.mean(self.episode_rewards))
                    if self.episode_rewards
                    else 0.0
                ),
                "ep_len_mean": (
                    float(np.mean(self.episode_lengths))
                    if self.episode_lengths
                    else 0.0
                ),
            }

            # Log to console
            print(f"\n[Step {self.n_calls}] Training Metrics:")
            for k, v in mean_metrics.items():
                print(f"  {k}: {v:.4f}")

            if self.wandb_run is not None:
                wandb_log = {f"rollout/{k}": v for k, v in mean_metrics.items()}
                self.wandb_run.log(wandb_log, step=self.n_calls)

            if self.n_calls % self.save_checkpoint == 0:
                save_path = f"{self.model_save_path}/model_{self.n_calls}_steps"
                print(f"Saving model to {save_path}")
                self.model.save(save_path)
                if self.wandb_run is not None:
                    self.wandb_run.save(save_path + ".zip")

        return True
