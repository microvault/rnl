import os
import time

from stable_baselines3.common.callbacks import BaseCallback

import numpy as np
from rnl.agents.evaluate import evaluate_agent, statistics
from rnl.training.utils import make_environemnt
from rnl.configs.config import RobotConfig, SensorConfig, EnvConfig, RenderConfig
from rnl.configs.rewards import RewardConfig


class DynamicTrainingCallback(BaseCallback):
    def __init__(
        self,
        check_freq: int,
        wandb_run,
        save_checkpoint: int,
        model_save_path: str,
        sample_checkpoint_freq: int,
        run_id: str,
        robot_config: RobotConfig,
        sensor_config: SensorConfig,
        env_config: EnvConfig,
        render_config: RenderConfig,
        mode: str,
        type_reward: RewardConfig,
    ):
        super().__init__(verbose=0)
        self.check_freq = check_freq
        self.wandb_run = wandb_run
        self.save_checkpoint = save_checkpoint
        self.model_save_path = model_save_path
        self.sample_checkpoint_freq = sample_checkpoint_freq
        self.run_id = run_id
        self.robot_config = robot_config
        self.sensor_config = sensor_config
        self.env_config = env_config
        self.render_config = render_config
        self.mode = mode
        self.type_reward = type_reward

        self.start_time = None
        self.episode_rewards = []
        self.episode_lengths = []

    def _init_callback(self) -> None:
        self.start_time = time.time()
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

    def _on_step(self) -> bool:
        # Inicialização de métricas de episódio
        if len(self.locals["infos"]) > 0:
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])

        # Avaliação periódica
        if self.n_calls % self.check_freq == 0:
            eval_env = make_environemnt(
                self.robot_config,
                self.sensor_config,
                self.env_config,
                self.render_config,
                self.mode,
                self.type_reward,
            )
            (
                sucess_rate,
                total_timesteps,
                percentage_unsafe,
                percentage_angular,
                ep_mean_length,
                avg_collision_steps,
                avg_goal_steps,
            ) = evaluate_agent(self.model, eval_env)

            # Coleta infos adicionais dos ambientes de treino
            infos_list = []
            for i in range(self.model.n_envs):
                env_info = self.training_env.env_method("get_infos", indices=i)[0]
                if env_info:
                    infos_list.extend(env_info)

            # Estatísticas de scores extras
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

            # Prepara dicionário de métricas
            mean_metrics = {
                # Avaliação
                "success_rate_mean": sucess_rate,
                "percentage_unsafe_mean": percentage_unsafe,
                "percentage_angular_mean": percentage_angular,
                "avg_collision_steps_mean": avg_collision_steps,
                "avg_goal_steps_mean": avg_goal_steps,
                "total_timesteps": total_timesteps,
                **{campo + "_mean": stats.get(campo + "_mean", 0.0) for campo in [
                    "time_score", "progress_score", "orientation_score", "obstacle_score"
                ]},
                "ep_rew_mean": float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
                "ep_len_mean": float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0,
            }

            # Logging apenas das métricas médias e totais
            for k, v in mean_metrics.items():
                self.logger.record(f"rollout/{k}", v)

            # W&B logging
            if self.wandb_run is not None:
                wandb_log = {f"rollout/{k}": v for k, v in mean_metrics.items()}
                self.wandb_run.log(wandb_log, step=self.n_calls)

            # Salvamento de checkpoint
            if self.n_calls % self.save_checkpoint == 0:
                save_path = f"{self.model_save_path}/model_{self.n_calls}_steps"
                print(f"Saving model to {save_path}")
                self.model.save(save_path)
                if self.wandb_run is not None:
                    self.wandb_run.save(save_path + ".zip")

        return True
