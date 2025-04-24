import os
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

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

    #     self.sample_file = f"sample_metrics_{self.run_id}.jsonl"
    #     self._reset_sample_file()  # Limpa o arquivo se já existir

    # def _reset_sample_file(self):
    #     if os.path.exists(self.sample_file):
    #         with open(self.sample_file, "w") as f:
    #             f.write("")

    def _init_callback(self) -> None:
        self.start_time = time.time()
        # Cria pasta de checkpoints se não existir
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

    def _on_step(self) -> bool:
        # Captura recompensas e comprimentos dos episódios
        if len(self.locals["infos"]) > 0:
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])

        # Avaliação principal + logs a cada "check_freq"
        if self.n_calls % self.check_freq == 0:
            eval_env = make_environemnt(
                self.robot_config,
                self.sensor_config,
                self.env_config,
                self.render_config,
                self.mode,
                self.type_reward,
            )
            evaluation_results = evaluate_agent(self.model, eval_env)
            (
                sucess_rate,
                total_timesteps,
                percentage_unsafe,
                percentage_angular,
                ep_mean_length,
                avg_collision_steps,
                avg_goal_steps,
            ) = evaluation_results

            self.logger.record("rollout/success_rate", sucess_rate)
            self.logger.record("rollout/total_timesteps", total_timesteps)
            self.logger.record("rollout/percentage_unsafe", percentage_unsafe)
            self.logger.record("rollout/percentage_angular", percentage_angular)
            self.logger.record("rollout/ep_mean_length", ep_mean_length)
            self.logger.record("rollout/avg_collision_steps", avg_collision_steps)
            self.logger.record("rollout/avg_goal_steps", avg_goal_steps)

            # Pega infos personalizadas
            infos_list = []
            for i in range(self.model.n_envs):
                env_info = self.training_env.env_method("get_infos", indices=i)[0]
                if env_info:
                    infos_list.extend(env_info)

            stats = {}
            for campo in [
                "obstacle_score",
                "orientation_score",
                "progress_score",
                "time_score",
            ]:
                if any(campo in info for info in infos_list):
                    media, _, _, desvio = statistics(infos_list, campo)
                    stats[campo + "_mean"] = media
                    stats[campo + "_std"] = desvio

            # Calcula FPS
            elapsed_time = time.time() - self.start_time
            fps = self.model.num_timesteps / elapsed_time if elapsed_time > 0 else 0

            # Cálculo das médias de reward e episódios
            mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
            mean_length = np.mean(self.episode_lengths) if self.episode_lengths else 0.0

            # Logs no logger SB3
            self.logger.record("time/fps", fps)
            self.logger.record("rollout/ep_rew_mean", mean_reward)
            self.logger.record("rollout/ep_len_mean", mean_length)

            # Se o wandb_run existe, faz log no WandB
            if self.wandb_run is not None:
                wandb_log = {
                    "rollout/success_rate": sucess_rate,
                    "rollout/total_timesteps": total_timesteps,
                    "rollout/percentage_unsafe": percentage_unsafe,
                    "rollout/percentage_angular": percentage_angular,
                    "rollout/ep_mean_length": ep_mean_length,
                    "rollout/avg_collision_steps": avg_collision_steps,
                    "rollout/avg_goal_steps": avg_goal_steps,
                    "time/fps": fps,
                    "rollout/ep_rew_mean": mean_reward,
                    "rollout/ep_len_mean": mean_length,
                }
                for k, v in stats.items():
                    wandb_log[k] = v

                self.wandb_run.log(wandb_log, step=self.n_calls)

                # Salva modelo em checkpoints
                if self.n_calls % (self.save_checkpoint * 5) == 0:
                    save_path = f"{self.model_save_path}/model_{self.n_calls}_steps"
                    self.model.save(save_path)
                    self.wandb_run.save(save_path + ".zip")

        # if self.n_calls % self.sample_checkpoint_freq == 0:
        #     # Faz uma avaliação rápida
        #     eval_env = make_environemnt()
        #     evaluation_results = evaluate_agent(self.model, eval_env)
        #     (
        #         sucess_rate, total_timesteps, percentage_unsafe,
        #         percentage_angular, ep_mean_length, avg_collision_steps,
        #         avg_goal_steps
        #     ) = evaluation_results
        #     merged_metrics = {
        #         "success_percentage": sucess_rate,
        #         "total_timesteps": total_timesteps,
        #         "percentage_unsafe": percentage_unsafe,
        #         "percentage_angular": percentage_angular,
        #         "ep_mean_length": ep_mean_length,
        #         "avg_collision_steps": avg_collision_steps,
        #         "avg_goal_steps": avg_goal_steps,

        #     }

        #     # Adiciona uma info de passo de treino (ou episódio) no dict
        #     merged_metrics["training_step"] = self.n_calls

        #     # Salva no arquivo .jsonl (cada linha um JSON)
        #     with open(self.sample_file, "a") as f:
        #         f.write(json.dumps(merged_metrics) + "\n")

        return True
