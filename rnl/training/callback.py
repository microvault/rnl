from stable_baselines3.common.callbacks import BaseCallback

from rnl.agents.evaluate import evaluate_agent, statistics
from rnl.training.utils import make_environemnt


class DynamicTrainingCallback(BaseCallback):
    def __init__(
        self,
        check_freq=100,
    ):
        super().__init__(verbose=0)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            eval_env = make_environemnt()
            evaluation_results = evaluate_agent(self.model, eval_env)
            sucess_rate, total_timesteps, percentage_unsafe, percentage_angular, ep_mean_length, avg_collision_steps, avg_goal_steps = evaluation_results

            self.logger.record("rollout/success_rate", sucess_rate)
            self.logger.record("rollout/total_timesteps", total_timesteps)
            self.logger.record("rollout/percentage_unsafe", percentage_unsafe)
            self.logger.record("rollout/percentage_angular", percentage_angular)
            self.logger.record("rollout/ep_mean_length", ep_mean_length)
            self.logger.record("rollout/avg_collision_steps", avg_collision_steps)
            self.logger.record("rollout/avg_goal_steps", avg_goal_steps)

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

        return True
