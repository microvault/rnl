from stable_baselines3.common.callbacks import BaseCallback
from rnl.agents.evaluate import statistics, evaluate_agent
import json
from rnl.training.utils import make_environemnt

class DynamicTrainingCallback(BaseCallback):
    def __init__(
        self,
        evaluator,
        justificativas_history,
        get_strategy_dict_func,
        check_freq=100
    ):
        super().__init__(verbose=0)
        self.evaluator = evaluator
        self.check_freq = check_freq
        self.justificativas_history = justificativas_history
        self.get_strategy_dict = get_strategy_dict_func

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:

        if self.n_calls % self.check_freq == 0:
            eval_env = make_environemnt()
            evaluation_results = evaluate_agent(self.model, eval_env)
            print(evaluation_results)
            self.logger.record("rollout/evaluation_results", json.dumps(evaluation_results))
            # Coleta infos de cada env
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
                "min_lidar",
            ]:
                if any(campo in info for info in infos_list):
                    media, _, _, desvio = statistics(infos_list, campo)
                    stats[campo + "_mean"] = media
                    stats[campo + "_std"] = desvio

            print(json.dumps(stats, indent=4))
            evaluation_result = self.evaluator.evaluate_training(
                self.get_strategy_dict(), stats, self.justificativas_history
            )
            print(evaluation_result)
            try:
                new_config = (evaluation_result if isinstance(evaluation_result, dict)
                              else json.loads(evaluation_result))
            except Exception as e:
                print("Erro ao parsear JSON:", e)
                new_config = {}

            if "justify" in new_config:
                self.justificativas_history.append({
                    "justify": new_config["justify"],
                    "config": new_config,
                })
                # Mantém histórico curto
                if len(self.justificativas_history) > 5:
                    self.justificativas_history.pop(0)

            if "strategy" in new_config:
                strategy = new_config["strategy"]

                # Extraindo informações de reward
                reward_info = strategy.get("reward", {})
                new_reward_type = reward_info.get("reward_type", None)
                new_params = {}
                if "parameters" in reward_info:
                    for param in reward_info["parameters"]:
                        new_params[param["key"]] = param["value"]

                # Extraindo informações do domain (map size e obstacle percentage)
                domain_info = strategy.get("domain", {})
                new_map_size = domain_info.get("map_size", {}).get("value", None)
                new_obstacle_percentage = domain_info.get("obstacle_percentage", {}).get("value", None)

                # Chama o método update_strategy do ambiente, passando os 3 novos parâmetros
                self.training_env.env_method(
                    "update_strategy",
                    new_map_size,
                    new_obstacle_percentage,
                    new_reward_type,
                    new_params
                )

        return True
