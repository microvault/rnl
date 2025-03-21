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
        check_freq=10000
    ):
        super().__init__(verbose=0)
        self.evaluator = evaluator
        self.check_freq = check_freq
        self.justificativas_history = justificativas_history
        self.get_strategy_dict = get_strategy_dict_func

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:

        use_agents = True
        if self.n_calls % self.check_freq == 0:
            eval_env = make_environemnt()
            evaluation_results = evaluate_agent(self.model, eval_env)
            self.logger.record("rollout/evaluation_results", json.dumps(evaluation_results))
            # Coleta infos de cada env
            infos_list = []
            for i in range(self.model.n_envs):
                env_info = self.training_env.env_method("get_infos", indices=i)[0]
                if env_info:
                    infos_list.extend(env_info)

            stats = {}
            if use_agents:
                for campo in [
                    "obstacle_score",
                    "orientation_score",
                    "progress_score",
                    "time_score",
                    "dist",
                    "alpha",
                    "min_lidar",
                    "max_lidar",
                ]:
                    if any(campo in info for info in infos_list):
                        media, _, _, desvio = statistics(infos_list, campo)
                        stats[campo + "_mean"] = media
                        stats[campo + "_std"] = desvio

            evaluation_result = {
                "strategy": {
                    "reward": {
                        "reward_type": "time",
                        "parameters": [
                            {"key": "scale_distance", "value": 0.1},
                            {"key": "scale_orientation", "value": 0.003},
                            {"key": "scale_time", "value": 0.001},
                            {"key": "scale_obstacle", "value": 0.01}
                        ]
                    },
                    "mode": {"mode": "easy-00"},
                    "action": {"action_type": "BalancedActions"}
                },
                "justify": "Dado que o robô está aprendendo a navegar em ambientes simples (easy-00), e as métricas mostram um progresso razoável, mas com espaço para melhorias na orientação e na prevenção de obstáculos, sugiro ajustar a recompensa para 'all' com um aumento na escala da distância e orientação para incentivar o robô a se mover mais eficientemente em direção ao objetivo e manter um alinhamento melhor. Reduzi a penalidade por tempo para permitir que o robô explore mais sem ser excessivamente penalizado. Aumentei a penalidade por obstaculos para ele aprender a desviar. Mudar para o modo 'easy-03' aumenta a complexidade do ambiente, preparando o robô para desafios maiores. A ação 'BalancedActions' oferece um bom compromisso entre velocidade e controle, adequado para esta fase de aprendizado."
            }
            # print(json.dumps(stats, indent=4))
            # self.evaluator.evaluate_training(
                # self.get_strategy_dict(), stats, self.justificativas_history
            # )
            # print(evaluation_result)
            try:
                new_config = (evaluation_result if isinstance(evaluation_result, dict)
                              else json.loads(evaluation_result))
            except Exception as e:
                print("Erro ao parsear JSON:", e)
                new_config = {}

            # Salva justificativa
            if "justify" in new_config:
                self.justificativas_history.append({
                    "justify": new_config["justify"],
                    "config": new_config,
                })
                # Mantém histórico curto
                if len(self.justificativas_history) > 5:
                    self.justificativas_history.pop(0)

            # Se vier “strategy”, atualiza env
            if "strategy" in new_config:
                strategy = new_config["strategy"]

                # Exemplo: obtendo action_type
                action_info = strategy.get("action", {})
                new_action_type = action_info.get("action_type", None)

                # Exemplo: obtendo reward
                reward_info = strategy.get("reward", {})
                new_reward_type = reward_info.get("reward_type", None)
                new_params = {}
                if "parameters" in reward_info:
                    for param in reward_info["parameters"]:
                        new_params[param["key"]] = param["value"]

                # Chama método interno do env que você definir para atualizar
                self.training_env.env_method(
                    "update_strategy",
                    new_action_type,
                    new_reward_type,
                    new_params
                )

        return True
