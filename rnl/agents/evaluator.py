import json

from google import genai
from google.genai import types


class LLMTrainingEvaluator:
    """
    Class to evaluate training metrics (average, std deviation, etc.)
    along with user input. Decides if retraining is necessary.
    """

    def __init__(self, evaluator_api_key: str):
        self.evaluator_api_key = evaluator_api_key

    def build_evaluation_prompt(
        self, context_json: dict, stats: dict, context: dict
    ) -> str:
        """
        Creates a prompt describing the metrics and desired behavior.
        """
        stats_info = json.dumps(stats, indent=2)
        base_info = json.dumps(context_json, indent=2)
        context_info = json.dumps(context, indent=2)

        prompt = (
            "Você é um assistente para configurar o treinamento RL de robôs.\n\n"
            f"Historico das ultimas avaliacoes: {context_info}\n\n"
            "1. Configurações Básicas:\n"
            f"   - Base de configurações: {base_info}\n"
            f"   - Métricas de treinamento: {stats_info}\n\n"
            "2. Detalhes das Métricas (Desvio padrão e média):\n"
            "   - obstacle_score: penalidade quando o sensor lidar tem medicoes muito perto de obstaculos\n"
            "   - orientation_score: maior recompensa se o robô estiver direcionado para o objetivo.\n"
            "   - progress_score: diferença entre a posição inicial e a posição atual do robô em relação ao objetivo.\n"
            "   - time_score: penalidade por tempo.\n"
            "   - action: média das ações (0 = ir para frente, 1 = virar à direita, 2 = virar à esquerda).\n"
            "   - dist: distância do robô ao objetivo.\n"
            "   - alpha: ângulo do robô em relação ao objetivo.\n"
            "   - min_lidar: menor medição do lidar.\n"
            "   - max_lidar: maior medição do lidar.\n\n"
            "3. Tarefa:\n"
            "   Usando as métricas e a base de configurações, avalie e retorne em formato JSON:\n"
            "     - Modifique a configuração da recompensa com a escala apropriada.\n"
            "     - O tipo de ação.\n"
            "     - O modo do ambiente.\n\n"
            "O objetivo é ensinar o robô a chegar ao alvo sem colidir com obstáculos, usando apenas os estados (medições do lidar, ângulo alpha, "
            "distância até o objetivo e a última ação tomada).\n"
            "O robô deve ser capaz de aprender a melhor política de ação para maximizar a recompensa total.\n"
            "Justifique suas escolhas no parametro 'justificativa'.\n"
        )

        return prompt

    def call_evaluator_llm(self, prompt: str):
        """
        Calls the Gemini LLM using the google.generativeai package.
        """
        client = genai.Client(api_key=self.evaluator_api_key)

        manual_schema = {
            "type": "object",
            "properties": {
                "justify": {"type": "string"},
                "strategy": {
                    "type": "object",
                    "properties": {
                        "reward": {
                            "type": "object",
                            "properties": {
                                "reward_type": {"type": "string"},
                                "parameters": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "key": {"type": "string"},
                                            "value": {"type": "number"},
                                        },
                                        "required": ["key", "value"],
                                    },
                                },
                            },
                            "required": ["reward_type", "parameters"],
                        },
                        "mode": {
                            "type": "object",
                            "properties": {"mode": {"type": "string"}},
                            "required": ["mode"],
                        },
                        "action": {
                            "type": "object",
                            "properties": {
                                "action_type": {"type": "string"},
                            },
                            "required": ["action_type"],
                        },
                    },
                    "required": ["reward", "mode", "action"],
                },
            },
            "required": ["strategy"],
        }

        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=manual_schema,
                temperature=0,
                top_p=0.95,
                top_k=20,
                candidate_count=1,
                seed=5,
            ),
        )

        if response is not None:
            return response.parsed
        else:
            raise ValueError("No candidates returned from Gemini API.")

    def evaluate_training(self, context: dict, stats: dict, history: dict):
        """
        Generates a prompt, calls the LLM, and returns the evaluation text.
        """
        prompt = self.build_evaluation_prompt(context, stats, history)
        llm_response = self.call_evaluator_llm(prompt)
        return llm_response


# Example usage:
# if __name__ == "__main__":
#     evaluator = LLMTrainingEvaluator(
#         evaluator_api_key="AIzaSyC2gTeqruWUdSltxkzl5tpwHvlS4Ffx1bI"  # Replace with your valid API key
#     )

#     # Example training metrics (collected after training)
#     example_stats = {
#         "obstacle_score_mean": -0.00010759832713556042,
#         "obstacle_score_std": 0.000239928065948947,
#         "orientation_score_mean": 0.0012805782593126652,
#         "orientation_score_std": 0.0008954319340984938,
#         "progress_score_mean": -0.00410694868449611,
#         "progress_score_std": 0.012642693778069883,
#         "time_score_mean": -0.00937007874015748,
#         "time_score_std": 0.0024294879717388206,
#         "dist_mean": 0.38105717799003874,
#         "dist_std": 0.160274870045089,
#         "alpha_mean": 0.45790818174166853,
#         "alpha_std": 0.2756886272246146,
#         "min_lidar_mean": 0.16303590130368908,
#         "min_lidar_std": 0.0877522568964114,
#         "max_lidar_mean": 0.4204654638653039,
#         "max_lidar_std": 0.1299746233048569
#     }

#     evaluation_result = evaluator.evaluate_training(get_strategy_dict(), example_stats)
#     print(evaluation_result)
