import json

import requests


class LLMTrainingConfigurator:
    """
    Classe para gerar a configuração de treinamento a partir de:
    1) JSON (ex.: definições de ações, recompensas, ambientes)
    2) Entrada de usuário (texto livre)
    Ela chama o LLM (ex.: Gemini) que retorna a estrutura de treino.
    """

    def __init__(self, gemini_api_url, gemini_api_key):
        self.gemini_api_url = gemini_api_url
        self.gemini_api_key = gemini_api_key

    def build_prompt(self, user_input: str, context_json: dict) -> str:
        """
        Monta o prompt que será enviado à LLM (Gemini).
        Pode incluir instruções sobre ações, recompensas e ambiente.
        """
        # Exemplo simples de prompt
        base_info = json.dumps(context_json, indent=2)
        prompt = (
            f"Você é um assistente que configura treinos de RL para robôs.\n\n"
            f"Contexto:\n{base_info}\n\n"
            f"Requisito do usuário:\n{user_input}\n\n"
            f"Retorne a configuração final de treinamento em formato JSON, "
            f"incluindo: parâmetros de PPO, ambiente, ações, estados, e função de recompensa."
        )
        return prompt

    def call_gemini(self, prompt: str) -> str:
        """
        Método que faria a chamada real pra API do Gemini.
        Aqui está só como placeholder. Ajuste com requests ou client adequado.
        """
        headers = {
            "Authorization": f"Bearer {self.gemini_api_key}",
            "Content-Type": "application/json",
        }
        payload = {"prompt": prompt, "max_tokens": 500}  # Ajuste conforme API
        # Exemplo de chamada (fictícia):
        # response = requests.post(self.gemini_api_url, headers=headers, json=payload)
        # return response.json()["generated_text"]
        return "{ 'status': 'placeholder', 'training_config': 'Exemplo de saída' }"

    def generate_training_config(self, user_input: str, context_json: dict) -> dict:
        """
        Gera a configuração de treinamento chamando a LLM com prompt adequado.
        Retorna o JSON final (em dicionário Python).
        """
        prompt = self.build_prompt(user_input, context_json)
        llm_response = self.call_gemini(prompt)

        # Converter a resposta em dicionário se vier em JSON
        try:
            training_config = json.loads(llm_response)
        except:
            # Caso falhe, retorna um dicionário padrão ou mensagem de erro
            training_config = {
                "status": "error",
                "message": "Falha ao interpretar resposta da LLM",
            }
        return training_config


class LLMTrainingEvaluator:
    """
    Classe para avaliar as métricas de treinamento (média, desvio padrão, etc.)
    junto com a entrada do usuário. Decide se precisa ou não refazer o treino.
    """

    def __init__(self, evaluator_api_url, evaluator_api_key):
        self.evaluator_api_url = evaluator_api_url
        self.evaluator_api_key = evaluator_api_key

    def build_evaluation_prompt(self, user_input: str, stats: dict) -> str:
        """
        Cria um prompt descrevendo as métricas e o comportamento desejado.
        """
        stats_info = json.dumps(stats, indent=2)
        prompt = (
            f"Aqui estão as métricas do treinamento:\n{stats_info}\n\n"
            f"Requisito do usuário: {user_input}\n\n"
            "Avalie se o comportamento aprendido atende ao requisito. "
            "Responda se está satisfatório ou se deve refazer o treinamento."
        )
        return prompt

    def call_evaluator_llm(self, prompt: str) -> str:
        """
        Chamada placeholder para o LLM de avaliação.
        """
        headers = {
            "Authorization": f"Bearer {self.evaluator_api_key}",
            "Content-Type": "application/json",
        }
        payload = {"prompt": prompt, "max_tokens": 300}
        # response = requests.post(self.evaluator_api_url, headers=headers, json=payload)
        # return response.json()["generated_text"]
        return "Placeholder de avaliação: 'Treinamento satisfatório'"

    def evaluate_training(self, user_input: str, stats: dict) -> str:
        """
        Gera prompt, chama o LLM e retorna texto de avaliação.
        """
        prompt = self.build_evaluation_prompt(user_input, stats)
        llm_response = self.call_evaluator_llm(prompt)
        return llm_response


# Exemplo de uso:
if __name__ == "__main__":
    # JSON com definições (ex.: tipos de ações, ambientes, funções de recompensa)
    context = {
        "environments": ["mapa_aleatorio_1", "mapa_aleatorio_2"],
        "actions": ["avancar", "recuar", "girar_esquerda", "girar_direita"],
        "reward_functions": [
            "colisao_negativa",
            "distancia_objetivo_pos",
            "movimento_suave_bonus",
        ],
    }

    configurator = LLMTrainingConfigurator(
        gemini_api_url="https://api.gemini.fake/v1/completions",
        gemini_api_key="MINHA_CHAVE_GEMINI",
    )

    # Exemplo: usuário quer curvas suaves e manter 1m das paredes
    user_input = "Preciso de curvas suaves, manter 1m de distância dos obstáculos e chegar rápido ao destino."

    training_config = configurator.generate_training_config(user_input, context)
    print("[Configuração de Treinamento]:")
    print(training_config)

    evaluator = LLMTrainingEvaluator(
        evaluator_api_url="https://api.outroLLM.fake/v1/evaluate",
        evaluator_api_key="MINHA_CHAVE_OUTRO_LLM",
    )

    # Exemplo de estatísticas (coletadas depois do treino)
    example_stats = {
        "time_to_goal_mean": 12.4,
        "time_to_goal_std": 2.1,
        "distance_to_walls_mean": 1.02,
        "collisions_count": 0,
    }

    evaluation_result = evaluator.evaluate_training(user_input, example_stats)
    print("[Avaliação de Treinamento]:")
    print(evaluation_result)
