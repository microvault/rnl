from google import genai
from google.genai import types


class LLMTrainingEvaluator:
    def __init__(self, evaluator_api_key: str):
        self.evaluator_api_key = evaluator_api_key
        self.manual_schema = self.define_manual_schema()

    def define_manual_schema(self, allow_domain: bool = False):
        base_reward_properties = {
            "scale_orientation": {"type": "number", "minimum": 0.001, "maximum": 0.1},
            "scale_distance": {"type": "number", "minimum": 0.01, "maximum": 0.1},
            "scale_time": {"type": "number", "minimum": 0.001, "maximum": 0.1},
            "scale_obstacle": {"type": "number", "minimum": 0.001, "maximum": 0.01},
            "scale_angular": {"type": "number", "minimum": 0.001, "maximum": 0.01},
        }
        # Se allow_domain=True, unimos domain + reward
        all_properties = dict(base_reward_properties)

        item_schema = {
            "type": "object",
            "properties": all_properties,
        }

        return {
            "type": "object",
            "properties": {"configurations": {"type": "array", "items": item_schema}},
        }

    def directed_reflection(self, best_population_metrics: dict) -> str:
        prompt = f"""
        Você é um engenheiro de recompensas. Analise as métricas do treinamento atual e proponha melhorias na função de recompensa para otimizar o desempenho do agente.​

        Com base nas métricas fornecidas, aplique as regras de análise para:​
            - Identificar possíveis causas de desempenho subótimo.
            - Forneça uma análise passo a passo justificando cada recomendação.
            - Não precisa mostrar como resolver, somente uma analise do que mudar.
            - As escalas de recompensa podem ser 0 também.

        ## Contexto:
            Somente é possivel ajustar os seguintes parametros:
            * Escala de recompensa por tempo (sempre negativa)
            * Escala de recompensa obstaculo (sempre negativa)
            * Escala de recompensa distancia (sempre negativa)
            * Escala de recompensa angulo (sempre negativa)
            * Escala de recompensa por acao angular (sempre negativa)

        ## Ambiente
        - 1000 steps totais, mas ~700 já levam o robô de ponta a ponta do mapa.
        - 3 ações: 0 = frente, 1 = esquerda, 2 = direita.
        - 8 estados: 5 leituras de LiDAR, distância ao objetivo, ângulo ao objetivo e estado do robô (frente/giro).

        ## Funções de recompensa:
            Colisão e chegada:
                Código:
                    def collision_and_target_reward(
                        distance: float, threshold: float, collision: bool, x: float, y: float, poly
                    ) -> Tuple[float, bool]:
                        if not poly.contains(Point(x, y)):
                            return -1.0, True
                        if distance < threshold:
                            return 1.0, True
                        if collision:
                            return -1.0, True
                        return 0.0, False
                Descrição:
                    Colidiu → -1.0 e termina; chegou → +1.0 e termina.
            Orientação:
                Código:
                    def orientation_reward(alpha: float, scale_orientation: float) -> float:
                        alpha_norm = 1.0 - (alpha / np.pi)
                        if alpha_norm < 0.0:
                            alpha_norm = 0.0
                        elif alpha_norm > 1.0:
                            alpha_norm = 1.0

                        return scale_orientation * alpha_norm - scale_orientation
                Descrição:
                    Orientação perfeita → 0.0; caso contrário penalidade proporcional.
            Tempo:
                Código:
                    def time_and_collision_reward(scale_time: float = 0.01) -> float:
                        return -scale_time
                Descrição:
                    Penalidade fixa a cada step.


            Progresso:
                Código:
                    def prog_reward(
                        current_distance: float,
                        min_distance: float,
                        max_distance: float,
                        scale_factor: float,
                    ) -> float:

                        reward = -scale_factor * current_distance
                        return reward
                Descrição:
                    Quanto mais perto do destino, menor a penalidade.

            Proximidade de obstaculo:
                Código:
                    def r3(x: float, threshold_collision: float, scale: float) -> float:
                        margin = 0.3
                        if x <= threshold_collision:
                            return -scale
                        elif x < threshold_collision + margin:
                            return -scale * (threshold_collision + margin - x) / margin
                        else:
                            return 0.0
                Descrição:
                    Penalidade cresce conforme se aproxima do obstáculo.

           Uso de ação angular:
                Código:
                    action_reward = 0

                    if action == 1 or action == 2:
                        action_reward = -scale_angular
                Descrição:
                    Ações 1 ou 2 (giro) recebem penalidade fixa.

        ## Regras de reflexão:
            - Porcentagem de insegurança alta → aumentar penalidade de proximidade.
            - Episódios longos + muitos comandos angulares → penalizar ações angulares e/ou tempo.

        ## Dados de um humano controlando o robo:
            - Taxa de sucesso: 100% (min. 0% - max. 100%)
            - Média de passos até o objetivo: 433.6 (min. 0 - max. 1000)
            - Média de passos até colisão: 30 (min. 0 - max. 1000)
            - Porcentagem de insegurança: 2% (min. 0% - max. 100%)
            - Porcentagem de uso de velocidade angular: 1.13 % (min. 0% - max. 100%)
            - Tempo por epsodio médio: 400 (min. 0 - max. 1000)

            Leve esses dados como referência para melhorar o agente.

        ## Exemplos de reflexão:

        ### Dados:
            - Taxa de sucesso: 20%
            - Média de passos até o objetivo: 30
            - Média de passos até colisão: 900
            - Porcentagem de insegurança: 22%
            - Porcentagem de uso de velocidade angular: 80 %
            - Tempo por epsodio médio: 834

        ### reflexão:
            - A Média de passos até colisão esta muito alta indicando que o robo esta girando em circulo. devo aumentar a penalidade
            para comandos angulares e remover o reward orientation para ver o se melhora.

        ### Dados:
            - Taxa de sucesso: 0%
            - Média de passos até o objetivo: 0
            - Média de passos até colisão: 200
            - Porcentagem de insegurança: 10%
            - Porcentagem de uso de velocidade angular: 2 %
            - Tempo por epsodio médio: 200

        ### reflexão:
            O agente está colidindo rapidamente, talvez seja por que esta somente indo reto e nao considerando as paredes. Aumentar a penalidade por colisao e reduzir a penalidade por proximidade.

        ### Dados:
            - Taxa de sucesso: 80%
            - Média de passos até o objetivo: 723
            - Média de passos até colisão: 150
            - Porcentagem de insegurança: 15%
            - Porcentagem de uso de velocidade angular: 70 %
            - Tempo por epsodio médio: 764

        ## reflexão:
            A alta porcentagem de insegurança sugere proximidade excessiva a obstáculos. Aumente a penalidade por proximidade, alem disso esta usando muita velocidade angular, significa que esta sendo muito instavel.
            Aumentar a velocidade angular para evitar colisões.

        ## Dados Atuais:
            - Taxa de sucesso: {best_population_metrics['success_percentage']}%
            - Média de passos até o objetivo: {best_population_metrics['avg_goal_steps']}
            - Média de passos até colisão: {best_population_metrics['avg_collision_steps']}
            - Porcentagem de insegurança: {best_population_metrics['percentage_unsafe']}
            - Recompensa média por tempo: {best_population_metrics['time_score_mean']}
            - Recompensa média por proximidade: {best_population_metrics['obstacle_score_mean']}
            - Recompensa média por orientação: {best_population_metrics['orientation_score_mean']}
            - Recompensa média por progresso: {best_population_metrics['progress_score_mean']}​

        ## reflexão:
        """

        client = genai.Client(api_key=self.evaluator_api_key)
        response = client.models.generate_content(
            model="gemini-1.5-pro",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=500,
                temperature=0.9,
                top_p=0.99,
                top_k=40,
                candidate_count=1,
                seed=5,
            ),
        )
        return str(response.text)

    def build_configurations_prompt(
        self, summary_data, history, reflections, num_populations
    ):
        objective = (
            "Por favor, analise cuidadosamente o feedback da política e forneça uma nova função de recompensa melhorada que possa resolver melhor a tarefa"
            f"no total de {num_populations} configurações. "
            "Mesmo que só tenhamos as métricas da melhor população, gere diferentes "
            "variações para comparar."
        )
        reflection_text = (
            "\n".join(reflections) if reflections else "Nenhuma reflexão anterior"
        )
        hist_str = ""
        for loop_idx, loop_entry in enumerate(history, start=1):
            hist_str += f"\nLoop {loop_idx}:"
            for pop_data in loop_entry.get("population_data", []):
                met = pop_data.get("metrics", {})
                hist_str += (
                    f"\n  Sucesso={met.get('success_pct', 0)}, "
                    f"Inseguro={met.get('unsafe_pct', 0)}, "
                    f"Angular={met.get('angular_use_pct', 0)}"
                )

        best_pop_text = ""
        for pop in summary_data:
            r = pop["rewards"]
            m = pop["metrics"]
            best_pop_text += (
                f"\nMelhor Pop {pop['pop_id']} -> Obst={r['obstacle']:.3f}, "
                f"Ang={r['angle']:.3f}, Dist={r['distance']:.3f}, Time={r['time']:.3f}, "
                f"Success={m['success_pct']:.2f}, Unsafe={m['unsafe_pct']:.2f}, Angular={m['angular_use_pct']:.2f}"
            )

        base_text = f"""
            Objetivo: {objective}

            Reflexões:
            {reflection_text}

            Histórico Simplificado:
            {hist_str}

            Melhor População:
            {best_pop_text}

            Retorne um JSON com o campo "configurations", contendo {num_populations} itens, onde cada item representa
            uma configuração de treino reward.

            Exemplo de JSON:

            {{
            "configurations": [
                {{
                "scale_orientation": 0.02,
                "scale_distance": 0.05,
                "scale_time": 0.01,
                "scale_obstacle": 0.004,
                "scale_angular": 0.004,
                }},
                {{
                "scale_orientation": 0.015,
                "scale_distance": 0.04,
                "scale_time": 0.008,
                "scale_obstacle": 0.003,
                "scale_angular": 0.005,
                }}
            ]
            }}

            Resposta JSON:
            """
        return base_text

    def request_configurations_for_all(
        self, summary_data, history, reflections, num_populations
    ):
        prompt = self.build_configurations_prompt(
            summary_data, history, reflections, num_populations
        )

        client = genai.Client(api_key=self.evaluator_api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",  # gemini-2.5-pro-exp-03-25 # gemini-2.0-flash-001
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=self.manual_schema,
                temperature=0.2,
                top_p=0.95,
                top_k=30,
                candidate_count=1,
                seed=5,
            ),
        )
        return response.parsed
