from google import genai
from google.genai import types


class LLMTrainingEvaluator:
    def __init__(self, evaluator_api_key: str, allow_domain: bool = True):
        self.evaluator_api_key = evaluator_api_key
        self.allow_domain = allow_domain
        self.manual_schema = self.define_manual_schema(self.allow_domain)

    def define_manual_schema(self, allow_domain: bool = True):
        base_reward_properties = {
            "scale_orientation": {"type": "number", "minimum": 0.001, "maximum": 0.1},
            "scale_distance": {"type": "number", "minimum": 0.01, "maximum": 0.1},
            "scale_time": {"type": "number", "minimum": 0.001, "maximum": 0.05},
            "scale_obstacle": {"type": "number", "minimum": 0.001, "maximum": 0.01},
        }
        domain_properties = {
            "obstacle_percentage": {"type": "integer", "minimum": 0, "maximum": 50},
            "map_size": {"type": "number", "minimum": 1.0, "maximum": 5.0},
        }
        # Se allow_domain=True, unimos domain + reward
        all_properties = dict(base_reward_properties)
        if allow_domain:
            all_properties.update(domain_properties)

        item_schema = {
            "type": "object",
            "properties": all_properties,
            "required": list(all_properties.keys()),
        }

        return {
            "type": "object",
            "properties": {"configurations": {"type": "array", "items": item_schema}},
            "required": ["configurations"],
        }

    # TODO: adicionar o codigo do ambiente e parametros
    def directed_reflection(self, best_population_metrics: dict) -> str:
        print(best_population_metrics)
        prompt = f"""
        Você é um engenheiro de recompensas. Analise as métricas do treinamento atual e proponha melhorias na função de recompensa e na configuração do ambiente para otimizar o desempenho do agente.​

        Com base nas métricas fornecidas, aplique as regras de análise para:​
            - Identificar possíveis causas de desempenho subótimo.
            - Forneça uma análise passo a passo justificando cada recomendação.
            - Não precisa mostrar como resolver, somente uma analise do que mudar.

        Contexto:
            Somente é possivel ajustar os seguintes parametros
            * Escala de recompensa por tempo
            * Escala de recompensa obstaculo
            * Escala de recompensa distancia
            * Escala de recompensa angulo
            * tamanho do mapa
            * Porcentagem de obstaculo

        ## Regras de Análise:

            - Ambiente Muito Difícil:
                Se a taxa de sucesso for baixa e a porcentagem de insegurança alta, considere reduzir o tamanho do mapa e a densidade de obstáculos.​

            - Ajuste de Obstáculos:
                Se a taxa de sucesso for razoável (>50%) mas a porcentagem de insegurança ainda alta, ajuste a densidade de obstáculos ou o tamanho do mapa para equilibrar a dificuldade.​

            - Penalidade por Proximidade:
                Se a porcentagem de insegurança for alta, aumente a penalidade por proximidade a obstáculos para incentivar o agente a manter distância segura.​

            - Comportamento de Giro:
                Se os passos médios até o objetivo ou até a colisão forem altos e a orientação angular também for alta, o agente pode estar preso em um comportamento de giro. Modifique a função de recompensa para incentivar a exploração e o progresso.​

        ## Exemplos de Análise:

        ### Métricas:
            - Taxa de sucesso: 20% (min. 0% - max. 100%)
            - Média de passos até o objetivo: 30 (min. 0 - max. 1000)
            - Média de passos até colisão: 1000 (min. 0 - max. 1000)
            - Porcentagem de insegurança: 10% (min. 0% - max. 100%)
            - Porcentagem de uso de velocidade angular: 100% (min. 0% - max. 100%)
            - Tempo por epsodio médio: 17.7 (min. 0 - max. 1000)
            - Media de si
            - Recompensas médias: tempo (-0.001), proximidade (-0.002), orientação (-0.003), distância (-0.004)​

        ### Análise:
            - A Taxa de sucesso esta em 20% indica um ambiente muito difícil. Reduza o tamanho do mapa de 3 para 2 para facilitar a tarefa.
            - A Média de passos até colisão esta muito alta indicando que o robo esta girando em circulo.

        ### Métricas:
            - Taxa de sucesso: 0% (0% - 100%)
            - Média de passos até o objetivo: 24 (0 - 1000)
            - Média de passos até colisão: 12 (0 - 1000)
            - Porcentagem de insegurança: 10% (0% - 100%)
            - Recompensas médias: tempo (-0.001), proximidade (-0.002), orientação (-0.003), distância (-0.004)​

        ### Análise:
            O agente está colidindo rapidamente. Reduza a densidade de obstáculos de 40% para 20% para permitir melhor navegação.

        ### Métricas:
            - Taxa de sucesso: 20%
            - Média de passos até o objetivo: 24
            - Média de passos até colisão: 12
            - Porcentagem de insegurança: 10%
            - Recompensas médias: tempo (-0.001), proximidade (-0.002), orientação (-0.003), distância (-0.004)​

        ## Análise:
            A alta porcentagem de insegurança sugere proximidade excessiva a obstáculos. Aumente a penalidade por proximidade de 0.001 para 0.002 para desencorajar esse comportamento.

        ## Métricas Atuais:
            - Taxa de sucesso: {best_population_metrics['success_percentage']}%
            - Média de passos até o objetivo: {best_population_metrics['avg_goal_steps']}
            - Média de passos até colisão: {best_population_metrics['avg_collision_steps']}
            - Porcentagem de insegurança: {best_population_metrics['percentage_unsafe']}
            - Recompensa média por tempo: {best_population_metrics['time_score_mean']}
            - Recompensa média por proximidade: {best_population_metrics['obstacle_score_mean']}
            - Recompensa média por orientação: {best_population_metrics['orientation_score_mean']}
            - Recompensa média por progresso: {best_population_metrics['progress_score_mean']}​

        ## Análise:
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
        print("Ref: ", str(response))
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
            uma configuração de treino (reward e domínio, se habilitado).

            Exemplo de JSON:

            {{
            "configurations": [
                {{
                "scale_orientation": 0.02,
                "scale_distance": 0.05,
                "scale_time": 0.01,
                "scale_obstacle": 0.004,
                "obstacle_percentage": 25,
                "map_size": 3.0
                }},
                {{
                "scale_orientation": 0.015,
                "scale_distance": 0.04,
                "scale_time": 0.008,
                "scale_obstacle": 0.003,
                "obstacle_percentage": 10,
                "map_size": 2.5
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
