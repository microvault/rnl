from google import genai
from google.genai import types
from importlib import reload

class LLMTrainingEvaluator:
    def __init__(self, evaluator_api_key: str, allow_domain: bool = True):
        self.evaluator_api_key = evaluator_api_key
        self.allow_domain = allow_domain
        self.manual_schema = self.define_manual_schema(self.allow_domain)

    def define_manual_schema(self, allow_domain: bool = True):
        base_reward_properties = {
            "scale_orientation": {"type": "number", "minimum": 0.001, "maximum": 0.1},
            "scale_distance":    {"type": "number", "minimum": 0.01,  "maximum": 0.1},
            "scale_time":        {"type": "number", "minimum": 0.001, "maximum": 0.05},
            "scale_obstacle":    {"type": "number", "minimum": 0.001, "maximum": 0.01}
        }
        domain_properties = {
            "obstacle_percentage": {"type": "integer", "minimum": 0,  "maximum": 50},
            "map_size":            {"type": "number",  "minimum": 1.0, "maximum": 5.0}
        }
        # Se allow_domain=True, unimos domain + reward
        all_properties = dict(base_reward_properties)
        if allow_domain:
            all_properties.update(domain_properties)

        item_schema = {
            "type": "object",
            "properties": all_properties,
            "required": list(all_properties.keys())
        }

        return {
            "type": "object",
            "properties": {
                "configurations": {
                    "type": "array",
                    "items": item_schema
                }
            },
            "required": ["configurations"]
        }

    def directed_reflection(self, best_population_metrics: dict) -> str:
        prompt = f"""
        Vocé é um engenheiro de recompensas (reward engineer). Por favor, analise cuidadosamente o feedback da política e forneça uma nova função de recompensa melhorada e a configuração do ambiente que possa a tarefa.

        # Regras
        1. Se o **Percentage Unsafe** estiver muito alto e **Success Percentage** muito baixo, seja o ue talvez o ambiente esteja muito difícil e seria melhor diminuir a porcentagem de obstáculos e tamanho do mapa.
        2. Se o **Percentage Unsafe** está muito alto mas **Success Percentage** manteve acima de 50% talvez o mapa esteja do tamanho bom mas a porcentagem de obstáculos seja muito alta ou o mapa esteja muito alto e porcentagem de obstáculos esteja muito baixa.
        3. Se **Percentage Unsafe** esteja muito alto, aumente a penalidade de proximidade de colisão
        4. Se **Avg Goal Steps** ou **Avg Goal Steps** esteja muito alto e angular esteja muito alto, o robô significa que fica girando em voltas e aprendeu a ficar nesse mínimo local. Sendo que não avança nem colide. Mude as recompensa faça ele explorer mais.

        # Exemplos
        **1. Exemplo**:
            Métricas:
                - Taxa de sucesso: 100%
                - Média de steps até o objetivo: 1000 steps
                - Média de steps até colisão: 25 steps
                - Porcentagem de steps em região de insegurança: 10%
                - Recompensa média por tempo: -0.001
                - Recompensa média por proximidade de obstáculo: -0.002
                - Recompensa média por orientação em relação ao objetivo: -0.003
                - Recompensa média por distância até o objetivo: -0.004
            Reflexão:
                O Success Percentage esta em 20%, o que pode significar que o ambiente está muito difícil, diminuir o tamanho do ambiente de 3 para 2.

        **2. Exemplo**:
            Métricas:
                - Taxa de sucesso: 20%
                - Média de steps até o objetivo: 24
                - Média de steps até colisão: 12
                - Porcentagem de steps em região de insegurança: 10%
                - Recompensa média por tempo: -0.001
                - Recompensa média por proximidade de obstáculo: -0.002
                - Recompensa média por orientação em relação ao objetivo: -0.003
                - Recompensa média por distância até o objetivo: -0.004
            Reflexão:
                O Avg Collision Step está muito baixo ou seja, o robô está colidindo muito rápido tentar diminuir o número de obstáculos de 40% para 20%.

        **3. Exemplo**:
            Métricas:
                - Taxa de sucesso: 20%
                - Média de steps até o objetivo: 24
                - Média de steps até colisão: 12
                - Porcentagem de steps em região de insegurança: 10%
                - Recompensa média por tempo: -0.001
                - Recompensa média por proximidade de obstáculo: -0.002
                - Recompensa média por orientação em relação ao objetivo: -0.003
                - Recompensa média por distância até o objetivo: -0.004
            Reflexão:
                O Percentage Unsafe está muito alto, aumentar a escala de penalidade de 0,001 para 0,002 para manter o robô mais longe de obstáculo.

        Métricas:
            - Success Percentage: {best_population_metrics['success_percentage']}%
            - Avg Goal Steps: {best_population_metrics['avg_goal_steps']}
            - Avg Collision Steps: {best_population_metrics['avg_collision_steps']}
            - Percentage Unsafe: {best_population_metrics['percentage_unsafe']}
            - Time Score Mean: {best_population_metrics['time_score_mean']}
            - Obstacle Score Mean: {best_population_metrics['obstacle_score_mean']}
            - Orientation Score Mean: {best_population_metrics['orientation_score_mean']}
            - Progress Score Mean: {best_population_metrics['progress_score_mean']}
        Reflexão:
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

    def build_configurations_prompt(self, summary_data, history, reflections, num_populations):
        objective = (
            "Por favor, analise cuidadosamente o feedback da política e forneça uma nova função de recompensa melhorada que possa resolver melhor a tarefa"
            f"no total de {num_populations} configurações. "
            "Mesmo que só tenhamos as métricas da melhor população, gere diferentes "
            "variações para comparar."
        )
        reflection_text = "\n".join(reflections) if reflections else "Nenhuma reflexão anterior"
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
            """
        return base_text

    def request_configurations_for_all(self, summary_data, history, reflections, num_populations):
        prompt = self.build_configurations_prompt(summary_data, history, reflections, num_populations)
        client = genai.Client(api_key=self.evaluator_api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash-001", # gemini-2.5-pro-exp-03-25 # gemini-2.0-flash-001
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
