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
            "required": list(all_properties.keys())  # Exige todos
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
        Análise da melhor população:
        - Success Percentage: {best_population_metrics['success_percentage']}%
        - Avg Goal Steps: {best_population_metrics['avg_goal_steps']}
        - Avg Collision Steps: {best_population_metrics['avg_collision_steps']}
        - Percentage Unsafe: {best_population_metrics['percentage_unsafe']}
        - Time Score Mean: {best_population_metrics['time_score_mean']}

        Sugira possíveis melhorias de treino, baseado nessas métricas.
        Retorne apenas um texto curto.
        """
        client = genai.Client(api_key=self.evaluator_api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=100,
                temperature=0.3,
            ),
        )
        return str(response.text)

    def build_configurations_prompt(self, summary_data, history, reflections, num_populations):
        objective = (
            "Crie múltiplas configurações (uma para cada futuro treinamento), "
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
            model="gemini-2.0-flash-001",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=self.manual_schema,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                candidate_count=3,
                seed=5,
            ),
        )
        return response.parsed
