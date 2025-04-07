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
        self.manual_schema = {
            "type": "object",
            "properties": {
                "strategies": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "reward": {
                                "type": "object",
                                "properties": {
                                    "scale_orientation": {"type": "number", "minimum": 0.001, "maximum": 0.1},
                                    "scale_distance": {"type": "number", "minimum": 0.01, "maximum": 0.1},
                                    "scale_time": {"type": "number", "minimum": 0.001, "maximum": 0.05},
                                    "scale_obstacle": {"type": "number", "minimum": 0.001, "maximum": 0.01}
                                },
                                "required": ["scale_orientation", "scale_distance", "scale_time", "scale_obstacle"]
                            },
                            "domain": {
                                "type": "object",
                                "properties": {
                                    "obstacle_percentage": {"type": "integer", "minimum": 0, "maximum": 50},
                                    "map_size": {"type": "number", "minimum": 1.0, "maximum": 5.0}
                                },
                                "required": ["obstacle_percentage", "map_size"]
                            }
                        },
                        "required": ["reward", "domain"]
                    }
                },
                "justify": {"type": "string"}
            },
            "required": ["strategies", "justify"]
        }

    def call_evaluator_llm(self, prompt: str):
        """
        Calls the Gemini LLM using the google.generativeai package.
        """
        client = genai.Client(api_key=self.evaluator_api_key)

        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=self.manual_schema,
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

    def build_evaluation_prompt(self, summary_data, history):
        """
        summary_data é um array onde cada elemento representa uma população, contendo:
        {
            'pop_id': 1,
            'rewards': {'obstacle': 0.01, 'angle': 0.02, 'distance': 0.03, 'time': 0.005},
            'metrics': {
                'success_pct': 80.0,
                'unsafe_pct': 20.0,
                'angular_use_pct': 50.0
            }
        }
        """
        # Objetivo principal
        objective = "Objetivo Macro: Ajustar as recompensas e domínio para melhorar taxa de sucesso e reduzir insegurança."

        # Resumo de histórico (curto)
        history_lines = []
        for loop_idx, loop_entry in enumerate(history, start=1):
            short_justify = loop_entry.get('justify', '')
            history_lines.append(f"Loop {loop_idx}: {short_justify}")
            for pop_idx, pop_data in enumerate(loop_entry.get('population_data', [])):
                metrics_info = pop_data.get('metrics', {})
                history_lines.append(
                    f"  Pop {pop_idx+1} -> Acerto: {metrics_info.get('success_pct','')}%, Inseguro: {metrics_info.get('unsafe_pct','')}%, AngVel: {metrics_info.get('angular_use_pct','')}%"
                )

        # Sumário atual
        current_summary_str = []
        for pop in summary_data:
            r = pop['rewards']
            m = pop['metrics']
            current_summary_str.append(
                f"Pop {pop['pop_id']}: [obst={r['obstacle']:.4f}, ang={r['angle']:.4f}, dist={r['distance']:.4f}, time={r['time']:.4f}] | "
                f"Acerto={m['success_pct']:.1f}%, Inseguro={m['unsafe_pct']:.1f}%, AngVel={m['angular_use_pct']:.1f}%"
            )

        # Exemplo de resposta
        json_example = (
            '{\n'
            '  "strategies": [\n'
            '    {\n'
            '      "reward": {\n'
            '        "scale_orientation": 0.015,\n'
            '        "scale_distance": 0.03,\n'
            '        "scale_time": 0.005,\n'
            '        "scale_obstacle": 0.002\n'
            '      },\n'
            '      "domain": {\n'
            '        "obstacle_percentage": 25,\n'
            '        "map_size": 3.5\n'
            '      }\n'
            '    }\n'
            '  ],\n'
            '  "justify": "Mudanças para aumentar taxa de sucesso e reduzir insegurança."\n'
            '}'
        )

        # Incluindo a abordagem de reflexão (CoT)
        thinking_section = "[Processo interno de análise passo a passo das métricas e do histórico, não exibido ao usuário.]"
        reflection_section = "[Reflexão sobre a coerência e consistência das análises, assegurando clareza no objetivo de navegação segura do robô.]"

        output_section = (
            f"{objective}\n\n"
            "Histórico Simplificado:\n"
            + "\n".join(history_lines)
            + "\n\nStatus Atual:\n"
            + "\n".join(current_summary_str)
            + "\n\n"
            "**Formato Obrigatório** (exemplo):\n"
            + json_example
            + "\n\n"
            "Adapte cada objeto em 'strategies' para cada população na mesma ordem, justificando o motivo das mudanças."
        )

        prompt = (
            f"<thinking>\n{thinking_section}\n</thinking>\n"
            f"<reflection>\n{reflection_section}\n</reflection>\n"
            f"<output>\n{output_section}\n</output>"
        )

        return prompt
