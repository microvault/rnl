import json
from google import genai
from google.genai import types

class LLMTrainingEvaluator:
    """
    Classe para avaliar métricas de treinamento e decidir se é necessário re-treinar.
    Se allow_domain=False, então não precisamos retornar as configurações de 'domain'.
    """

    def __init__(self, evaluator_api_key: str, allow_domain: bool = True):
        self.evaluator_api_key = evaluator_api_key
        self.allow_domain = allow_domain

        if self.allow_domain:
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
                                    "required": [
                                        "scale_orientation",
                                        "scale_distance",
                                        "scale_time",
                                        "scale_obstacle"
                                    ]
                                },
                                "domain": {
                                    "type": "object",
                                    "properties": {
                                        "obstacle_percentage": {"type": "integer", "minimum": 0, "maximum": 50},
                                        "map_size": {"type": "number", "minimum": 1.0, "maximum": 5.0}
                                    },
                                    "required": ["obstacle_percentage", "map_size"]
                                },
                            },
                            "required": ["reward", "domain"]
                        }
                    },
                    "justify": {"type": "string"}
                },
                "required": ["strategies", "justify"]
            }
        else:
            # Se não permitimos domínio, removemos do schema
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
                                    "required": [
                                        "scale_orientation",
                                        "scale_distance",
                                        "scale_time",
                                        "scale_obstacle"
                                    ]
                                }
                            },
                            "required": ["reward"]
                        }
                    },
                    "justify": {"type": "string"}
                },
                "required": ["strategies", "justify"]
            }

    def call_evaluator_llm(self, prompt: str):
        """
        Chama o modelo Gemini via google.generativeai.
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
            raise ValueError("Nenhuma resposta retornada da API Gemini.")

    def gerar_reflexao(self, summary_data, history):
        """
        Cria um texto de análise simples sobre cada população e um histórico resumido.
        """
        reflexoes = []
        for pop in summary_data:
            m = pop['metrics']
            reflexao = (
                f"População {pop['pop_id']} obteve {m['success_pct']}% de sucesso, "
                f"{m['unsafe_pct']}% de insegurança, "
                f"e {m['angular_use_pct']}% de uso angular."
            )
            reflexoes.append(reflexao)

        # Resumo do histórico
        resumo_historico = []
        for idx, entry in enumerate(history):
            resumo_historico.append(f"Loop {idx+1}: {entry['justify']}")

        texto_reflexao = "Resumo por População:\n" + "\n".join(reflexoes)
        texto_reflexao += "\n\nHistórico Resumido:\n" + "\n".join(resumo_historico)

        return texto_reflexao

    def build_evaluation_prompt(self, summary_data, history):
        """
        Monta um prompt com resumo atual e histórico para o LLM.
        """
        # Mostra a análise
        print(self.gerar_reflexao(summary_data, history))

        regras = ""

        # Objetivo principal
        if self.allow_domain:
            objective = (
                "Objetivo: Ajustar recompensas e domínio para melhorar taxa de sucesso, reduzir insegurança e diminuir a porcentagem de vezes que usa velocidade angular."
            )
        else:
            objective = (
                "Objetivo: Ajustar recompensas para melhorar taxa de sucesso, reduzir insegurança e diminuir a porcentagem de vezes que usa velocidade angular."
            )

        # Histórico simplificado
        history_lines = []
        for loop_idx, loop_entry in enumerate(history, start=1):
            short_justify = loop_entry.get('justify', '')
            history_lines.append(f"Loop {loop_idx}: {short_justify}")
            for pop_idx, pop_data in enumerate(loop_entry.get('population_data', [])):
                m = pop_data.get('metrics', {})
                history_lines.append(
                    f"  Pop {pop_idx+1} -> Acerto: {m.get('success_pct','')}%, "
                    f"Inseguro: {m.get('unsafe_pct','')}%, "
                    f"Veloc Angular: {m.get('angular_use_pct','')}%"
                )

        # Status atual
        current_summary_str = []
        for pop in summary_data:
            r = pop['rewards']
            m = pop['metrics']
            current_summary_str.append(
                f"Pop {pop['pop_id']}: [obst={r['obstacle']:.4f}, ang={r['angle']:.4f}, "
                f"dist={r['distance']:.4f}, time={r['time']:.4f}] -> "
                f"Sucesso={m['success_pct']:.1f}%, Inseguro={m['unsafe_pct']:.1f}%, "
                f"VelAngular={m['angular_use_pct']:.1f}%"
            )

        # Exemplo de resposta (JSON)
        if self.allow_domain:
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
        else:
            json_example = (
                '{\n'
                '  "strategies": [\n'
                '    {\n'
                '      "reward": {\n'
                '        "scale_orientation": 0.015,\n'
                '        "scale_distance": 0.03,\n'
                '        "scale_time": 0.005,\n'
                '        "scale_obstacle": 0.002\n'
                '      }\n'
                '    }\n'
                '  ],\n'
                '  "justify": "Seja direto, somente dizendo o por que utilizou esse parametro e como estava o desempenho"\n'
                '}'
            )

        # Conteúdo final do prompt
        output_section = (
            f"{objective}\n\n"
            "Histórico Simplificado:\n"
            + "\n".join(history_lines)
            + "\n\nStatus Atual:\n"
            + "\n".join(current_summary_str)
            + "\n\n"
            "**Formato Obrigatório** (exemplo):\n"
            + json_example
            + "\n\nAdapte cada objeto em 'strategies' para cada população na mesma ordem.\n"
        )

        return output_section
