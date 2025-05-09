from __future__ import annotations

import json
import logging

import backoff
from google import genai
from google.genai import types
import google.api_core.exceptions as gae
import re


DEFAULT_CONFIG = {
    "scale_orientation": 0.0,
    "scale_distance": 0.0,
    "scale_time": 0.01,
    "scale_obstacle": 0.0,
    "scale_angular": 0.0,
}


class LLMTrainingEvaluator:
    def __init__(self, api_key: str, num_populations: int) -> None:
            self.api_key = api_key
            self.num_populations = num_populations
            self.client = genai.Client(api_key=self.api_key)
            self.schema = self._build_schema(num_populations)

    def _build_schema(self, n: int) -> dict:
        item = {
            "type": "object",
            "required": [
                "scale_orientation","scale_distance",
                "scale_time","scale_obstacle","scale_angular"
            ],
            "properties": {
                "scale_orientation": {"type": "number"},
                "scale_distance":   {"type": "number"},
                "scale_time":       {"type": "number"},
                "scale_obstacle":   {"type": "number"},
                "scale_angular":    {"type": "number"},
            },
        }
        return {
            "type": "object",
            "required": ["configurations"],
            "properties": {
                "configurations": {
                    "type": "array",
                    "minItems": n,
                    "maxItems": n,
                    "items": item,
                }
            },
        }

    # ---------- helpers ----------
    def _clean_json_text(self, txt: str) -> str | None:
        txt = re.sub(r"```(?:json)?", "", txt).strip()
        if "{" not in txt or "}" not in txt:
            return None
        txt = txt[txt.find("{") : txt.rfind("}") + 1]
        txt = re.sub(r",\s*}", "}", txt)          # vírgula pendurada
        txt = txt.replace("NaN", "null")          # NaN → null
        return txt

    def _only_json_prompt(self, p: str) -> str:
        return p.strip() + "\n\nSomente JSON, sem markdown ou explicações."

    def _pad_configs(self, data: dict) -> dict:
        cfgs = data.get("configurations", [])
        while len(cfgs) < self.num_populations:
            cfgs.append(DEFAULT_CONFIG.copy())
        data["configurations"] = cfgs[: self.num_populations]
        return data

    @backoff.on_exception(backoff.expo,
                          (gae.InternalServerError, gae.TooManyRequests, ValueError),
                          max_tries=4, jitter=None)
    def _call_gemini(self, prompt: str) -> dict:
        resp = self.client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=prompt + "\n\nSó JSON, sem explicações.",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                candidate_count=1,           # + rápido
                temperature=0.5,
                max_output_tokens=250,
            ),
        )
        if resp.parsed:
            return self._pad_configs(resp.parsed)

        for part in resp.candidates[0].content.parts:
            txt = getattr(part, "text", "")
            clean = self._clean_json_text(txt)
            if clean:
                try:
                    return self._pad_configs(json.loads(clean))
                except json.JSONDecodeError as e:
                    logging.debug("Falha manual: %s\n%s", e, clean)

        raise ValueError("Nenhum JSON parseável")

    def directed_reflection(self, best_population_metrics, history, summary_data, task) -> str:

        hist_str = ""
        for loop_idx, loop_entry in enumerate(history, start=1):
            for pop_data in loop_entry.get("population_data", []):
                met = pop_data.get("metrics", {})
                scales = pop_data.get("scales", {})
                hist_str += (
                    f"\n  Sucesso={met.get('success_pct', 0)}, "
                    f"Inseguro={met.get('unsafe_pct', 0)}, "
                    f"Angular={met.get('angular_use_pct', 0)}"
                    f" Escala obstaculo={scales['scale obstacle']}"
                    f" Escala orientacao={scales['scale angle']}"
                    f" Escala distancia={scales['scale distance']}"
                    f" Escala tempo={scales['scale time']}"
                    f" Escala angular={scales['scale angular']}"

                )


        scale_pop_text = ""
        for pop in summary_data:
            s = pop["scales"]
            scale_pop_text += (
                f"\nEscala Obstaculo={s['scale obstacle']}, "
                f"Escala Orientacão={s['scale angle']}, Escala Distancia={s['scale distance']}, Escala Tempo={s['scale time']}, Escala Angular={s['scale angular']},"
            )

        prompt = f"""
        Você é um engenheiro de recompensas. Analise as métricas do treinamento atual e proponha melhorias na função de recompensa para otimizar o desempenho do agente.​

        Com base nas métricas fornecidas, aplique as regras de análise para:​
            - Identificar possíveis causas de desempenho subótimo.
            - Forneça uma análise passo a passo justificando cada recomendação.
            - Não precisa mostrar como resolver, somente uma analise do que mudar.
            - As escalas de recompensa podem ser 0 também.
            - O objetivo 1 Taxa de sucesso 100% ou quase;
            - O objetivo 2 é usar o minimo de garantir velocidade angular
            - O objetivo 3 é ter uma taxa de insegurança baixa

        ## Contexto:
            Somente é possivel ajustar os seguintes parametros:
            * Escala de recompensa por tempo (sempre negativa) -> minimo: 0.001, maximo: 0.1
            * Escala de recompensa obstaculo (sempre negativa) -> minimo: 0.001, maximo: 0.1
            * Escala de recompensa distancia (sempre negativa) -> minimo: 0.001, maximo: 0.1
            * Escala de recompensa angulo (sempre negativa) -> minimo: 0.001, maximo: 0.1
            * Escala de recompensa por acao angular (sempre negativa) -> minimo: 0.001, maximo: 0.1

        ## Ambiente
            {task}

        ## Funções de recompensa:
            Colisão e chegada: Colidiu → -1.0 e termina; chegou → +1.0 e termina.
            Orientação: Orientação perfeita → 0.0; caso contrário penalidade proporcional.
            Tempo: Penalidade fixa a cada step.
            Progresso: Quanto mais perto do destino, menor a penalidade.
            Proximidade de obstaculo: Penalidade cresce conforme se aproxima do obstáculo.
            Uso de ação angular: Ações 1 ou 2 (giro) recebem penalidade fixa.

        ## Regras de reflexão:
            - Porcentagem de insegurança alta → aumentar penalidade de proximidade.
            - Episódios longos + muitos comandos angulares → penalizar ações angulares e/ou tempo.
            - Nem sempre prefica usar todas os modulos de recompensa, pode zera-los, exemplo se a insegurança estiver muito baixa, e ja nao esta zerada, não precisa usar o reward de obstacle.
            - Seja o mais direto e curto possivel, evite ações desnecessárias. Faca em 1 paragrafo no imperativo, mas mostrando numeros e justificativas claras e curtas.
            - Use pequenas escalas de recompensa, pois o ambiente é muito pequeno.
            - Leve em consideracao a melhor populacao e o historico, caso veja que nao esta progredindo, volte com as recompensas.
            - Utilize recompensas isoladas, como por exemplo so usar time e zerar as reatantes para ver o resultado de cada um.
            - O que pode ser feito é pedir para isolar cada recompensa selecionando qual delas você quer testar.

        ### Historico de dados:
            {hist_str}

        ### Escalas de recompensas atuais:
            {scale_pop_text}

        ### Dados Atuais:
            - Porcentagem de sucesso: {best_population_metrics['success_percentage']}%
            - Porcentagem de passos até o objetivo: {best_population_metrics['avg_goal_steps']}
            - Porcentagem de passos até colisão: {best_population_metrics['avg_collision_steps']}
            - Porcentagem de insegurança: {best_population_metrics['percentage_unsafe']}
            - Porcentagem de uso de velocidade angular: {best_population_metrics['percentage_angular']}
            - Recompensa média por tempo: {best_population_metrics['time_score_mean']}
            - Recompensa média por proximidade: {best_population_metrics['obstacle_score_mean']}
            - Recompensa média por orientação: {best_population_metrics['orientation_score_mean']}
            - Recompensa média por progresso: {best_population_metrics['progress_score_mean']}​

        ### reflexão:
        """

        print(f"\033[32m{prompt}\033[0m")


        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=500,
                temperature=0.6,
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
            f" no total de {num_populations} configurações. "
            "Mesmo que só tenhamos as métricas da melhor população, gere diferentes "
            "variações para comparar. Porem siga o feedback anteriormente fornecido."
            "Alem disso gere outras amostras para explorar mais."
            "Se pedir para testar recompensas isoladas, crie uma configuracao usando somente um valor isolado de cada moduloe zerando os restantea para ver o que cada modulo pode contribuir separadamente"
        )
        reflection_text = (
            "\n- ".join(reflections) if reflections else "Nenhuma reflexão anterior"
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

            Retorne um JSON com exatamente {num_populations} configurações dentro do campo "configurations".

            Não retorne mais do que {num_populations}.

            Cada item deve conter os seguintes campos numéricos:
            - scale_orientation
            - scale_distance
            - scale_time
            - scale_obstacle
            - scale_angular

            Exemplo (com {num_populations} itens):

            {{
              "configurations": [
                {{
                  "scale_orientation": 0.02,
                  "scale_distance": 0.05,
                  "scale_time": 0.01,
                  "scale_obstacle": 0.004,
                  "scale_angular": 0.004
                }},
              ]
            }}

            Retorne SOMENTE o JSON (sem comentários, sem texto explicativo).
            """

        print(f"\033[34m{base_text}\033[0m")

        return base_text

    def request_configurations_for_all(
        self, summary_data, history, reflections, num_populations
    ) -> dict:
        """Envia prompt, garante JSON e devolve dicionário pronto para uso."""
        prompt = self.build_configurations_prompt(
            summary_data, history, reflections, num_populations
        )
        try:
            return self._call_gemini(prompt)
        except Exception:
            logging.exception("Gemini falhou — usando configs padrão")
            return {
                "configurations": [
                    DEFAULT_CONFIG.copy() for _ in range(self.num_populations)
                ]
            }
