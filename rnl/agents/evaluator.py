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
        return p.strip() + "\n\nJSON only, no markdown or explanations."

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
            contents = prompt + "\n\nJSON only, no explanations.",
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
                    logging.debug("Manual failure: %s\n%s", e, clean)

        raise ValueError("No parseable JSON")

    def directed_reflection(self, best_population_metrics, history, summary_data, task) -> str:

        hist_str = ""
        for loop_idx, loop_entry in enumerate(history, start=1):
            for pop_data in loop_entry.get("population_data", []):
                met = pop_data.get("metrics", {})
                scales = pop_data.get("scales", {})
                hist_str += (
                    f"\n  Success={met.get('success_pct', 0)}, "
                    f"Unsafe={met.get('unsafe_pct', 0)}, "
                    f"Angular={met.get('angular_use_pct', 0)}"
                    f" Obstacle scale={scales['scale obstacle']}"
                    f" Orientation scale={scales['scale angle']}"
                    f" Distance scale={scales['scale distance']}"
                    f" Time scale={scales['scale time']}"
                    f" Angular scale={scales['scale angular']}"
                )

        scale_pop_text = ""
        for pop in summary_data:
            s = pop["scales"]
            scale_pop_text += (
                f"\nObstacle Scale={s['scale obstacle']}, "
                f"Orientation Scale={s['scale angle']}, Distance Scale={s['scale distance']}, Time Scale={s['scale time']}, Angular Scale={s['scale angular']},"
            )

        prompt = f"""
        You are a reward engineer. Analyze the current training metrics and propose improvements to the reward function to optimize agent performance.

        Based on the provided metrics, apply analysis rules to:
            - Identify potential causes of suboptimal performance.
            - Provide a step-by-step analysis justifying each recommendation.
            - Do not show how to fix it, only analyze what to change.
            - Reward scales can also be set to 0.
            - Goal 1: Achieve a success rate of 100% or close;
            - Goal 2: Minimize angular velocity usage;
            - Goal 3: Maintain a low unsafe rate

        ## Context:
            Only the following parameters can be adjusted:
            * Time reward scale (always negative) -> min: 0.001, max: 0.1
            * Obstacle reward scale (always negative) -> min: 0.001, max: 0.1
            * Distance reward scale (always negative) -> min: 0.001, max: 0.1
            * Angle reward scale (always negative) -> min: 0.001, max: 0.1
            * Angular action reward scale (always negative) -> min: 0.001, max: 0.1

        ## Environment
            {task}

        ## Reward functions:
            Collision and goal: Collided → -1.0 and ends; reached → +1.0 and ends.
            Orientation: Perfect orientation → 0.0; otherwise proportional penalty.
            Time: Fixed penalty per step.
            Progress: The closer to the goal, the lower the penalty.
            Obstacle proximity: Penalty increases when close to obstacles.
            Angular action usage: Actions 1 or 2 (turning) receive fixed penalty.

        ## Reflection rules:
            - High unsafe percentage → increase obstacle proximity penalty.
            - Long episodes + many angular commands → penalize angular actions and/or time.
            - It's not necessary to use all reward modules; they can be set to zero. For example, if unsafe rate is very low, obstacle reward can be zeroed.
            - Be as direct and concise as possible; avoid unnecessary actions. Write in 1 paragraph using imperative voice, but include clear and short justifications with numbers.
            - Use small reward scales since the environment is small.
            - Consider the best population and history; if no progress is seen, revert previous rewards.
            - Use isolated rewards, e.g., only time reward and zero the others to see individual impact.
            - One approach is to isolate each reward by selecting which one to test.

        ### Data history:
            {hist_str}

        ### Current reward scales:
            {scale_pop_text}

        ### Current Data:
            - Success rate: {best_population_metrics['success_percentage']}%
            - Steps to goal (avg): {best_population_metrics['avg_goal_steps']}
            - Steps to collision (avg): {best_population_metrics['avg_collision_steps']}
            - Unsafe percentage: {best_population_metrics['percentage_unsafe']}
            - Angular velocity usage: {best_population_metrics['percentage_angular']}
            - Avg time reward: {best_population_metrics['time_score_mean']}
            - Avg obstacle reward: {best_population_metrics['obstacle_score_mean']}
            - Avg orientation reward: {best_population_metrics['orientation_score_mean']}
            - Avg progress reward: {best_population_metrics['progress_score_mean']}

        ### Reflection:
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
            "Please carefully analyze the policy feedback and provide a new improved reward function that can better solve the task"
            f" across a total of {num_populations} configurations. "
            "Even though we only have the metrics from the best population, generate different "
            "variations for comparison. However, follow the previously provided feedback."
            "Also generate other samples to explore further."
            "If asked to test isolated rewards, create a configuration using only one active reward module at a time, setting the others to zero to evaluate each module's individual contribution."
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
                    f"\n  Success={met.get('success_pct', 0)}, "
                    f"Unsafe={met.get('unsafe_pct', 0)}, "
                    f"Angular={met.get('angular_use_pct', 0)}"
                )

        best_pop_text = ""
        for pop in summary_data:
            r = pop["rewards"]
            m = pop["metrics"]
            best_pop_text += (
                f"\nBest Pop {pop['pop_id']} -> Obst={r['obstacle']:.3f}, "
                f"Ang={r['angle']:.3f}, Dist={r['distance']:.3f}, Time={r['time']:.3f}, "
                f"Success={m['success_pct']:.2f}, Unsafe={m['unsafe_pct']:.2f}, Angular={m['angular_use_pct']:.2f}"
            )

        base_text = f"""
        Objective: {objective}

        Reflections:
        {reflection_text}

        Simplified History:
        {hist_str}

        Best Population:
        {best_pop_text}

        Return a JSON with exactly {num_populations} configurations inside the "configurations" field.

        Do not return more than {num_populations}.

        Each item must contain the following numeric fields:
        - scale_orientation
        - scale_distance
        - scale_time
        - scale_obstacle
        - scale_angular

        Example (with {num_populations} items):

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

        Return ONLY the JSON (no comments, no explanatory text).
        """


        print(f"\033[34m{base_text}\033[0m")

        return base_text

    def request_configurations_for_all(
        self, summary_data, history, reflections, num_populations
    ) -> dict:
        prompt = self.build_configurations_prompt(
            summary_data, history, reflections, num_populations
        )
        try:
            return self._call_gemini(prompt)
        except Exception:
            logging.exception("Gemini failed — using default configs")
            return {
                "configurations": [
                    DEFAULT_CONFIG.copy() for _ in range(self.num_populations)
                ]
            }
