import multiprocessing
import time

from rnl.configs.config import (
    EnvConfig,
    NetworkConfig,
    RenderConfig,
    RobotConfig,
    SensorConfig,
    TrainerConfig,
)
from rnl.training.learn import training
from rnl.agents.evaluator import LLMTrainingEvaluator
from rnl.configs.rewards import RewardConfig
import json
import os

def train_worker(
    robot_config,
    sensor_config,
    env_config,
    render_config,
    trainer_config,
    network_config,
    reward_config,
):
    metrics, eval = training(
        robot_config,
        sensor_config,
        env_config,
        render_config,
        trainer_config,
        network_config,
        reward_config,
        print_parameter=False,
    )

    return metrics, eval


def run_parallel_trainings(list_of_configs):
    start_time = time.time()
    results = []

    with multiprocessing.Pool(processes=len(list_of_configs)) as pool:
        async_results = []
        for cfg in list_of_configs:
            args = (
                cfg["robot_config"],
                cfg["sensor_config"],
                cfg["env_config"],
                cfg["render_config"],
                cfg["trainer_config"],
                cfg["network_config"],
                cfg["reward_config"],
            )
            async_results.append(pool.apply_async(train_worker, args))

        for res in async_results:
            results.append(res.get())

    print(f"Tempo total: {time.time() - start_time:.2f} seg")
    return results


def run_multiple_parallel_trainings(num_loops, initial_configs):
    evaluator = LLMTrainingEvaluator(evaluator_api_key=initial_configs[0]["trainer_config"].llm_api_key)
    history = []
    current_configs = initial_configs.copy()

    for loop in range(num_loops):
        print(f"\n=== Loop {loop+1}/{num_loops} ===")
        results = run_parallel_trainings(current_configs)

        population_summaries = []
        population_data = []
        for i, (metrics, scores) in enumerate(results, start=1):
            summary_dict = {
                "pop_id": i,
                "rewards": {
                    "obstacle": metrics.get('obstacle_score_mean', 0.0),
                    "angle": metrics.get('orientation_score_mean', 0.0),
                    "distance": metrics.get('progress_score_mean', 0.0),
                    "time": metrics.get('time_score_mean', 0.0)
                },
                "metrics": {
                    "success_pct": scores[0],   # Taxa de sucesso
                    "total_steps": scores[1],  # Total de steps
                    "unsafe_pct": scores[2],   # % tempo em zona insegura
                    "angular_use_pct": scores[3]  # % uso de velocidade angular
                }
            }
            population_summaries.append(summary_dict)
            population_data.append({
                "population": i,
                "metrics": summary_dict["metrics"],
                "params": summary_dict["rewards"]
            })

        # Prepara prompt
        evaluation_prompt = evaluator.build_evaluation_prompt(
            summary_data=population_summaries,
            history=history
        )

        print(evaluation_prompt)
        print("###############")

        # Chama LLM
        llm_response = evaluator.call_evaluator_llm(evaluation_prompt)
        print(llm_response)

        strategies = llm_response.get("strategies", [])
        justify = llm_response.get("justify", "")
        print("\n=== LLM Resposta (estratégias) ===")
        print(json.dumps(llm_response, indent=2))

        # Atualiza config de acordo com cada população
        new_configs = []
        for idx, cfg in enumerate(current_configs):
            if idx < len(strategies):
                s = strategies[idx]
                reward_cfg = s["reward"]
                domain_cfg = s["domain"]

                # Exemplo de como atualizar cada config
                # (depende do que de fato quer aplicar em env_config, reward, etc.)
                new_cfg = {
                    "robot_config": cfg["robot_config"],
                    "sensor_config": cfg["sensor_config"],
                    "env_config": EnvConfig(
                        scalar=cfg["env_config"].scalar,
                        folder_map=cfg["env_config"].folder_map,
                        name_map=cfg["env_config"].name_map,
                        timestep=cfg["env_config"].timestep,
                        obstacle_percentage=domain_cfg["obstacle_percentage"],
                        map_size=domain_cfg["map_size"]
                    ),
                    "render_config": cfg["render_config"],
                    "trainer_config": cfg["trainer_config"],
                    "network_config": cfg["network_config"],
                    "reward_config": RewardConfig(
                        params={
                            "scale_orientation": reward_cfg["scale_orientation"],
                            "scale_distance":   reward_cfg["scale_distance"],
                            "scale_time":       reward_cfg["scale_time"],
                            "scale_obstacle":   reward_cfg["scale_obstacle"],
                        }
                    )
                }

                new_configs.append(new_cfg)
            else:
                new_configs.append(cfg)

        history_entry = {
            "loop": loop + 1,
            "strategies": strategies,
            "justify": justify,
            "population_data": population_data
        }
        history.append(history_entry)
        current_configs = new_configs

    # print("\n=== RESULTADO FINAL ===")
    # for h in history:
    #     print(f"Loop {h['loop']}: Justify={h['justify']}")
    #     for pd in h["population_data"]:
    #         print(f"  Pop {pd['population']}: {pd['metrics']}, {pd['params']}")
    return history

def update_training_configs(current_configs, strategy, population_data):
    """Atualiza configurações baseado na estratégia recomendada"""
    new_configs = []

    for idx, config in enumerate(current_configs):
        # Aplicar ajustes progressivos
        new_config = {
            "map_size": strategy["domain"]["map_size"]["value"],
            "obstacle_percentage": strategy["domain"]["obstacle_percentage"]["value"],
            "reward_scale": strategy["reward"]["parameters"][0]["value"],
            "exploration_rate": max(0.1, config["exploration_rate"] * 0.95)
        }
        new_configs.append(new_config)

    return new_configs

def analyze_final_results(history):
    """Gera relatório comparativo"""
    for entry in history:
        print(f"\nLoop {entry['loop']}:")
        for pop in entry['population_data']:
            print(f"População {pop['population']}:")
            print(f"  Sucesso: {pop['metrics'].get('success_rate', 0)}%")
            print(f"  Parâmetros: {pop['params']}")

def print_training_results_formatted(all_results):

    for loop_idx, loop_results in enumerate(all_results, start=1):
        print(f"\nLoop {loop_idx}:")
        for training_idx, (metrics, scores) in enumerate(loop_results, start=1):
            result_str = f"\Population {training_idx}:\n"
            result_str += " Metrics:\n"
            for key, value in metrics.items():
                result_str += f"   {key:<25s}: {value:>8.4f}\n"
            accuracy = scores[0]
            total_steps_score = scores[1]
            unsafe_steps = scores[2] * 100
            angular_steps = scores[3] * 100
            result_str += " Scores:\n"
            result_str += f"   {'Accuracy (%)':<25s}: {accuracy:>8.2f}%\n"
            result_str += f"   {'Total Steps':<25s}: {total_steps_score:>8}\n"
            result_str += f"   {'Unsafe Zone Steps (%)':<25s}: {unsafe_steps:>8.2f}%\n"
            result_str += f"   {'Angular Velocity Steps (%)':<25s}: {angular_steps:>8.2f}%\n"
            result_str += "-" * 50 + "\n"
            print(result_str, end="\n")

if __name__ == "__main__":
    robot_config = RobotConfig(
        base_radius=0.105,
        vel_linear=[0.0, 0.22],
        vel_angular=[1.0, 2.84],
        wheel_distance=0.16,
        weight=1.0,
        threshold=0.1,  # 4 # 0.03
        collision=0.075,  # 2 # 0.075
        path_model="None",
    )
    sensor_config = SensorConfig(fov=270.0, num_rays=5, min_range=0.0, max_range=3.5)
    env_config = EnvConfig(scalar=100, folder_map="", name_map="", timestep=1000, obstacle_percentage=40.0, map_size=5)
    render_config = RenderConfig(controller=False, debug=True, plot=False)

    trainer_config = TrainerConfig(
        pretrained="None",
        use_agents=False,
        max_timestep_global=100,
        seed=1,
        batch_size=8,
        num_envs=4,
        device="cpu",
        checkpoint_path="ppo_policy_network",
        use_wandb=False,
        wandb_api_key="",
        llm_api_key=str(os.environ.get("GEMINI_API_KEY")),
        lr=0.0003,
        learn_step=512,
        gae_lambda=0.95,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
        update_epochs=10,
        name="rnl-v1",
        verbose=False,
    )

    network_config = NetworkConfig(
        hidden_size=[20, 10],
        mlp_activation="ReLU",
        type_model="MlpPolicy",
    )

    reward_config = RewardConfig(
        params={
            "scale_orientation": 0.02,
            "scale_distance": 0.06,
            "scale_time": 0.01,
            "scale_obstacle": 0.004,
        },
    )

    config1 = {
        "robot_config": robot_config,
        "sensor_config": sensor_config,
        "env_config": env_config,
        "render_config": render_config,
        "trainer_config": trainer_config,
        "network_config": network_config,
        "reward_config": reward_config,
    }

    pop = 2

    base_configs = [config1]

    configs = base_configs * pop

    num_loops = 4
    all_results = run_multiple_parallel_trainings(num_loops, configs)

    # print_training_results_formatted(all_results)
