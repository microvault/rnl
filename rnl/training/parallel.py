import multiprocessing
import os

from rnl.agents.evaluator import LLMTrainingEvaluator
from rnl.configs.config import (
    EnvConfig,
    NetworkConfig,
    RenderConfig,
    RobotConfig,
    SensorConfig,
    TrainerConfig,
)
from rnl.configs.rewards import RewardConfig
from rnl.training.learn import training


def train_worker(
    robot_config,
    sensor_config,
    env_config,
    render_config,
    trainer_config,
    network_config,
    reward_config,
):
    metrics = training(
        robot_config,
        sensor_config,
        env_config,
        render_config,
        trainer_config,
        network_config,
        reward_config,
        print_parameter=False,
    )

    return metrics


def run_parallel_trainings(list_of_configs):
    results = []
    best_index = None
    best_criteria = None

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
            metrics = res.get()
            results.append(metrics)

    print_population_metrics(results)

    for i, metrics in enumerate(results):
        candidate = (
            metrics["success_percentage"],
            metrics["avg_goal_steps"],
            -metrics["avg_collision_steps"],
        )
        if best_criteria is None or candidate > best_criteria:
            best_criteria = candidate
            best_index = i

    if best_index is not None:
        return results[best_index]
    else:
        return None


def run_multiple_parallel_trainings(
    num_loops, initial_configs, allow_domain_modifications, num_populations
):
    evaluator = LLMTrainingEvaluator(
        evaluator_api_key=initial_configs[0]["trainer_config"].llm_api_key,
        allow_domain=allow_domain_modifications,
    )
    history = []
    reflections = []
    current_configs = initial_configs.copy()

    for loop in range(num_loops):
        print(f"\n=== Loop {loop+1}/{num_loops} ===")

        best_metrics = run_parallel_trainings(current_configs)
        reflection = evaluator.directed_reflection(best_metrics)
        reflections.append(reflection)

        summary_data = [
            {
                "pop_id": 1,
                "rewards": {
                    "obstacle": best_metrics.get("obstacle_score_mean", 0.0),
                    "angle": best_metrics.get("orientation_score_mean", 0.0),
                    "distance": best_metrics.get("progress_score_mean", 0.0),
                    "time": best_metrics.get("time_score_mean", 0.0),
                },
                "metrics": {
                    "success_pct": best_metrics.get("success_percentage", 0.0),
                    "unsafe_pct": best_metrics.get("percentage_unsafe", 0.0) * 100,
                    "angular_use_pct": best_metrics.get("percentage_angular", 0.0)
                    * 100,
                },
            }
        ]

        llm_response = evaluator.request_configurations_for_all(
            summary_data, history, reflections, num_populations
        )
        print(llm_response)
        new_configs_data = llm_response.get("configurations", [])

        new_configs = []
        for idx, cfg in enumerate(current_configs):
            if idx < len(new_configs_data):
                item = new_configs_data[idx]
                obstacle_percentage = (
                    item.get("obstacle_percentage", 40)
                    if allow_domain_modifications
                    else 40
                )
                map_size = (
                    item.get("map_size", 3.0) if allow_domain_modifications else 3.0
                )

                updated_env = EnvConfig(
                    scalar=cfg["env_config"].scalar,
                    folder_map=cfg["env_config"].folder_map,
                    name_map=cfg["env_config"].name_map,
                    timestep=cfg["env_config"].timestep,
                    obstacle_percentage=obstacle_percentage,
                    map_size=map_size,
                )
                updated_reward = RewardConfig(
                    params={
                        "scale_orientation": item.get("scale_orientation", 0.02),
                        "scale_distance": item.get("scale_distance", 0.06),
                        "scale_time": item.get("scale_time", 0.01),
                        "scale_obstacle": item.get("scale_obstacle", 0.004),
                    }
                )
                new_cfg = {
                    "robot_config": cfg["robot_config"],
                    "sensor_config": cfg["sensor_config"],
                    "env_config": updated_env,
                    "render_config": cfg["render_config"],
                    "trainer_config": cfg["trainer_config"],
                    "network_config": cfg["network_config"],
                    "reward_config": updated_reward,
                }
                new_configs.append(new_cfg)
            else:
                new_configs.append(cfg)

        history.append(
            {
                "loop": loop + 1,
                "reflection": reflection,
                "population_data": [
                    {
                        "metrics": summary_data[0]["metrics"],
                        "params": summary_data[0]["rewards"],
                    }
                ],
            }
        )

        current_configs = new_configs
        print_new_configs(new_configs)

    return history


def print_training_results_formatted(all_results):
    for loop_idx, loop_data in enumerate(all_results, start=1):
        print(f"\n=== Loop {loop_idx} ===")
        for pop_idx, metrics in enumerate(loop_data, start=1):
            print(f"\nPopulação {pop_idx}:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"   {key:<25s}: {value:>8.4f}")
                else:
                    print(f"   {key:<25s}: {value}")


def print_new_configs(configs):
    print("\n=== Resumo das Novas Configurações ===")
    for idx, cfg in enumerate(configs, start=1):
        reward = cfg["reward_config"].params
        env = cfg["env_config"]
        print(
            f"População {idx}: "
            f"Reward -> (Ori: {reward['scale_orientation']:.4f}, Dist: {reward['scale_distance']:.4f}, "
            f"Time: {reward['scale_time']:.4f}, Obst: {reward['scale_obstacle']:.4f}), "
            f"Domínio -> (Obs%: {env.obstacle_percentage}, Map Size: {env.map_size})"
        )


def print_population_metrics(population_summaries):
    print("\n=== Métricas das Populações ===")
    for i, pop in enumerate(population_summaries):
        print(
            f"População {i+1}: "
            f"Sucesso = {pop['success_percentage']}%, "
            f"Inseguro = {pop['percentage_unsafe']*100}%, "
            f"Vel Angular = {pop['percentage_angular']*100}%, "
            f"Collisions = {pop['avg_collision_steps']:.2f}, "
            f"Goal Steps = {pop['avg_goal_steps']:.2f}"
        )


if __name__ == "__main__":
    robot_config = RobotConfig(
        base_radius=0.105,
        vel_linear=[0.0, 0.22],
        vel_angular=[1.0, 2.84],
        wheel_distance=0.16,
        weight=1.0,
        threshold=0.1,
        collision=0.075,
        path_model="None",
    )
    sensor_config = SensorConfig(fov=270.0, num_rays=5, min_range=0.0, max_range=3.5)
    env_config = EnvConfig(
        scalar=50,
        folder_map="",
        name_map="",
        timestep=1000,
        obstacle_percentage=40.0,
        map_size=5,
    )
    render_config = RenderConfig(controller=False, debug=True, plot=False)

    trainer_config = TrainerConfig(
        pretrained="None",
        use_agents=True,
        max_timestep_global=100000,
        seed=1,
        batch_size=8,
        num_envs=4,
        device="cpu",
        checkpoint=10000,
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
        verbose=True,
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
    all_results = run_multiple_parallel_trainings(
        num_loops, configs, allow_domain_modifications=False, num_populations=pop
    )
