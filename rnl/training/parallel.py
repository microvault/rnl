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


# Função wrapper que chama o treinamento e retorna métricas
def train_worker(robot_config, sensor_config, env_config, render_config, trainer_config, network_config):
    # Aqui você pode chamar diretamente a função 'training(...)'
    # ou criar uma cópia dela (se precisar modificar algo).
    training(
        robot_config,
        sensor_config,
        env_config,
        render_config,
        trainer_config,
        network_config,
    )

    # Exemplo de métricas retornadas (você pode coletar de dentro do training)
    metrics = {
        "reward_mean": 123.45,
        "episodes": 1000,
    }
    return metrics

def run_parallel_trainings(list_of_configs):
    start_time = time.time()
    results = []

    # Multiprocessing
    with multiprocessing.Pool(processes=len(list_of_configs)) as pool:
        # Dispara tudo em paralelo
        async_results = []
        for cfg in list_of_configs:
            args = (
                cfg["robot_config"],
                cfg["sensor_config"],
                cfg["env_config"],
                cfg["render_config"],
                cfg["trainer_config"],
                cfg["network_config"],
            )
            async_results.append(pool.apply_async(train_worker, args))

        # Coleta os resultados de cada processo
        for res in async_results:
            results.append(res.get())

    print(f"Tempo total: {time.time() - start_time:.2f} seg")
    return results

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
    env_config = EnvConfig(scalar=100, folder_map="", name_map="", timestep=1000)
    render_config = RenderConfig(controller=False, debug=True, plot=False)

    trainer_config = TrainerConfig(
        pretrained="None",
        use_agents=False,
        max_timestep_global=10000,
        seed=1,
        batch_size=8,
        num_envs=2,
        device="cpu",
        checkpoint_path="ppo_policy_network",
        use_wandb=False,
        wandb_api_key="",
        llm_api_key="",
        lr=0.0003,
        learn_step=512,
        gae_lambda=0.95,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
        update_epochs=10,
        name="rnl-v1",
        verbose=False
    )

    network_config = NetworkConfig(
        hidden_size=[20, 10],
        mlp_activation="ReLU",
        type_model="MlpPolicy",
    )

    # Exemplos de configurações diferentes pra cada treinamento
    config1 = {
        "robot_config": robot_config,
        "sensor_config": sensor_config,
        "env_config": env_config,
        "render_config": render_config,
        "trainer_config": trainer_config,
        "network_config": network_config,
    }
    config2 = {
        "robot_config": robot_config,
        "sensor_config": sensor_config,
        "env_config": env_config,
        "render_config": render_config,
        "trainer_config": trainer_config,
        "network_config": network_config,
    }

    config3 = {
        "robot_config": robot_config,
        "sensor_config": sensor_config,
        "env_config": env_config,
        "render_config": render_config,
        "trainer_config": trainer_config,
        "network_config": network_config,
    }

    config4 = {
        "robot_config": robot_config,
        "sensor_config": sensor_config,
        "env_config": env_config,
        "render_config": render_config,
        "trainer_config": trainer_config,
        "network_config": network_config,
    }

    config5 = {
        "robot_config": robot_config,
        "sensor_config": sensor_config,
        "env_config": env_config,
        "render_config": render_config,
        "trainer_config": trainer_config,
        "network_config": network_config,
    }

    config6 = {
        "robot_config": robot_config,
        "sensor_config": sensor_config,
        "env_config": env_config,
        "render_config": render_config,
        "trainer_config": trainer_config,
        "network_config": network_config,
    }

    # Rode quantos quiser
    configs = [config1, config2]
    all_results = run_parallel_trainings(configs)

    # all_results vai ter a lista de métricas de cada treino
    print("Resultados finais:", all_results)
