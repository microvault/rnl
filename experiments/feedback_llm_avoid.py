import rnl as vault
import os

gemini_api_key = os.environ.get("GEMINI_API_KEY")

task = (
    "- O mapa é 2x2 m, onde o robô começa sempre de um dos lado do quadrado com randomização de ângulo theta, e o target randomizado nas outra extremidade onde o robô foi iniciado.\n"
    "- Possui somente um unico obstaculo no meio do do mapa com tamanho de 0.5x0.5 cm.\n"
    "- 500 steps totais, mas ~80 já levam o robô de ponta a ponta do mapa.\n"
    "- 3 ações: 0 = frente, 1 = esquerda, 2 = direita.\n"
    "- 6 estados: 3 valores de leituras de LiDAR, 1 valor de distância ao objetivo, 1 valor de ângulo ao objetivo e 1 valor do estado do robô (frente/giro).\n"
)

def train():
    param_robot = vault.robot(
        base_radius=0.105,
        max_vel_linear=0.22,
        max_vel_angular=2.84,
        wheel_distance=0.16,
        weight=1.0,
        threshold=0.10,
        collision=0.075,
        path_model="",
    )
    param_sensor = vault.sensor(
        fov=90.0,
        num_rays=3,
        min_range=0.0,
        max_range=3.5,
    )
    param_env = vault.make(
        scalar=30,
        folder_map="",
        name_map="",
        max_timestep=500,
        type="avoid",
        grid_size=[0, 0],
        map_size=0,
        noise=False,
        obstacle_percentage=0
    )
    param_render = vault.render(controller=False, debug=True)

    model = vault.Trainer(
        param_robot,
        param_sensor,
        param_env,
        param_render,
    )
    model.learn(
        population=3,
        loop_feedback=20,
        description_task=task,
        pretrained="None",
        use_agents=True,
        max_timestep_global=20000,
        seed=1,
        hidden_size=[16, 16],
        activation="LeakyReLU",
        batch_size=64,
        num_envs=12,
        device="mps",
        checkpoint=20000,
        checkpoint_path="./checkpoint_models_llm_avoid",
        use_wandb=False,
        wandb_api_key="",
        llm_api_key=str(gemini_api_key),
        lr=1e-3,
        learn_step=256,
        gae_lambda=0.90,
        ent_coef=0.002,
        vf_coef=0.4,
        max_grad_norm=0.5,
        update_epochs=3,
        clip_range_vf=0.2,
        target_kl=0.025,
        name="",
        verbose=False,
        policy="PPO",
    )

if __name__ == "__main__":
    train()
