#!/usr/bin/env python3
import wandb
import rnl as vault

sweep_config = {
    'method': 'random',
    'metric': {'name': 'time_score_mean', 'goal': 'maximize'},
    'parameters': {
        'max_timestep_global': {'values': [20000, 30000, 40000]},
        'activation': {'values': ['ReLU', 'LeakyReLU', 'Tanh', 'Sigmoid']},
        'batch_size': {'values': [8, 16, 32]},
        'lr': {'values': [1e-2, 1e-3, 1e-4, 1e-5]},
        'learn_step': {'values': [128, 256, 512]},
        'gae_lambda': {'min': 0.90, 'max': 0.95},
        'ent_coef': {'min': 0.001, 'max': 0.005},
        'vf_coef': {'min': 0.3, 'max': 0.5},
        'update_epochs': {'values': [3, 5, 7, 9]},
        'clip_range_vf': {'min': 0.2, 'max': 0.5},
        'target_kl': {'values': [0.025, 0.05]},
    }
}

def train():
    wandb.init(project="sweep-avoid")
    cfg = wandb.config

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
    metrics = model.learn(
        population=0,
        loop_feedback=0,
        description_task="",
        pretrained="None",
        use_agents=False,
        max_timestep_global=30000,
        seed=1,
        hidden_size=[16, 16],
        activation=cfg.activation,
        batch_size=cfg.batch_size,
        num_envs=8,
        device="mps",
        checkpoint=40001,
        checkpoint_path=".",
        use_wandb=False,
        wandb_api_key="",
        llm_api_key="",
        lr=cfg.lr,
        learn_step=cfg.learn_step,
        gae_lambda=cfg.gae_lambda,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=0.5,
        update_epochs=cfg.update_epochs,
        clip_range_vf=cfg.clip_range_vf,
        target_kl=cfg.target_kl,
        name="",
        verbose=False,
        policy="PPO",
    )

    wandb.log({'time_score_mean': metrics['time_score_mean']})

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="sweep-avoid")
    wandb.agent(sweep_id, function=train)
