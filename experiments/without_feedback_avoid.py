import rnl as vault

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

    metrics = model.learn(
        population=4,
        loop_feedback=10,
        description_task="",
        pretrained="None",
        use_agents=False,
        max_timestep_global=40000,
        seed=1,
        hidden_size=[16, 16],
        activation="LeakyReLU",
        batch_size=64,
        num_envs=12,
        device="mps",
        checkpoint=40001,
        checkpoint_path="./checkpoint_models_without_feedback_avoid",
        use_wandb=False,
        wandb_api_key="",
        llm_api_key="",
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
