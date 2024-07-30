from collections import deque

import gym
import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

import wandb
from microvault.algorithms.agent import Agent
from microvault.components.replaybuffer import ReplayBuffer
from microvault.engine.collision import Collision
from microvault.environment.generate_world import GenerateWorld, Generator
from microvault.environment.robot import Robot
from microvault.models.model import ModelActor, ModelCritic
from microvault.training.config import TrainerConfig
from microvault.training.engine import Engine
from microvault.training.training import Training

cs = ConfigStore.instance()
cs.store(name="trainer_config", node=TrainerConfig)

wandb.login()

config_path = "../microvault/microvault/configs/sweep.yaml"
path_sweep = OmegaConf.load(config_path)
sweep_configuration = OmegaConf.to_container(path_sweep, resolve=True)

sweep_id = wandb.sweep(sweep=sweep_configuration, project="rl-sweep")


@hydra.main(config_path="../configs", config_name="default", version_base="1.2")
def finetune_experiment(cfg: TrainerConfig) -> None:
    config_default = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    with wandb.init(project="rl-sweep", config=config_default):

        batch_size = wandb.config.batch_size
        epochs = wandb.config.epochs
        learning_rate_actor = wandb.config.learning_rate_actor
        learning_rate_critic = wandb.config.learning_rate_critic
        noise = wandb.config.noise
        nstep = wandb.config.nstep
        weight_decay = wandb.config.weight_decay
        noise_std = wandb.config.noise_std
        seed = wandb.config.seed
        layers_actor_l1 = wandb.config.layers_actor_l1
        layers_actor_l2 = wandb.config.layers_actor_l2
        layers_critic_l1 = wandb.config.layers_critic_l1
        layers_critic_l2 = wandb.config.layers_critic_l2

        engine = Engine(seed=seed, device=cfg.engine.device)

        engine.seed_everything()
        engine.set_device()

        replaybuffer = ReplayBuffer(
            buffer_size=cfg.replay_buffer.buffer_size,
            batch_size=batch_size,
            gamma=cfg.agent.gamma,
            nstep=nstep,
            state_dim=cfg.environment.state_size,
            action_dim=cfg.environment.action_size,
            device=cfg.engine.device,
        )

        modelActor = ModelActor(
            state_dim=cfg.environment.state_size,
            action_dim=cfg.environment.action_size,
            max_action=cfg.robot.max_action,
            l1=layers_actor_l1,
            l2=layers_actor_l2,
            device=cfg.engine.device,
            noise_std=noise_std,
            batch_size=batch_size,
        )

        modelCritic = ModelCritic(
            state_dim=cfg.environment.state_size,
            action_dim=cfg.environment.action_size,
            l1=layers_critic_l1,
            l2=layers_critic_l2,
            device=cfg.engine.device,
            batch_size=batch_size,
        )

        agent = Agent(
            modelActor=modelActor,
            modelCritic=modelCritic,
            state_size=cfg.environment.state_size,
            action_size=cfg.environment.action_size,
            max_action=cfg.robot.max_action,
            min_action=cfg.robot.min_action,
            update_every_step=cfg.agent.update_every_step,
            gamma=cfg.agent.gamma,
            tau=cfg.agent.tau,
            lr_actor=learning_rate_actor,
            lr_critic=learning_rate_critic,
            weight_decay=weight_decay,
            noise=noise,
            noise_clip=cfg.agent.noise_clip,
            device=cfg.engine.device,
            pretrained=cfg.engine.pretrained,
            nstep=nstep,
        )

        collision = Collision()
        generate = GenerateWorld()

        generate = Generator(
            collision=collision,
            generate=generate,
            grid_lenght=cfg.environment.grid_lenght,
            random=cfg.environment.random,
        )

        robot = Robot(
            collision=collision,
            time=cfg.environment.timestep,
            min_radius=cfg.robot.min_radius,
            max_radius=cfg.robot.max_radius,
            max_grid=cfg.environment.grid_lenght,
            wheel_radius=cfg.robot.wheel_radius,
            wheel_base=cfg.robot.wheel_base,
            fov=cfg.robot.fov,
            num_rays=cfg.robot.num_rays,
            max_range=cfg.robot.max_range,
        )

        env = gym.make(
            "microvault/NaviEnv-v0",
            rgb_array=False,
            max_episode=epochs,
            robot=robot,
            generator=generate,
            agent=agent,
            collision=collision,
            timestep=cfg.environment.timestep,
            threshold=cfg.environment.threshold,
            num_rays=cfg.robot.num_rays,
            fov=cfg.robot.fov,
            max_range=cfg.robot.max_range,
            grid_lenght=cfg.environment.grid_lenght,
            state_size=cfg.environment.state_size,
        )

        trainer = Training(env, agent, replaybuffer)

        scores_deque = deque(maxlen=100)
        scores = []

        for epoch in np.arange(1, epochs + 1):
            print(f"Epoch: {epoch}/{epochs}")
            (
                critic_loss,
                actor_loss,
                q,
                max_q,
                intrinsic_reward,
                error,
                score,
                elapsed_time,
            ) = trainer.train_one_epoch(
                batch_size,
                cfg.environment.timestep,
            )

            scores_deque.append(score)
            scores.append(score)
            mean_score = np.mean(scores_deque)

            wandb.log(
                {
                    "epoch": epoch,
                    "critic_loss": critic_loss,
                    "actor_loss": actor_loss,
                    "q": q,
                    "max_q": max_q,
                    "time_per_epoch": elapsed_time,
                    "mean_reward": mean_score,
                    "intrinsic_reward": intrinsic_reward,
                    "error": error,
                }
            )


if __name__ == "__main__":
    wandb.agent(sweep_id, function=finetune_experiment, count=8)