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
from microvault.training.train import Training

cs = ConfigStore.instance()
cs.store(name="trainer_config", node=TrainerConfig)

config_path = "../microvault/microvault/configs/sweep.yaml"
path_sweep = OmegaConf.load(config_path)
sweep_configuration = OmegaConf.to_container(path_sweep, resolve=True)

sweep_id = wandb.sweep(sweep=sweep_configuration, project="rl-sweep")

wandb.login()


@hydra.main(config_path="../configs", config_name="default", version_base="1.2")
def run_experiment(cfg: TrainerConfig) -> None:
    config_default = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    with wandb.init(project="rl-sweep", config=config_default):

        batch_size = wandb.config.batch_size
        epochs = wandb.config.epochs
        learning_rate_actor = wandb.config.learning_rate_actor
        learning_rate_critic = wandb.config.learning_rate_critic
        noise = wandb.config.noise
        nstep = wandb.config.nstep
        weight_decay = wandb.config.weight_decay
        desired_distance = wandb.config.desired_distance
        scalar = wandb.config.scalar
        scalar_decay = wandb.config.scalar_decay
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
            desired_distance=desired_distance,
            scalar=scalar,
            scalar_decay=scalar_decay,
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
            max_episode=1000,
            robot=robot,
            generator=generate,
            agent=agent,
            collision=collision,
        )

        trainer = Training(env, agent, replaybuffer)

        if cfg.engine.visualize:
            state = env.reset()
            env.render()

        else:
            scalar_deque = deque(maxlen=100)
            scalar_decay_deque = deque(maxlen=100)
            distance_deque = deque(maxlen=100)
            scores_deque = deque(maxlen=100)
            time_foward_deque = deque(maxlen=100)

            success_count = 0
            failure_count = 0
            last_success_rate = 0
            last_failure_rate = 0

            scores = []

            for epoch in np.arange(1, epochs + 1):
                (
                    critic_loss,
                    actor_loss,
                    q,
                    max_q,
                    intrinsic_reward,
                    error,
                    score,
                    scalar_deque,
                    scalar_decay_deque,
                    distance_deque,
                    elapsed_time,
                ) = trainer.train_one_epoch(
                    batch_size,
                    cfg.environment.timestep,
                    scalar_deque,
                    scalar_decay_deque,
                    distance_deque,
                )

                mean_scalar_deque = np.mean(scalar_deque)
                mean_scalar_decay_deque = np.mean(scalar_decay_deque)
                mean_distance_deque = np.mean(distance_deque)
                mean_time_foward_deque = np.mean(time_foward_deque)
                scores_deque.append(score)
                scores.append(score)
                mean_score = np.mean(scores_deque)

                wandb.log(
                    {
                        "epoch": epoch,
                        "mean_scalar": mean_scalar_deque,
                        "mean_scalar_decay": mean_scalar_decay_deque,
                        "mean_distance": mean_distance_deque,
                        "critic_loss": critic_loss,
                        "actor_loss": actor_loss,
                        "q": q,
                        "max_q": max_q,
                        "time_per_epoch": elapsed_time,
                        "mean_reward": mean_score,
                        "mean_time_foward": mean_time_foward_deque,
                        "intrinsic_reward": intrinsic_reward,
                        "error": error,
                    }
                )


if __name__ == "__main__":
    wandb.agent(sweep_id, function=run_experiment, count=8)
