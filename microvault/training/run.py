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

wandb.login()


@hydra.main(config_path="../configs", config_name="default", version_base="1.2")
def run_experiment(cfg=TrainerConfig) -> None:
    config_default = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    with wandb.init(project="rl-sweep", config=config_default):

        engine = Engine(seed=cfg.engine.seed, device=cfg.engine.device)

        engine.seed_everything()
        engine.set_device()

        replaybuffer = ReplayBuffer(
            buffer_size=cfg.replay_buffer.buffer_size,
            batch_size=cfg.engine.batch_size,
            gamma=cfg.agent.gamma,
            nstep=cfg.agent.nstep,
            state_dim=cfg.environment.state_size,
            action_dim=cfg.environment.action_size,
            device=cfg.engine.device,
        )

        modelActor = ModelActor(
            state_dim=cfg.environment.state_size,
            action_dim=cfg.environment.action_size,
            max_action=cfg.robot.max_action,
            l1=cfg.network.layers_actor_l1,
            l2=cfg.network.layers_actor_l2,
            device=cfg.engine.device,
            batch_size=cfg.engine.batch_size,
        )

        modelCritic = ModelCritic(
            state_dim=cfg.environment.state_size,
            action_dim=cfg.environment.action_size,
            l1=cfg.network.layers_critic_l1,
            l2=cfg.network.layers_critic_l2,
            device=cfg.engine.device,
            batch_size=cfg.engine.batch_size,
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
            lr_actor=cfg.agent.lr_actor,
            lr_critic=cfg.agent.lr_critic,
            weight_decay=cfg.agent.weight_decay,
            noise=cfg.agent.noise,
            noise_clip=cfg.agent.noise_clip,
            device=cfg.engine.device,
            pretrained=cfg.engine.pretrained,
            nstep=cfg.agent.nstep,
            desired_distance=cfg.noise_layer.desired_distance,
            scalar=cfg.noise_layer.scalar,
            scalar_decay=cfg.noise_layer.scalar_decay,
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
            env.reset()
            env.render()

        else:
            scalar_deque = deque(maxlen=100)
            scalar_decay_deque = deque(maxlen=100)
            distance_deque = deque(maxlen=100)
            scores_deque = deque(maxlen=100)
            scores = []

            for epoch in np.arange(1, cfg.engine.epochs + 1):

                print("Epoch: ", epoch)

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
                    cfg.engine.batch_size,
                    cfg.environment.timestep,
                    scalar_deque,
                    scalar_decay_deque,
                    distance_deque,
                )

                mean_scalar_deque = np.mean(scalar_deque)
                mean_scalar_decay_deque = np.mean(scalar_decay_deque)
                mean_distance_deque = np.mean(distance_deque)
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
                        "intrinsic_reward": intrinsic_reward,
                        "error": error,
                    }
                )


if __name__ == "__main__":
    run_experiment()
