from collections import deque

import gym
import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

import wandb
from microvault.algorithms.agent import Agent
from microvault.components.replaybuffer import ReplayBuffer
from microvault.configs.config import TrainerConfig
from microvault.engine.collision import Collision
from microvault.engine.engine import Engine
from microvault.environment.generate_world import GenerateWorld, Generator
from microvault.environment.robot import Robot
from microvault.models.model import QModel
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
        learning_rate_model = wandb.config.learning_rate_model
        weight_decay = wandb.config.weight_decay
        seed = wandb.config.seed
        layers_model_l1 = wandb.config.layers_model_l1
        layers_model_l2 = wandb.config.layers_model_l2

        engine = Engine(seed=seed, device=cfg.engine.device)

        engine.seed_everything()
        engine.set_device()

        replaybuffer = ReplayBuffer(
            buffer_size=cfg.replay_buffer.buffer_size,
            batch_size=batch_size,
            state_dim=cfg.environment.state_size,
            action_dim=cfg.environment.action_size,
            device=cfg.engine.device,
        )

        model = QModel(
            state_size=cfg.environment.state_size,
            action_size=cfg.environment.action_size,
            fc1_units=layers_model_l1,
            fc2_units=layers_model_l2,
            batch_size=batch_size,
            device=cfg.engine.device,
        )

        agent = Agent(
            model=model,
            state_size=cfg.environment.state_size,
            action_size=cfg.environment.action_size,
            gamma=cfg.agent.gamma,
            tau=cfg.agent.tau,
            lr_model=learning_rate_model,
            weight_decay=weight_decay,
            device=cfg.engine.device,
            pretrained=cfg.engine.pretrained,
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
            wheel_radius=cfg.robot.wheel_radius,
            wheel_base=cfg.robot.wheel_base,
            fov=cfg.robot.fov,
            num_rays=cfg.robot.num_rays,
            max_range=cfg.robot.max_range,
        )

        env = gym.make(
            "microvault/NaviEnv-v0",
            rgb_array=False,
            robot=robot,
            generator=generate,
            agent=agent,
            collision=collision,
            state_size=cfg.environment.state_size,
            fps=cfg.environment.fps,
            timestep=cfg.environment.timestep,
            threshold=cfg.environment.threshold,
            grid_lenght=cfg.environment.grid_lenght,
        )

        trainer = Training(env, agent, replaybuffer)

        scores_deque = deque(maxlen=100)
        scores = []

        eps_start = 1.0
        eps_end = 0.01
        eps_decay = 0.995

        eps = eps_start
        for epoch in np.arange(1, epochs + 1):
            (
                model_loss,
                q,
                max_q,
                mean_action,
                score,
                elapsed_time,
            ) = trainer.train_one_epoch(
                batch_size,
                cfg.environment.timestep,
                eps,
            )

            eps = max(eps_end, eps_decay * eps)

            scores_deque.append(score)
            scores.append(score)
            mean_score = np.mean(scores_deque)

            print(
                "\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}".format(
                    epoch, mean_score, score
                )
            )

            wandb.log(
                {
                    "epoch": epoch,
                    "model_loss": model_loss,
                    "q": q,
                    "max_q": max_q,
                    "time_per_epoch": elapsed_time,
                    "mean_reward": mean_score,
                }
            )


if __name__ == "__main__":
    wandb.agent(sweep_id, function=finetune_experiment, count=8)
