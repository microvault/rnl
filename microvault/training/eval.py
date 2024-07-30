import gym
import hydra
from hydra.core.config_store import ConfigStore

from microvault.algorithms.agent import Agent
from microvault.engine.collision import Collision
from microvault.environment.generate_world import GenerateWorld, Generator
from microvault.environment.robot import Robot
from microvault.models.model import ModelActor, ModelCritic
from microvault.training.config import TrainerConfig
from microvault.training.engine import Engine

cs = ConfigStore.instance()
cs.store(name="trainer_config", node=TrainerConfig)


@hydra.main(config_path="../configs", config_name="default", version_base="1.2")
def eval_experiment(cfg=TrainerConfig) -> None:
    engine = Engine(seed=cfg.engine.seed, device=cfg.engine.device)

    engine.seed_everything()
    engine.set_device()

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
        rgb_array=True,
        max_episode=cfg.engine.num_episodes,
        robot=robot,
        generator=generate,
        agent=agent,
        collision=collision,
        state_size=cfg.environment.state_size,
        fps=cfg.environment.fps,
        timestep=cfg.environment.timestep,
        threshold=cfg.environment.threshold,
        size=cfg.robot.size,
        num_rays=cfg.robot.num_rays,
        fov=cfg.robot.fov,
        max_range=cfg.robot.max_range,
        grid_lenght=cfg.environment.grid_lenght,
    )

    env.reset()
    env.render()


if __name__ == "__main__":
    eval_experiment()
