import gym
import hydra
from hydra.core.config_store import ConfigStore

from microvault.algorithms.agent import Agent
from microvault.configs.config import TrainerConfig
from microvault.engine.collision import Collision
from microvault.engine.engine import Engine
from microvault.environment.generate_world import GenerateWorld, Generator
from microvault.environment.robot import Robot
from microvault.models.model import QModel

cs = ConfigStore.instance()
cs.store(name="trainer_config", node=TrainerConfig)


@hydra.main(config_path="../configs", config_name="default", version_base="1.2")
def eval_experiment(cfg=TrainerConfig) -> None:
    engine = Engine(seed=cfg.engine.seed, device=cfg.engine.device)

    engine.seed_everything()
    engine.set_device()

    model = QModel(
        state_size=cfg.environment.state_size,
        action_size=cfg.environment.action_size,
        fc1_units=cfg.network.layers_model_l1,
        fc2_units=cfg.network.layers_model_l2,
        batch_size=cfg.engine.batch_size,
        device=cfg.engine.device,
    )

    agent = Agent(
        model=model,
        state_size=cfg.environment.state_size,
        action_size=cfg.environment.action_size,
        gamma=cfg.agent.gamma,
        tau=cfg.agent.tau,
        lr_model=cfg.agent.lr_model,
        weight_decay=cfg.agent.weight_decay,
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
        rgb_array=True,
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

    env.reset()
    env.render()


if __name__ == "__main__":
    eval_experiment()
