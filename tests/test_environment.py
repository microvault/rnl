import gym
import pytest
from omegaconf import OmegaConf

from rnl.algorithms.agent import Agent
from rnl.engine.collision import Collision
from rnl.environment.generate_world import GenerateWorld, Generator
from rnl.environment.robot import Robot
from rnl.models.model import QModel

config_path = "../rnl/rnl/configs/default.yaml"
path = OmegaConf.load(config_path)
cfg = OmegaConf.to_container(path, resolve=True)


@pytest.fixture
def environment_instance():

    model = QModel(
        state_size=cfg["environment"]["state_size"],
        action_size=cfg["environment"]["action_size"],
        fc1_units=cfg["network"]["layers_model_l1"],
        fc2_units=cfg["network"]["layers_model_l2"],
        device=cfg["engine"]["device"],
        batch_size=cfg["engine"]["batch_size"],
    )

    agent = Agent(
        model=model,
        state_size=cfg["environment"]["state_size"],
        action_size=cfg["environment"]["action_size"],
        gamma=cfg["agent"]["gamma"],
        tau=cfg["agent"]["tau"],
        lr_model=cfg["agent"]["lr_model"],
        weight_decay=cfg["agent"]["weight_decay"],
        device=cfg["engine"]["device"],
        pretrained=cfg["engine"]["pretrained"],
    )

    collision = Collision()
    generate = GenerateWorld()

    generate = Generator(
        collision=collision,
        generate=generate,
        grid_lenght=cfg["environment"]["grid_lenght"],
        random=cfg["environment"]["random"],
    )

    robot = Robot(
        collision=collision,
        wheel_radius=cfg["robot"]["wheel_radius"],
        wheel_base=cfg["robot"]["wheel_base"],
        fov=cfg["robot"]["fov"],
        num_rays=cfg["robot"]["num_rays"],
        max_range=cfg["robot"]["max_range"],
    )

    env = gym.make(
        "rnl/NaviEnv-v0",
        rgb_array=False,
        robot=robot,
        generator=generate,
        agent=agent,
        collision=collision,
        timestep=cfg["environment"]["timestep"],
        threshold=cfg["environment"]["threshold"],
        grid_lenght=cfg["environment"]["grid_lenght"],
        state_size=cfg["environment"]["state_size"],
    )

    return env
