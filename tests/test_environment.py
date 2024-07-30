import gym
import numpy as np
import pytest
from omegaconf import OmegaConf

from microvault.algorithms.agent import Agent
from microvault.engine.collision import Collision
from microvault.environment.generate_world import GenerateWorld, Generator
from microvault.environment.robot import Robot
from microvault.models.model import ModelActor, ModelCritic

config_path = "../microvault/microvault/configs/default.yaml"
path = OmegaConf.load(config_path)
cfg = OmegaConf.to_container(path, resolve=True)


@pytest.fixture
def environment_instance():

    modelActor = ModelActor(
        state_dim=cfg["environment"]["state_size"],
        action_dim=cfg["environment"]["action_size"],
        max_action=cfg["robot"]["max_action"],
        l1=cfg["network"]["layers_actor_l1"],
        l2=cfg["network"]["layers_actor_l2"],
        device=cfg["engine"]["device"],
        noise_std=cfg["network"]["noise_std"],
        batch_size=cfg["engine"]["batch_size"],
    )

    modelCritic = ModelCritic(
        state_dim=cfg["environment"]["state_size"],
        action_dim=cfg["environment"]["action_size"],
        l1=cfg["network"]["layers_critic_l1"],
        l2=cfg["network"]["layers_critic_l2"],
        device=cfg["engine"]["device"],
        batch_size=cfg["engine"]["batch_size"],
    )

    agent = Agent(
        modelActor=modelActor,
        modelCritic=modelCritic,
        state_size=cfg["environment"]["state_size"],
        action_size=cfg["environment"]["action_size"],
        max_action=cfg["robot"]["max_action"],
        min_action=cfg["robot"]["min_action"],
        update_every_step=cfg["agent"]["update_every_step"],
        gamma=cfg["agent"]["gamma"],
        tau=cfg["agent"]["tau"],
        lr_actor=cfg["agent"]["lr_actor"],
        lr_critic=cfg["agent"]["lr_critic"],
        weight_decay=cfg["agent"]["weight_decay"],
        noise=cfg["agent"]["noise"],
        noise_clip=cfg["agent"]["noise_clip"],
        device=cfg["engine"]["device"],
        pretrained=cfg["engine"]["pretrained"],
        nstep=cfg["agent"]["nstep"],
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
        time=cfg["environment"]["timestep"],
        min_radius=cfg["robot"]["min_radius"],
        max_radius=cfg["robot"]["max_radius"],
        max_grid=cfg["environment"]["grid_lenght"],
        wheel_radius=cfg["robot"]["wheel_radius"],
        wheel_base=cfg["robot"]["wheel_base"],
        fov=cfg["robot"]["fov"],
        num_rays=cfg["robot"]["num_rays"],
        max_range=cfg["robot"]["max_range"],
    )

    env = gym.make(
        "microvault/NaviEnv-v0",
        rgb_array=False,
        max_episode=cfg["engine"]["num_episodes"],
        robot=robot,
        generator=generate,
        agent=agent,
        collision=collision,
        timestep=cfg["environment"]["timestep"],
        threshold=cfg["environment"]["threshold"],
        num_rays=cfg["robot"]["num_rays"],
        fov=cfg["robot"]["fov"],
        max_range=cfg["robot"]["max_range"],
        grid_lenght=cfg["environment"]["grid_lenght"],
        state_size=cfg["environment"]["state_size"],
    )

    return env


def test_step(environment_instance):
    environment_instance.reset()

    for t in range(10):
        action = np.random.uniform(0.0, 1.0, size=2)
        next_state, reward, done, info = environment_instance.step(action)
