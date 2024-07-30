import os

import numpy as np
import pytest
from omegaconf import OmegaConf

from microvault.algorithms.agent import Agent
from microvault.components.replaybuffer import ReplayBuffer
from microvault.models.model import ModelActor, ModelCritic

config_path = "../microvault/microvault/configs/default.yaml"
path = OmegaConf.load(config_path)
cfg = OmegaConf.to_container(path, resolve=True)


@pytest.fixture
def actor_instance_default():
    return ModelActor(
        state_dim=cfg["environment"]["state_size"],
        action_dim=cfg["environment"]["action_size"],
        max_action=cfg["robot"]["max_action"],
        l1=cfg["network"]["layers_actor_l1"],
        l2=cfg["network"]["layers_actor_l2"],
        device=cfg["engine"]["device"],
        batch_size=cfg["engine"]["batch_size"],
    )


@pytest.fixture
def critic_instance_default():
    return ModelCritic(
        state_dim=cfg["environment"]["state_size"],
        action_dim=cfg["environment"]["action_size"],
        l1=cfg["network"]["layers_critic_l1"],
        l2=cfg["network"]["layers_critic_l2"],
        device=cfg["engine"]["device"],
        batch_size=cfg["engine"]["batch_size"],
    )


@pytest.fixture
def agent_instance_config(actor_instance_default, critic_instance_default):

    agent = Agent(
        modelActor=actor_instance_default,
        modelCritic=critic_instance_default,
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

    return agent


@pytest.fixture
def replay_buffer():
    return ReplayBuffer(
        buffer_size=cfg["replay_buffer"]["buffer_size"],
        batch_size=32,
        gamma=cfg["agent"]["gamma"],
        nstep=cfg["agent"]["nstep"],
        state_dim=cfg["environment"]["state_size"],
        action_dim=cfg["environment"]["action_size"],
        device=cfg["engine"]["device"],
    )


def test_learn(agent_instance_config, replay_buffer):
    agent = agent_instance_config
    memory = replay_buffer

    state_size = cfg["environment"]["state_size"]
    action_size = cfg["environment"]["action_size"]

    while True:
        state = np.random.randn(state_size).astype(np.float32)
        action = np.random.randn(action_size).astype(np.float32)
        reward = np.random.randn(1).astype(np.float32)
        next_state = np.random.randn(state_size).astype(np.float32)
        done = np.random.choice([0, 1])

        memory.add(state, action, reward[0], next_state, np.bool_(done))

        if len(memory) > 32:
            break

    n_iteration = 5

    results = agent.learn(memory, n_iteration)

    assert isinstance(results, tuple)
    assert len(results) == 6
    for result in results:
        assert isinstance(result, float)


def test_save_model(agent_instance_config):
    filename = "network"
    version = "v1"
    agent_instance_config.save(filename, version)

    critic_path = filename + "_critic_" + version + ".pth"
    critic_optimizer_path = filename + "_critic_optimizer_" + version + ".pth"
    actor_path = filename + "_actor_" + version + ".pth"
    actor_optimizer_path = filename + "_actor_optimizer_" + version + ".pth"

    for path in [critic_path, critic_optimizer_path, actor_path, actor_optimizer_path]:
        if os.path.exists(path):
            os.remove(path)
