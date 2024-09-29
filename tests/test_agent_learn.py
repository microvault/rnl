import os

import numpy as np
import pytest
from omegaconf import OmegaConf

from rnl.algorithms.agent import Agent
from rnl.components.replaybuffer import ReplayBuffer
from rnl.models.model import QModel

config_path = "../rnl/rnl/configs/default.yaml"
path = OmegaConf.load(config_path)
cfg = OmegaConf.to_container(path, resolve=True)


@pytest.fixture
def model_instance_default():
    return QModel(
        state_size=cfg["environment"]["state_size"],
        action_size=cfg["environment"]["action_size"],
        fc1_units=cfg["network"]["layers_model_l1"],
        fc2_units=cfg["network"]["layers_model_l2"],
        device=cfg["engine"]["device"],
        batch_size=cfg["engine"]["batch_size"],
    )


@pytest.fixture
def agent_instance_config(model_instance_default):

    agent = Agent(
        model=model_instance_default,
        state_size=cfg["environment"]["state_size"],
        action_size=cfg["environment"]["action_size"],
        gamma=cfg["agent"]["gamma"],
        tau=cfg["agent"]["tau"],
        lr_model=cfg["agent"]["lr_model"],
        weight_decay=cfg["agent"]["weight_decay"],
        device=cfg["engine"]["device"],
        pretrained=cfg["engine"]["pretrained"],
    )

    return agent


@pytest.fixture
def replay_buffer():
    return ReplayBuffer(
        buffer_size=cfg["replay_buffer"]["buffer_size"],
        batch_size=32,
        state_dim=cfg["environment"]["state_size"],
        action_dim=cfg["environment"]["action_size"],
        device=cfg["engine"]["device"],
    )


def test_learn(agent_instance_config, replay_buffer):
    agent = agent_instance_config
    memory = replay_buffer

    state_size = cfg["environment"]["state_size"]

    while True:
        state = np.random.randn(state_size).astype(np.float32)
        action = np.random.choice([0, 1, 2, 3])
        reward = np.random.randn(1).astype(np.float32)
        next_state = np.random.randn(state_size).astype(np.float32)
        done = np.random.choice([0, 1])

        memory.add(state, action, reward[0], next_state, np.bool_(done))

        if len(memory) > 32:
            break

    n_iteration = 5

    results = agent.learn(memory, n_iteration)

    assert isinstance(results, tuple)
    assert len(results) == 3
    for result in results:
        assert isinstance(result, float)


def test_save_model(agent_instance_config):
    filename = "network"
    version = "v1"
    agent_instance_config.save(filename, version)

    critic_path = filename + "_model_" + version + ".pth"
    critic_optimizer_path = filename + "_model_optimizer_" + version + ".pth"

    for path in [critic_path, critic_optimizer_path]:
        if os.path.exists(path):
            os.remove(path)
