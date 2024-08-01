import numpy as np
import pytest
from omegaconf import OmegaConf

from microvault.components.replaybuffer import ReplayBuffer

config_path = "../microvault/microvault/configs/default.yaml"
path = OmegaConf.load(config_path)
cfg = OmegaConf.to_container(path, resolve=True)


@pytest.fixture
def replay_buffer():
    return ReplayBuffer(
        buffer_size=cfg["replay_buffer"]["buffer_size"],
        batch_size=32,
        state_dim=cfg["environment"]["state_size"],
        action_dim=cfg["environment"]["action_size"],
        device=cfg["engine"]["device"],
    )


def test_add_experience(replay_buffer):
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

    assert len(memory) >= 32


def test_sample_experiences(replay_buffer):
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

    states, actions, rewards, next_states, dones = memory.sample()

    assert states.shape[0] == 32
    assert actions.shape[0] == 32
    assert rewards.shape[0] == 32
    assert next_states.shape[0] == 32
    assert dones.shape[0] == 32
