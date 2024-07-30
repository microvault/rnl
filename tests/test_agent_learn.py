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
def agent_instance_config():
    modelActor = ModelActor()
    modelCritic = ModelCritic()

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


@pytest.fixture
def actor_instance_default():
    return ModelActor()


@pytest.fixture
def critic_instance_default():
    return ModelCritic()


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
    episode = 1

    results = agent.learn(memory, n_iteration, episode)

    assert isinstance(results, tuple)
    assert len(results) == 6
    for result in results:
        assert isinstance(result, float)


def test_save_model(agent_instance_config, actor_instance_default, tmpdir):
    model = actor_instance_default
    path = str(tmpdir)
    filename = "network"
    version = "v1"
    agent_instance_config.save(model, path, filename, version)
