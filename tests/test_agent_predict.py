import numpy as np
import pytest
from omegaconf import OmegaConf

from microvault.algorithms.agent import Agent
from microvault.models.model import ModelActor, ModelCritic

config_path = "../microvault/microvault/configs/default.yaml"
path = OmegaConf.load(config_path)
cfg = OmegaConf.to_container(path, resolve=True)


@pytest.fixture
def agent_instance_default():
    modelActor = ModelActor()
    modelCritic = ModelCritic()
    return Agent(modelActor=modelActor, modelCritic=modelCritic)


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


def test_default_values(agent_instance_default):
    assert agent_instance_default.state_size == 14
    assert agent_instance_default.action_size == 2
    assert agent_instance_default.max_action == 1.0
    assert agent_instance_default.min_action == -1.0
    assert agent_instance_default.update_every_step == 2
    assert agent_instance_default.gamma == 0.99
    assert agent_instance_default.tau == 1e-3
    assert agent_instance_default.lr_actor == 1e-3
    assert agent_instance_default.lr_critic == 1e-3
    assert agent_instance_default.weight_decay == 0.0
    assert agent_instance_default.noise == 0.2
    assert agent_instance_default.noise_clip == 0.5
    assert agent_instance_default.device == "cpu"
    assert not agent_instance_default.pretrained
    assert agent_instance_default.nstep == 1


def test_predict_size_action_with_default(agent_instance_default):
    action = agent_instance_default.predict(np.zeros(14, dtype=np.float32))
    assert action.shape == (2,)


def test_predict_size_action_with_config(agent_instance_config):
    action = agent_instance_config.predict(np.zeros(14, dtype=np.float32))
    assert action.shape == (2,)


def test_states_size(agent_instance_config):
    action = agent_instance_config.predict(np.zeros(14, dtype=np.float32))
    assert action.shape == (2,)


def test_action_type(agent_instance_config):
    action = agent_instance_config.predict(np.zeros(14, dtype=np.float32))
    assert action.dtype == np.float32, "O tipo da ação não é np.float32."


def test_action_within_limits(agent_instance_config):
    action = agent_instance_config.predict(np.zeros(14, dtype=np.float32))
    assert np.all(action >= -1.0) and np.all(
        action <= 1.0
    ), "Ação não está dentro dos limites."


# TODO: teste scalar
