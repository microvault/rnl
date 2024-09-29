import numpy as np
import pytest
from omegaconf import OmegaConf

from rnl.algorithms.agent import Agent
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
def agent_instance_default(model_instance_default):
    return Agent(model=model_instance_default)


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


def test_default_values(agent_instance_default):
    assert agent_instance_default.state_size == 13
    assert agent_instance_default.action_size == 4
    assert agent_instance_default.gamma == 0.99
    assert agent_instance_default.tau == 1e-3
    assert agent_instance_default.lr_model == 1e-3
    assert agent_instance_default.weight_decay == 0.0
    assert agent_instance_default.device == "cpu"
    assert not agent_instance_default.pretrained


def test_predict_size_action_with_default(agent_instance_default):
    action = agent_instance_default.predict(np.zeros(13, dtype=np.float32))
    assert action.shape == ()


def test_predict_size_action_with_config(agent_instance_config):
    action = agent_instance_config.predict(np.zeros(13, dtype=np.float32))
    assert action.shape == ()


def test_states_size(agent_instance_config):
    action = agent_instance_config.predict(np.zeros(13, dtype=np.float32))
    assert action.shape == ()


def test_action_type(agent_instance_config):
    action = agent_instance_config.predict(np.zeros(13, dtype=np.float32))
    assert action.dtype == np.int64, "O tipo da ação não é np.int64."


def test_action_within_limits(agent_instance_config):
    action = agent_instance_config.predict(np.zeros(13, dtype=np.float32))
    assert np.all(action >= 0) and np.all(
        action <= 3
    ), "Ação não está dentro dos limites."
