import pytest
from omegaconf import OmegaConf
from torch import optim

from rnl.engine.sanity import Sanity
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
def sanity_instance():
    return Sanity()


@pytest.fixture
def optimizer_instance(model_instance_default):
    return optim.Adam(model_instance_default.parameters(), lr=0.001)


@pytest.fixture
def registered_optimizer(sanity_instance, optimizer_instance):
    sanity_instance.register(optimizer_instance)
    return sanity_instance


def test_model(model_instance_default):
    assert model_instance_default


def test_changing_parameters_model_actor(model_instance_default, registered_optimizer):
    registered_optimizer.add_module_changing_check(model_instance_default)


def test_unchanging_parameters_model_actor(
    model_instance_default, registered_optimizer
):
    registered_optimizer.add_module_unchanging_check(model_instance_default.fc1)


def test_changing_tensor_model_actor(model_instance_default, registered_optimizer):
    registered_optimizer.add_tensor_changing_check(
        model_instance_default.fc1.weight, tensor_name="fc1.weight"
    )


def test_unchanging_tensor_model_actor(model_instance_default, registered_optimizer):
    registered_optimizer.add_tensor_unchanging_check(
        model_instance_default.fc1.bias, tensor_name="fc1.bias"
    )


def test_output_module_model_actor(model_instance_default, registered_optimizer):
    registered_optimizer.add_module_output_range_check(
        model_instance_default, output_range=(-1, 1)
    )


def test_nan_module_model_actor(model_instance_default, registered_optimizer):
    registered_optimizer.add_module_nan_check(model_instance_default)


def test_nan_layer_model_actor(model_instance_default, registered_optimizer):
    registered_optimizer.add_module_nan_check(model_instance_default.fc1)


def test_inf_module_model_actor(model_instance_default, registered_optimizer):
    registered_optimizer.add_module_inf_check(model_instance_default)


def test_inf_layer_model_actor(model_instance_default, registered_optimizer):
    registered_optimizer.add_module_inf_check(model_instance_default.fc1)
