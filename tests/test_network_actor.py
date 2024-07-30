import pytest
import torch
from torch import optim

from microvault.engine.sanity import Sanity
from microvault.models.model import ModelActor


@pytest.fixture
def model_actor_instance():
    return ModelActor(
        state_dim=2,
        action_dim=2,
        max_action=1,
        l1=400,
        l2=300,
        device="cpu",
        noise_std=0.5,
        batch_size=64,
    )


@pytest.fixture
def sanity_instance():
    return Sanity()


@pytest.fixture
def optimizer_instance(model_actor_instance):
    return optim.Adam(model_actor_instance.parameters(), lr=0.001)


@pytest.fixture
def registered_optimizer(sanity_instance, optimizer_instance):
    sanity_instance.register(optimizer_instance)
    return sanity_instance


def test_model_actor(model_actor_instance):
    assert model_actor_instance


def test_model_actor_forward(model_actor_instance):
    actions = torch.rand(2)
    action = model_actor_instance.forward(actions)
    assert action.any()


def test_changing_parameters_model_actor(model_actor_instance, registered_optimizer):
    registered_optimizer.add_module_changing_check(model_actor_instance)


def test_unchanging_parameters_model_actor(model_actor_instance, registered_optimizer):
    registered_optimizer.add_module_unchanging_check(model_actor_instance.l1)


def test_changing_tensor_model_actor(model_actor_instance, registered_optimizer):
    registered_optimizer.add_tensor_changing_check(
        model_actor_instance.l1.weight, tensor_name="l1.weight"
    )


def test_unchanging_tensor_model_actor(model_actor_instance, registered_optimizer):
    registered_optimizer.add_tensor_unchanging_check(
        model_actor_instance.l1.bias, tensor_name="l1.bias"
    )


def test_output_module_model_actor(model_actor_instance, registered_optimizer):
    registered_optimizer.add_module_output_range_check(
        model_actor_instance, output_range=(-1, 1)
    )


def test_nan_module_model_actor(model_actor_instance, registered_optimizer):
    registered_optimizer.add_module_nan_check(model_actor_instance)


def test_nan_layer_model_actor(model_actor_instance, registered_optimizer):
    registered_optimizer.add_module_nan_check(model_actor_instance.l1)


def test_inf_module_model_actor(model_actor_instance, registered_optimizer):
    registered_optimizer.add_module_inf_check(model_actor_instance)


def test_inf_layer_model_actor(model_actor_instance, registered_optimizer):
    registered_optimizer.add_module_inf_check(model_actor_instance.l1)
