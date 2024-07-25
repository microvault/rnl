import pytest
import torch
from torch import optim

from microvault.models.model import ModelCritic
from microvault.training.sanity import Sanity


@pytest.fixture
def model_critic_instance():
    return ModelCritic(
        state_dim=13,
        action_dim=2,
        l1=400,
        l2=300,
        device="cpu",
        batch_size=128,
    )


@pytest.fixture
def sanity_instance():
    return Sanity()


@pytest.fixture
def optimizer_instance(model_critic_instance):
    return optim.Adam(model_critic_instance.parameters(), lr=0.001)


@pytest.fixture
def registered_optimizer(sanity_instance, optimizer_instance):
    sanity_instance.register(optimizer_instance)
    return sanity_instance


def test_model_critic(model_critic_instance):
    assert model_critic_instance


def test_model_critic_forward(model_critic_instance):
    batch_size = 128
    actions = torch.rand(batch_size, 2)
    states = torch.rand(batch_size, 13)
    q1, q2 = model_critic_instance.forward(states, actions)
    assert q1.any() and q2.any()


def test_changing_parameters_model_critic(model_critic_instance, registered_optimizer):
    registered_optimizer.add_module_changing_check(model_critic_instance)


def test_unchanging_parameters_model_critic(
    model_critic_instance, registered_optimizer
):
    registered_optimizer.add_module_unchanging_check(model_critic_instance.l1)


def test_changing_tensor_model_critic(model_critic_instance, registered_optimizer):
    registered_optimizer.add_tensor_changing_check(
        model_critic_instance.l1.weight, tensor_name="l1.weight"
    )


def test_unchanging_tensor_model_critic(model_critic_instance, registered_optimizer):
    registered_optimizer.add_tensor_unchanging_check(
        model_critic_instance.l1.bias, tensor_name="l1.bias"
    )


def test_output_module_model_critic(model_critic_instance, registered_optimizer):
    registered_optimizer.add_module_output_range_check(
        model_critic_instance, output_range=(-1, 1)
    )


def test_nan_module_model_critic(model_critic_instance, registered_optimizer):
    registered_optimizer.add_module_nan_check(model_critic_instance)


def test_nan_layer_model_critic(model_critic_instance, registered_optimizer):
    registered_optimizer.add_module_nan_check(model_critic_instance.l1)


def test_inf_module_model_critic(model_critic_instance, registered_optimizer):
    registered_optimizer.add_module_inf_check(model_critic_instance)


def test_inf_layer_model_critic(model_critic_instance, registered_optimizer):
    registered_optimizer.add_module_inf_check(model_critic_instance.l1)
