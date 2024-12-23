import copy
from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rnl.networks.custom_components import NoisyLinear


class EvolvableMLP(nn.Module):
    """The Evolvable Multi-layer Perceptron class"""

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        hidden_size: List[int],
        num_atoms: int,
        mlp_activation: str,
        mlp_output_activation: str,
        min_hidden_layers: int,
        max_hidden_layers: int,
        min_mlp_nodes: int,
        max_mlp_nodes: int,
        support: torch.Tensor,
        noise_std: float,
        device: str,
    ) -> None:
        super().__init__()

        assert (
            num_inputs > 0
        ), "'num_inputs' cannot be less than or equal to zero, please enter a valid integer."
        assert (
            num_outputs > 0
        ), "'num_outputs' cannot be less than or equal to zero, please enter a valid integer."
        for num in hidden_size:
            assert (
                num > 0
            ), "'hidden_size' cannot contain zero, please enter a valid integer."
        assert len(hidden_size) != 0, "MLP must contain at least one hidden layer."
        assert (
            min_hidden_layers < max_hidden_layers
        ), "'min_hidden_layers' must be less than 'max_hidden_layers."
        assert (
            min_mlp_nodes < max_mlp_nodes
        ), "'min_mlp_nodes' must be less than 'max_mlp_nodes."

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.mlp_activation = mlp_activation
        self.mlp_output_activation = mlp_output_activation
        self.min_hidden_layers = min_hidden_layers
        self.max_hidden_layers = max_hidden_layers
        self.min_mlp_nodes = min_mlp_nodes
        self.max_mlp_nodes = max_mlp_nodes
        self.hidden_size = hidden_size
        self.num_atoms = num_atoms
        self.support = support
        self.device = device
        self.noise_std = noise_std
        self._net_config = {
            "hidden_size": self.hidden_size,
            "mlp_activation": self.mlp_activation,
            "mlp_output_activation": self.mlp_output_activation,
            "min_hidden_layers": self.min_hidden_layers,
            "max_hidden_layers": self.max_hidden_layers,
            "min_mlp_nodes": self.min_mlp_nodes,
            "max_mlp_nodes": self.max_mlp_nodes,
        }

        self.feature_net, self.value_net, self.advantage_net = self.create_net()

    @property
    def net_config(self):
        return self._net_config

    def get_activation(self, activation_names):
        """Returns activation function for corresponding activation name.

        :param activation_names: Activation function name
        :type activation_names: str
        """
        activation_functions = {
            "Tanh": nn.Tanh,
            "Identity": nn.Identity,
            "ReLU": nn.ReLU,
            "ELU": nn.ELU,
            "Softsign": nn.Softsign,
            "Sigmoid": nn.Sigmoid,
            "Softplus": nn.Softplus,
            "Softmax": nn.Softmax,
            "LeakyReLU": nn.LeakyReLU,
            "PReLU": nn.PReLU,
            "GELU": nn.GELU,
            None: nn.Identity,
        }

        return (
            activation_functions[activation_names](dim=-1)
            if activation_names == "Softmax"
            else activation_functions[activation_names]()
        )

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        if hasattr(layer, "weight"):
            torch.nn.init.orthogonal_(layer.weight, std)
        elif hasattr(layer, "weight_mu") and hasattr(layer, "weight_sigma"):
            torch.nn.init.orthogonal_(layer.weight_mu, std)
            torch.nn.init.orthogonal_(layer.weight_sigma, std)

        if hasattr(layer, "bias"):
            torch.nn.init.constant_(layer.bias, bias_const)
        elif hasattr(layer, "bias_mu") and hasattr(layer, "bias_sigma"):
            torch.nn.init.constant_(layer.bias_mu, bias_const)
            torch.nn.init.constant_(layer.bias_sigma, bias_const)

        return layer

    def create_mlp(
        self,
        input_size,
        output_size,
        hidden_size,
        output_vanish,
        output_activation,
        noisy=False,
        rainbow_feature_net=False,
    ):
        """Creates and returns multi-layer perceptron."""
        net_dict = OrderedDict()
        if noisy:
            net_dict["linear_layer_0"] = NoisyLinear(
                input_size, hidden_size[0], self.noise_std
            )
        else:
            net_dict["linear_layer_0"] = nn.Linear(input_size, hidden_size[0])
        net_dict["linear_layer_0"] = self.layer_init(net_dict["linear_layer_0"])
        net_dict["layer_norm_0"] = nn.LayerNorm(hidden_size[0])
        net_dict["activation_0"] = self.get_activation(
            self.mlp_output_activation
            if (len(hidden_size) == 1 and rainbow_feature_net)
            else self.mlp_activation
        )
        if len(hidden_size) > 1:
            for l_no in range(1, len(hidden_size)):
                if noisy:
                    net_dict[f"linear_layer_{str(l_no)}"] = NoisyLinear(
                        hidden_size[l_no - 1], hidden_size[l_no], self.noise_std
                    )
                else:
                    net_dict[f"linear_layer_{str(l_no)}"] = nn.Linear(
                        hidden_size[l_no - 1], hidden_size[l_no]
                    )
                net_dict[f"linear_layer_{str(l_no)}"] = self.layer_init(
                    net_dict[f"linear_layer_{str(l_no)}"]
                )
                net_dict[f"layer_norm_{str(l_no)}"] = nn.LayerNorm(hidden_size[l_no])
                net_dict[f"activation_{str(l_no)}"] = self.get_activation(
                    self.mlp_activation
                    if not rainbow_feature_net
                    else self.mlp_output_activation
                )
        if not rainbow_feature_net:
            if noisy:
                output_layer = NoisyLinear(hidden_size[-1], output_size, self.noise_std)
            else:
                output_layer = nn.Linear(hidden_size[-1], output_size)
            output_layer = self.layer_init(output_layer)
            if output_vanish:
                output_layer.weight_mu.data.mul_(0.1)
                output_layer.bias_mu.data.mul_(0.1)
                output_layer.weight_sigma.data.mul_(0.1)
                output_layer.bias_sigma.data.mul_(0.1)

            net_dict["linear_layer_output"] = output_layer
            if output_activation is not None:
                net_dict["activation_output"] = self.get_activation(output_activation)
        net = nn.Sequential(net_dict)
        return net

    def create_net(self):
        """Creates and returns neural network."""

        feature_net = self.create_mlp(
            input_size=self.num_inputs,
            output_size=self.hidden_size[0],
            hidden_size=[self.hidden_size[0]],
            output_vanish=False,
            output_activation=self.mlp_activation,
            rainbow_feature_net=True,
        )
        value_net = self.create_mlp(
            input_size=self.hidden_size[0],
            output_size=self.num_atoms,
            hidden_size=self.hidden_size[1:],
            output_vanish=True,
            output_activation=None,
            noisy=True,
        )
        advantage_net = self.create_mlp(
            input_size=self.hidden_size[0],
            output_size=self.num_atoms * self.num_outputs,
            hidden_size=self.hidden_size[1:],
            output_vanish=True,
            output_activation=None,
            noisy=True,
        )
        value_net, advantage_net, feature_net = (
            value_net.to(self.device),
            advantage_net.to(self.device),
            feature_net.to(self.device),
        )

        return feature_net, value_net, advantage_net

    def reset_noise(self):
        """Resets noise of value and advantage networks."""
        for layer in self.value_net:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        for layer in self.advantage_net:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

    def forward(self, x, q=True, log=False):
        """Returns output of neural network.

        :param x: Neural network input
        :type x: torch.Tensor() or np.array
        :param q: Return Q value if using rainbow, defaults to True
        :type q: bool, optional
        :param log: Return log softmax instead of softmax, defaults to False
        :type log: bool, optional
        """
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(np.array(x))
            x = x.to(self.device)

        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = self.feature_net(x)

        value = self.value_net(x)
        advantage = self.advantage_net(x)
        value = value.view(-1, 1, self.num_atoms)
        advantage = advantage.view(-1, self.num_outputs, self.num_atoms)
        x = value + advantage - advantage.mean(1, keepdim=True)
        if log:
            x = F.log_softmax(x, dim=2)
            return x
        else:
            x = F.softmax(x, dim=2)

        # Output at this point is (batch_size, actions, num_support)
        if q:
            x = torch.sum(x * self.support.expand_as(x), dim=2)

        return x

    @property
    def init_dict(self):
        init_dict = {
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "hidden_size": self.hidden_size,
            "num_atoms": self.num_atoms,
            "mlp_activation": self.mlp_activation,
            "mlp_output_activation": self.mlp_output_activation,
            "min_hidden_layers": self.min_hidden_layers,
            "max_hidden_layers": self.max_hidden_layers,
            "min_mlp_nodes": self.min_mlp_nodes,
            "max_mlp_nodes": self.max_mlp_nodes,
            "support": self.support,
            "noise_std": self.noise_std,
            "device": self.device,
        }
        return init_dict

    def add_mlp_layer(self):
        """Adds a hidden layer to neural network."""
        # add layer to hyper params
        if len(self.hidden_size) < self.max_hidden_layers:  # HARD LIMIT
            self.hidden_size += [self.hidden_size[-1]]
            self.recreate_nets()
        else:
            self.add_mlp_node()

    def remove_mlp_layer(self):
        """Removes a hidden layer from neural network."""
        if len(self.hidden_size) > self.min_hidden_layers:  # HARD LIMIT
            self.hidden_size = self.hidden_size[:-1]
            self.recreate_nets(shrink_params=True)
        else:
            self.add_mlp_node()

    def add_mlp_node(self, hidden_layer=None, numb_new_nodes=None):
        """Adds nodes to hidden layer of neural network.

        :param hidden_layer: Depth of hidden layer to add nodes to, defaults to None
        :type hidden_layer: int, optional
        :param numb_new_nodes: Number of nodes to add to hidden layer, defaults to None
        :type numb_new_nodes: int, optional
        """
        if hidden_layer is None:
            hidden_layer = np.random.randint(0, len(self.hidden_size), 1)[0]
        else:
            hidden_layer = min(hidden_layer, len(self.hidden_size) - 1)
        if numb_new_nodes is None:
            numb_new_nodes = np.random.choice([16, 32, 64], 1)[0]

        if (
            self.hidden_size[hidden_layer] + numb_new_nodes <= self.max_mlp_nodes
        ):  # HARD LIMIT
            self.hidden_size[hidden_layer] += numb_new_nodes
            self.recreate_nets()

        return {"hidden_layer": hidden_layer, "numb_new_nodes": numb_new_nodes}

    def remove_mlp_node(self, hidden_layer=None, numb_new_nodes=None):
        """Removes nodes from hidden layer of neural network.

        :param hidden_layer: Depth of hidden layer to remove nodes from, defaults to None
        :type hidden_layer: int, optional
        :param numb_new_nodes: Number of nodes to remove from hidden layer, defaults to None
        :type numb_new_nodes: int, optional
        """
        if hidden_layer is None:
            hidden_layer = np.random.randint(0, len(self.hidden_size), 1)[0]
        else:
            hidden_layer = min(hidden_layer, len(self.hidden_size) - 1)
        if numb_new_nodes is None:
            numb_new_nodes = np.random.choice([16, 32, 64], 1)[0]

        if (
            self.hidden_size[hidden_layer] - numb_new_nodes > self.min_mlp_nodes
        ):  # HARD LIMIT
            self.hidden_size[hidden_layer] -= numb_new_nodes
            self.recreate_nets(shrink_params=True)

        return {"hidden_layer": hidden_layer, "numb_new_nodes": numb_new_nodes}

    def recreate_nets(self, shrink_params=False):
        """Recreates neural networks."""
        new_feature_net, new_value_net, new_advantage_net = self.create_net()
        new_feature_net = self.shrink_preserve_parameters(
            old_net=self.feature_net, new_net=new_feature_net
        )
        new_value_net = self.shrink_preserve_parameters(
            old_net=self.value_net, new_net=new_value_net
        )
        new_advantage_net = self.shrink_preserve_parameters(
            old_net=self.advantage_net, new_net=new_advantage_net
        )

        self.feature_net, self.value_net, self.advantage_net = (
            new_feature_net,
            new_value_net,
            new_advantage_net,
        )

    def clone(self):
        """Returns clone of neural net with identical parameters."""
        init_dict = {
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "hidden_size": self.hidden_size,
            "num_atoms": self.num_atoms,
            "mlp_activation": self.mlp_activation,
            "mlp_output_activation": self.mlp_output_activation,
            "min_hidden_layers": self.min_hidden_layers,
            "max_hidden_layers": self.max_hidden_layers,
            "min_mlp_nodes": self.min_mlp_nodes,
            "max_mlp_nodes": self.max_mlp_nodes,
            "support": self.support,
            "noise_std": self.noise_std,
            "device": self.device,
        }
        clone = EvolvableMLP(**copy.deepcopy(init_dict))
        clone.load_state_dict(self.state_dict())
        return clone

    def preserve_parameters(self, old_net, new_net):
        """Returns new neural network with copied parameters from old network.

        :param old_net: Old neural network
        :type old_net: nn.Module()
        :param new_net: New neural network
        :type new_net: nn.Module()
        """
        old_net_dict = dict(old_net.named_parameters())

        for key, param in new_net.named_parameters():
            if key in old_net_dict.keys():
                if old_net_dict[key].data.size() == param.data.size():
                    param.data = old_net_dict[key].data
                else:
                    if "norm" not in key:
                        old_size = old_net_dict[key].data.size()
                        new_size = param.data.size()
                        if len(param.data.size()) == 1:
                            param.data[: min(old_size[0], new_size[0])] = old_net_dict[
                                key
                            ].data[: min(old_size[0], new_size[0])]
                        else:
                            param.data[
                                : min(old_size[0], new_size[0]),
                                : min(old_size[1], new_size[1]),
                            ] = old_net_dict[key].data[
                                : min(old_size[0], new_size[0]),
                                : min(old_size[1], new_size[1]),
                            ]

        return new_net

    def shrink_preserve_parameters(self, old_net, new_net):
        """Returns shrunk new neural network with copied parameters from old network.

        :param old_net: Old neural network
        :type old_net: nn.Module()
        :param new_net: New neural network
        :type new_net: nn.Module()
        """
        old_net_dict = dict(old_net.named_parameters())

        for key, param in new_net.named_parameters():
            if key in old_net_dict.keys():
                if old_net_dict[key].data.size() == param.data.size():
                    param.data = old_net_dict[key].data
                else:
                    if "norm" not in key:
                        old_size = old_net_dict[key].data.size()
                        new_size = param.data.size()
                        min_0 = min(old_size[0], new_size[0])
                        if len(param.data.size()) == 1:
                            param.data[:min_0] = old_net_dict[key].data[:min_0]
                        else:
                            min_1 = min(old_size[1], new_size[1])
                            param.data[:min_0, :min_1] = old_net_dict[key].data[
                                :min_0, :min_1
                            ]
        return new_net
