import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rnl.networks.custom_components import NoisyLinear


class MakeEvolvable(nn.Module):
    """Wrapper to make a neural network evolvable

    :param network: Input neural network
    :type network: nn.Module
    :param input_tensor: Example input tensor so forward pass can be made to detect the network architecture
    :type input_tensor: torch.Tensor
    :param num_atoms: Number of atoms for Rainbow DQN, defaults to 51
    :type num_atoms: int, optional
    :param secondary_input_tensor: Second input tensor if network performs forward pass with two tensors, for example, \
        off-policy algorithms that use a critic(s) with environments that have RGB image observations and thus require CNN \
        architecture, defaults to None
    :type secondary_input_tensor: torch.Tensor, optional
    :param min_hidden_layers: Minimum number of hidden layers the fully connected layer will shrink down to, defaults to 1
    :type min_hidden_layers: int, optional
    :param max_hidden_layers: Maximum number of hidden layers the fully connected layer will expand to, defaults to 3
    :type max_hidden_layers: int, optional
    :param min_mlp_nodes: Minimum number of nodes a layer can have within the fully connected layer, defaults to 64
    :type min_mlp_nodes: int, optional
    :param max_mlp_nodes: Maximum number of nodes a layer can have within the fully connected layer, defaults to 1024
    :type max_mlp_nodes: int, optional
    :param output_vanish: Vanish output by multiplying by 0.1, defaults to False
    :type output_vanish: bool, optional
    :param init_layers: Initialise network layers, defaults to False
    :type init_layers: bool, optional
    :param support: Atoms support tensor, defaults to None
    :type support: torch.Tensor(), optional
    :param rainbow: Using Rainbow DQN, defaults to False
    :type rainbow: bool, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    """

    def __init__(
        self,
        network,
        input_tensor,
        secondary_input_tensor=None,
        num_atoms=51,
        min_hidden_layers=1,
        max_hidden_layers=3,
        min_mlp_nodes=64,
        max_mlp_nodes=1024,
        output_vanish=False,
        init_layers=False,
        support=None,
        rainbow=True,
        device="cpu",
        **kwargs,
    ):
        super().__init__()
        assert (
            min_hidden_layers < max_hidden_layers
        ), "'min_hidden_layers' must be less than 'max_hidden_layers."
        assert (
            min_mlp_nodes < max_mlp_nodes
        ), "'min_mlp_nodes' must be less than 'max_mlp_nodes."

        if not kwargs:
            assert isinstance(
                network, nn.Module
            ), f"'network' must be of type 'nn.Module'.{type(network)}"

        self.init_layers = init_layers
        self.min_hidden_layers = min_hidden_layers
        self.max_hidden_layers = max_hidden_layers
        self.min_mlp_nodes = min_mlp_nodes
        self.max_mlp_nodes = max_mlp_nodes
        self.output_vanish = output_vanish
        self.device = device

        #### Rainbow attributes
        self.rainbow = rainbow  #### add in as a doc string
        self.num_atoms = num_atoms
        self.support = support

        # Set the layer counters
        self.lin_counter = -1
        self.extra_critic_dims = (
            secondary_input_tensor.shape[-1]
            if secondary_input_tensor is not None
            else None
        )

        # Set placeholder attributes (needed for init_dict function to work)
        self.input_tensor = input_tensor.to(self.device)
        self.secondary_input_tensor = (
            secondary_input_tensor.to(self.device)
            if secondary_input_tensor is not None
            else secondary_input_tensor
        )

        # If first instance, network used to instantiate, upon cloning, init_dict used instead
        if not kwargs:
            self.detect_architecture(
                network.to(self.device), self.input_tensor, self.secondary_input_tensor
            )
        else:
            for key, value in kwargs.items():
                setattr(self, key, value)

        self.feature_net, self.value_net, self.advantage_net = self.create_nets()

    def forward(self, x, xc=None, q=True):
        """Returns output of neural network.

        :param x: Neural network input
        :type x: torch.Tensor() or np.array
        :param xc: Actions to be evaluated by critic, defaults to None
        :type xc: torch.Tensor() or np.array, optional
        :param q: Return Q value if using rainbow, defaults to True
        :type q: bool, optional
        """
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(np.array(x))

        x = x.to(self.device)

        if x.dtype != torch.float32:
            x = x.type(torch.float32)

        batch_size = x.size(0)

        x = self.feature_net(x)

        advantage = self.advantage_net(x)
        value = self.value_net(x)

        value = value.view(batch_size, 1, self.num_atoms)
        advantage = advantage.view(batch_size, self.num_outputs, self.num_atoms)

        x = value + advantage - advantage.mean(1, keepdim=True)
        x = F.softmax(x.view(-1, self.num_atoms), dim=-1).view(
            -1, self.num_outputs, self.num_atoms
        )
        x = x.clamp(min=1e-3)

        if q:
            x = torch.sum(x * self.support, dim=2)

        return x

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        """Initialize the weights of a neural network layer using orthogonal initialization and set the biases to a constant value.

        :param layer: Neural network layer
        :type layer: nn.Module
        :param std: Standard deviation, defaults to sqrt(2)
        :type std: float
        :param bias_const: Bias value, defaults to 0.0
        :type bias_const: float
        """
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_activation(self, activation_names):
        """Returns activation function for corresponding activation name.

        :param activation_names: Activation function name
        :type activation_names: str
        """
        activation_functions = {
            "Tanh": nn.Tanh,
            "Linear": nn.Identity,
            "ReLU": nn.ReLU,
            "ELU": nn.ELU,
            "Softsign": nn.Softsign,
            "Sigmoid": nn.Sigmoid,
            "Softplus": nn.Softplus,
            "Softmax": nn.Softmax,
            "LeakyReLU": nn.LeakyReLU,
            "PReLU": nn.PReLU,
            "GELU": nn.GELU,
        }

        return (
            activation_functions[activation_names](dim=-1)
            if activation_names == "Softmax"
            else activation_functions[activation_names]()
        )

    def get_normalization(self, normalization_name, layer_size):
        """Returns normalization layer for corresponding normalization name.

        :param normalization_names: Normalization layer name
        :type normalization_names: str
        :param layer_size: The layer after which the normalization layer will be applied
        :param layer_size: int
        """
        normalization_functions = {
            "BatchNorm2d": nn.BatchNorm2d,
            "BatchNorm3d": nn.BatchNorm3d,
            "InstanceNorm2d": nn.InstanceNorm2d,
            "InstanceNorm3d": nn.InstanceNorm3d,
            "LayerNorm": nn.LayerNorm,
        }

        return normalization_functions[normalization_name](layer_size)

    def detect_architecture(self, network, input_tensor, secondary_input_tensor=None):
        """Detect the architecture of a neural network.

        :param network: Neural network whose architecture is being detected
        :type network: nn.Module
        :param input_tensor: Tensor used to perform forward pass to detect layers
        :type input_tensor: torch.Tensor
        :param secondary_input_tensor: Second tensor used to perform forward pass if forward
        method of neural network takes two tensors as arguments, defaults to None
        :type secondary_input_tensor: torch.Tensor, optional
        """
        in_features_list = []
        out_features_list = []

        mlp_layer_info = dict()

        def register_hooks(module):
            def forward_hook(module, input, output):
                # Convolutional layer detection

                # Linear layer detection
                if isinstance(module, nn.Linear):
                    self.lin_counter += 1
                    in_features_list.append(module.in_features)
                    out_features_list.append(module.out_features)

                # Normalization layer detection
                elif isinstance(
                    module,
                    (
                        nn.BatchNorm2d,
                        nn.BatchNorm3d,
                        nn.InstanceNorm2d,
                        nn.InstanceNorm3d,
                        nn.LayerNorm,
                    ),
                ):
                    if "norm_layers" not in mlp_layer_info.keys():
                        mlp_layer_info["norm_layers"] = dict()
                    mlp_layer_info["norm_layers"][self.lin_counter] = str(
                        module.__class__.__name__
                    )

                # Detect activation layer (supported currently by AgileRL)
                elif isinstance(
                    module,
                    (
                        nn.Tanh,
                        nn.Identity,
                        nn.ReLU,
                        nn.ELU,
                        nn.Softsign,
                        nn.Sigmoid,
                        nn.Softplus,
                        nn.Softmax,
                        nn.LeakyReLU,
                        nn.PReLU,
                        nn.GELU,
                    ),
                ):
                    if "activation_layers" not in mlp_layer_info.keys():
                        mlp_layer_info["activation_layers"] = dict()
                    mlp_layer_info["activation_layers"][self.lin_counter] = str(
                        module.__class__.__name__
                    )

                else:
                    raise Exception(
                        f"{module} not currently supported, use an alternative layer."
                    )

            if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not isinstance(module, type(network))
            ):
                hooks.append(module.register_forward_hook(forward_hook))

        hooks = []
        network.apply(register_hooks)

        # Forward pass to collect network data necessary to make network evolvable
        with torch.no_grad():
            if secondary_input_tensor is None:
                network(input_tensor)
            else:
                network(input_tensor, secondary_input_tensor)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Save neural network information as attributes
        self.num_inputs, *self.hidden_size = in_features_list
        *_, self.num_outputs = out_features_list
        if len(self.hidden_size) == 0:
            raise TypeError("Network must have at least one hidden layer.")
        self.mlp_layer_info = mlp_layer_info

        if len(out_features_list) - 1 in mlp_layer_info["activation_layers"].keys():
            self.mlp_output_activation = mlp_layer_info["activation_layers"][
                len(out_features_list) - 1
            ]
        else:
            self.mlp_output_activation = None
        activation_function_set = set(mlp_layer_info["activation_layers"].values())
        if self.mlp_output_activation is not None:
            activation_function_set.remove(self.mlp_output_activation)
        if len(activation_function_set) > 1:
            raise TypeError(
                "All activation functions other than the output layer activation must be the same."
            )
        else:
            self.mlp_activation = list(mlp_layer_info["activation_layers"].values())[0]

        self.arch = "mlp"

        # Reset the layer counters
        self.lin_counter = -1

    def create_mlp(
        self,
        input_size,
        output_size,
        hidden_size,
        name,
        mlp_activation,
        mlp_output_activation,
        noisy=True,
        rainbow_feature_net=True,
    ):
        """Creates and returns multi-layer perceptron.

        :param input_size: Input dimensions to first MLP layer
        :type input_size: int
        :param output_size: Output dimensions from last MLP layer
        :type output_size: int
        :param hidden_size: Hidden layer sizes
        :type hidden_size: list[int]
        :param name: Layer name
        :type name: str
        ####
        :param noisy:
        :type noisy:
        :param rainbow_feature_net:
        :type rainbow_feature_net:
        """

        net_dict = OrderedDict()
        if noisy:
            net_dict[f"{name}_linear_layer_0"] = NoisyLinear(input_size, hidden_size[0])
        else:
            net_dict[f"{name}_linear_layer_0"] = nn.Linear(input_size, hidden_size[0])

        if self.init_layers:
            net_dict[f"{name}_linear_layer_0"] = self.layer_init(
                net_dict[f"{name}_linear_layer_0"]
            )

        if ("norm_layers" in self.mlp_layer_info.keys()) and (
            0 in self.mlp_layer_info["norm_layers"].keys()
        ):
            net_dict[f"{name}_layer_norm_0"] = self.get_normalization(
                self.mlp_layer_info["norm_layers"][0], hidden_size[0]
            )

        if ("activation_layers" in self.mlp_layer_info.keys()) and (
            0 in self.mlp_layer_info["activation_layers"].keys()
        ):
            net_dict[f"{name}_activation_0"] = self.get_activation(
                mlp_output_activation
                if (len(hidden_size) == 1 and rainbow_feature_net)
                else mlp_activation
            )

        if len(hidden_size) > 1:
            for l_no in range(1, len(hidden_size)):
                if noisy:
                    net_dict[f"{name}_linear_layer_{str(l_no)}"] = NoisyLinear(
                        hidden_size[l_no - 1], hidden_size[l_no]
                    )
                else:
                    net_dict[f"{name}_linear_layer_{str(l_no)}"] = nn.Linear(
                        hidden_size[l_no - 1], hidden_size[l_no]
                    )
                if self.init_layers:
                    net_dict[f"{name}_linear_layer_{str(l_no)}"] = self.layer_init(
                        net_dict[f"{name}_linear_layer_{str(l_no)}"]
                    )
                if ("norm_layers" in self.mlp_layer_info.keys()) and (
                    l_no in self.mlp_layer_info["norm_layers"].keys()
                ):
                    net_dict[f"{name}_layer_norm_{str(l_no)}"] = self.get_normalization(
                        self.mlp_layer_info["norm_layers"][l_no], hidden_size[l_no]
                    )
                if l_no in self.mlp_layer_info["activation_layers"].keys():
                    net_dict[f"{name}_activation_{str(l_no)}"] = self.get_activation(
                        mlp_activation
                        if not rainbow_feature_net
                        else mlp_output_activation
                    )
        if not rainbow_feature_net:
            if noisy:
                output_layer = NoisyLinear(hidden_size[-1], output_size)
            else:
                output_layer = nn.Linear(hidden_size[-1], output_size)
            if self.init_layers:
                output_layer = self.layer_init(output_layer)

            if self.output_vanish:
                output_layer.weight.data.mul_(0.1)
                output_layer.bias.data.mul_(0.1)

            net_dict[f"{name}_linear_layer_output"] = output_layer
            if mlp_output_activation is not None:
                net_dict[f"{name}_activation_output"] = self.get_activation(
                    mlp_output_activation
                )

        return nn.Sequential(net_dict)

    def create_nets(self):
        """Creates and returns the feature and value net."""

        feature_net = self.create_mlp(
            input_size=self.num_inputs,
            output_size=128,
            hidden_size=[128],
            name="feature",
            rainbow_feature_net=True,
            mlp_activation=self.mlp_activation,
            mlp_output_activation="ReLU",
        )
        value_net = self.create_mlp(
            input_size=128,
            output_size=self.num_atoms,
            hidden_size=self.hidden_size,
            noisy=True,
            name="value",
            mlp_output_activation=self.mlp_output_activation,
            mlp_activation=self.mlp_activation,
        )
        advantage_net = self.create_mlp(
            input_size=128,
            output_size=self.num_atoms * self.num_outputs,
            hidden_size=self.hidden_size,
            noisy=True,
            name="advantage",
            mlp_output_activation=self.mlp_output_activation,
            mlp_activation=self.mlp_activation,
        )

        feature_net = feature_net.to(self.device)
        value_net = value_net.to(self.device) if value_net is not None else value_net
        advantage_net = (
            advantage_net.to(self.device)
            if advantage_net is not None
            else advantage_net
        )
        return feature_net, value_net, advantage_net

    @property
    def init_dict(self):
        """Returns model information in dictionary."""
        init_dict = {
            "network": None,
            "input_tensor": self.input_tensor,
            "secondary_input_tensor": self.secondary_input_tensor,
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "hidden_size": self.hidden_size,
            "mlp_activation": self.mlp_activation,
            "mlp_output_activation": self.mlp_output_activation,
            "device": self.device,
            "extra_critic_dims": self.extra_critic_dims,
            "output_vanish": self.output_vanish,
            "init_layers": self.init_layers,
            "arch": self.arch,
            "mlp_layer_info": self.mlp_layer_info,
            "num_atoms": self.num_atoms,
            "rainbow": self.rainbow,
            "support": self.support,
        }

        return init_dict

    def add_mlp_layer(self):
        """Adds a hidden layer to value network."""
        if len(self.hidden_size) < self.max_hidden_layers:  # HARD LIMIT
            self.hidden_size += [self.hidden_size[-1]]
            self.mlp_layer_info["activation_layers"][
                len(self.hidden_size) - 1
            ] = self.mlp_activation
            if self.mlp_output_activation is not None:
                self.mlp_layer_info["activation_layers"][
                    len(self.hidden_size)
                ] = self.mlp_output_activation
            self.recreate_nets()
        else:
            self.add_mlp_node()

    def remove_mlp_layer(self):
        """Removes a hidden layer from value network."""
        if len(self.hidden_size) > self.min_hidden_layers:  # HARD LIMIT
            self.hidden_size = self.hidden_size[:-1]
            if len(self.hidden_size) in self.mlp_layer_info["activation_layers"].keys():
                if self.mlp_output_activation is None:
                    self.mlp_layer_info["activation_layers"].pop(len(self.hidden_size))
                else:
                    self.mlp_layer_info["activation_layers"].pop(
                        len(self.hidden_size) + 1
                    )
                    self.mlp_layer_info["activation_layers"][
                        len(self.hidden_size)
                    ] = self.mlp_output_activation
            else:
                if self.mlp_output_activation is not None:
                    self.mlp_layer_info["activation_layers"].pop(
                        len(self.hidden_size) + 1
                    )
                    self.mlp_layer_info["activation_layers"][
                        len(self.hidden_size)
                    ] = self.mlp_output_activation

            if (
                "norm_layers" in self.mlp_layer_info.keys()
                and len(self.hidden_size) in self.mlp_layer_info["norm_layers"]
            ):
                self.mlp_layer_info["norm_layers"].pop(len(self.hidden_size))

            self.recreate_nets(shrink_params=True)
        else:
            self.add_mlp_node()

    def add_mlp_node(self, hidden_layer=None, numb_new_nodes=None):
        """Adds nodes to hidden layer of value network.

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

    def reset_noise(self):
        """Resets noise of value and advantage networks."""
        for layer in self.value_net:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        for layer in self.advantage_net:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

    def recreate_nets(self, shrink_params=False):
        """Recreates neural networks.

        :param shrink_params: Boolean flag to shrink parameters
        :type shrink_params: bool
        """
        new_feature_net, new_value_net, new_advantage_net = self.create_nets()

        if shrink_params:
            if self.value_net is not None:
                new_value_net = self.shrink_preserve_parameters(
                    old_net=self.value_net, new_net=new_value_net
                )
            if self.advantage_net is not None:
                new_advantage_net = self.shrink_preserve_parameters(
                    old_net=self.advantage_net, new_net=new_advantage_net
                )
            new_feature_net = self.shrink_preserve_parameters(
                old_net=self.feature_net, new_net=new_feature_net
            )
        else:
            if self.value_net is not None:
                new_value_net = self.preserve_parameters(
                    old_net=self.value_net, new_net=new_value_net
                )
            if self.advantage_net is not None:
                new_advantage_net = self.preserve_parameters(
                    old_net=self.advantage_net, new_net=new_advantage_net
                )
            new_feature_net = self.preserve_parameters(
                old_net=self.feature_net, new_net=new_feature_net
            )

        self.feature_net, self.value_net, self.advantage_net = (
            new_feature_net,
            new_value_net,
            new_advantage_net,
        )

    def clone(self):
        """Returns clone of neural net with identical parameters."""
        clone = MakeEvolvable(**copy.deepcopy(self.init_dict))
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
                        elif len(param.data.size()) == 2:
                            param.data[
                                : min(old_size[0], new_size[0]),
                                : min(old_size[1], new_size[1]),
                            ] = old_net_dict[key].data[
                                : min(old_size[0], new_size[0]),
                                : min(old_size[1], new_size[1]),
                            ]
                        elif len(param.data.size()) == 3:
                            param.data[
                                : min(old_size[0], new_size[0]),
                                : min(old_size[1], new_size[1]),
                                : min(old_size[2], new_size[2]),
                            ] = old_net_dict[key].data[
                                : min(old_size[0], new_size[0]),
                                : min(old_size[1], new_size[1]),
                                : min(old_size[2], new_size[2]),
                            ]
                        elif len(param.data.size()) == 4:
                            param.data[
                                : min(old_size[0], new_size[0]),
                                : min(old_size[1], new_size[1]),
                                : min(old_size[2], new_size[2]),
                                : min(old_size[3], new_size[3]),
                            ] = old_net_dict[key].data[
                                : min(old_size[0], new_size[0]),
                                : min(old_size[1], new_size[1]),
                                : min(old_size[2], new_size[2]),
                                : min(old_size[3], new_size[3]),
                            ]
                        elif len(param.data.size()) == 5:
                            param.data[
                                : min(old_size[0], new_size[0]),
                                : min(old_size[1], new_size[1]),
                                : min(old_size[2], new_size[2]),
                                : min(old_size[3], new_size[3]),
                                : min(old_size[4], new_size[4]),
                            ] = old_net_dict[key].data[
                                : min(old_size[0], new_size[0]),
                                : min(old_size[1], new_size[1]),
                                : min(old_size[2], new_size[2]),
                                : min(old_size[3], new_size[3]),
                                : min(old_size[4], new_size[4]),
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
