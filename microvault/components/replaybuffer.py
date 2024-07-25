from collections import deque

import numpy as np
import torch

from microvault.components.sumtree import SumTree


class ReplayBuffer:

    # Memória de replay priorizada
    # Fonte: https://arxiv.org/abs/1511.05952
    def __init__(
        self,
        buffer_size: int = 1048576,
        batch_size: int = 32,
        gamma: float = 0.99,
        nstep: float = 10,
        state_dim: int = 13,
        action_dim: int = 2,
        epsilon: float = 0.01,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment_per_sampling: float = 1e-4,
        absolute_error_upper: float = 1.0,
        device: str = "cpu",
    ):
        """
        A árvore é composta por uma árvore de soma que contém as pontuações de prioridade em sua folha e também uma matriz de dados.
        """

        self.tree = SumTree(buffer_size)
        self.epsilon = epsilon  # small amount to avoid zero priority
        self.alpha = (
            alpha  # [0..1] convert the importance of TD error to priority, often 0.6
        )
        self.beta = (
            beta  # importance-sampling, from initial value increasing to 1, often 0.4
        )
        self.beta_increment_per_sampling = (
            beta_increment_per_sampling  # annealing the bias, often 1e-3
        )
        self.absolute_error_upper = absolute_error_upper  # clipped abs error
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_step = nstep
        self.n_step_buffer = deque(maxlen=nstep)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

    def __len__(self):
        return len(self.tree)

    def is_full(self):
        return len(self.tree) >= self.tree.capacity

    def multistep(self):
        # Retorno do buffer de n-step
        ret = 0
        for idx in range(1):
            ret += (self.gamma**idx) * self.n_step_buffer[idx][2]

        return (
            self.n_step_buffer[0][0],
            self.n_step_buffer[0][1],
            ret,
            self.n_step_buffer[-1][3],
            self.n_step_buffer[-1][4],
        )

    def add(
        self,
        state: np.ndarray = None,
        action: np.ndarray = None,
        reward: np.float64 = None,
        next_state: np.float64 = None,
        done: bool = False,
    ):
        assert isinstance(
            state[0], np.float32
        ), f"State is not of type (np.float32) in REPLAY BUFFER -> state type: {type(state)}."

        # assert isinstance(action[0], np.float32), "Action is not of type (np.float32) in REPLAY BUFFER -> action type: {}.".format(type(action))
        assert isinstance(
            reward, (int, np.float64)
        ), f"Reward is not of type (np.float64 / int) in REPLAY BUFFER -> reward: {type(reward)}."

        assert isinstance(
            next_state[0], np.float32
        ), f"Next State is not of type (np.float32) in REPLAY BUFFER -> next state type: {type(next_state)}."

        assert isinstance(
            done, np.bool_
        ), f"Done is not of type (np.bool_) in REPLAY BUFFER -> done type: {type(done)}."

        assert (
            state.shape[0] == self.state_dim
        ), f"The size of the state is not {self.state_dim} in REPLAY BUFFER -> state size: {next_state.shape[0]}."

        assert (
            action.shape[0] == self.action_dim
        ), f"The size of the action is not {self.action_dim} in REPLAY BUFFER -> action size: {action.shape[0]}."

        if isinstance(reward, np.float64):
            assert (
                reward.size == 1
            ), f"The size of the reward is not (1) in REPLAY BUFFER -> reward size: {reward.size}."

        assert (
            next_state.shape[0] == self.state_dim
        ), f"The size of the next_state is not {self.state_dim} in REPLAY BUFFER -> next_state size: {next_state.shape[0]}."

        assert (
            state.ndim == 1
        ), f"The ndim of the state is not (1) in REPLAY BUFFER -> state ndim: {state.ndim}."

        assert (
            action.ndim == 1
        ), f"The ndim of the action is not (1) in REPLAY BUFFER -> action ndim: {action.ndim}."

        if isinstance(reward, np.float64):
            assert (
                reward.ndim == 0
            ), f"The ndim of the reward is not (0) in REPLAY BUFFER -> reward ndim: {reward.ndim}."

        assert (
            next_state.ndim == 1
        ), f"The ndim of the next_state is not (1) in REPLAY BUFFER -> next_state ndim: {next_state.ndim}."

        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_step:
            (state, action, reward, next_state, done) = self.multistep()

        sample = (state, action, reward, next_state, done)
        if reward is None:
            priority = np.amax(self.tree.tree[-self.tree.capacity :])
            if priority == 0:
                priority = self.absolute_error_upper
        else:
            priority = min(
                (abs(reward) + self.epsilon) ** self.alpha, self.absolute_error_upper
            )
        self.tree.add(sample, priority)

    def sample(self):
        """
        - First, to sample a minibatch of size k the range [0, priority_total] is divided into k ranges.
        - Then a value is uniformly sampled from each range.
        - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
        - Then, we calculate IS weights for each minibatch element.
        """

        minibatch = []

        idxs = np.empty((self.batch_size,), dtype=np.int32)
        is_weights = np.empty((self.batch_size,), dtype=np.float32)

        # Calculate the priority segment
        # Divide the Range[0, ptotal] into n ranges
        priority_segment = (
            self.tree.total_priority / self.batch_size
        )  # priority segment

        # Increase the beta each time we sample a new minibatch
        self.beta = np.amin(
            [1.0, self.beta + self.beta_increment_per_sampling]
        )  # max = 1

        # Calculate the max_weight
        p_min = (
            np.amin(self.tree.tree[-self.tree.capacity :]) / self.tree.total_priority
        )
        max_weight = (p_min * self.batch_size) ** (-self.beta)
        max_weight = max(max_weight, 1e-5)  # Evita divisão por zero

        priority_segment = self.tree.total_priority / self.batch_size

        for i in range(self.batch_size):
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            index, priority, data = self.tree.get_leaf(value)

            priority = max(priority, 1e-5)  # Evita zero
            sampling_probabilities = priority / self.tree.total_priority
            sampling_probabilities = max(sampling_probabilities, 1e-5)  # Evita zero

            is_weights[i] = (
                np.power(self.batch_size * sampling_probabilities, -self.beta)
                / max_weight
            )
            is_weights[i] = max(is_weights[i], 1e-5)  # Evita valores inválidos

            idxs[i] = index
            minibatch.append(data)

        states = (
            torch.from_numpy(np.vstack([e[0] for e in minibatch if e is not None]))
            .float()
            .to(self.device)
        )
        actions = (
            torch.from_numpy(np.vstack([e[1] for e in minibatch if e is not None]))
            .float()
            .to(self.device)
        )
        rewards = (
            torch.from_numpy(np.vstack([e[2] for e in minibatch if e is not None]))
            .float()
            .to(self.device)
        )
        next_states = (
            torch.from_numpy(np.vstack([e[3] for e in minibatch if e is not None]))
            .float()
            .to(self.device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e[4] for e in minibatch if e is not None]).astype(np.uint8)
            )
            .float()
            .to(self.device)
        )

        # assert isinstance(
        #     states, torch.Tensor
        # ), "State is not of type torch.Tensor in REPLAY BUFFER."
        # assert isinstance(
        #     actions, torch.Tensor
        # ), "Actions is not of type torch.Tensor in REPLAY BUFFER."
        # assert isinstance(
        #     rewards, torch.Tensor
        # ), "Rewards is not of type torch.Tensor in REPLAY BUFFER."
        # assert isinstance(
        #     next_states, torch.Tensor
        # ), "Next states is not of type torch.Tensor in REPLAY BUFFER."
        # assert isinstance(
        #     dones, torch.Tensor
        # ), "Dones is not of type torch.Tensor in REPLAY BUFFER."

        # assert (
        #     states.dtype == torch.float32
        # ), "The (state) tensor elements are not of type torch.float32 in the REPLAY BUFFER -> {}.".format(
        #     states.dtype
        # )
        # assert (
        #     actions.dtype == torch.float32
        # ), "The (actions) tensor elements are not of type torch.float32 in the REPLAY BUFFER -> {}.".format(
        #     actions.dtype
        # )
        # assert (
        #     rewards.dtype == torch.float32
        # ), "The (rewards) tensor elements are not of type torch.float32 in the REPLAY BUFFER -> {}.".format(
        #     rewards.dtype
        # )
        # assert (
        #     next_states.dtype == torch.float32
        # ), "The (next_states) tensor elements are not of type torch.float32 in the REPLAY BUFFER -> {}.".format(
        #     next_states.dtype
        # )
        # assert (
        #     dones.dtype == torch.int
        # ), "The (dones) tensor elements are not of type torch.float32 in the REPLAY BUFFER -> {}.".format(
        #     dones.dtype
        # )

        return idxs, states, actions, rewards, next_states, dones, is_weights

    def batch_update(self, idxs, errors) -> None:
        """
        Update the priorities on the tree
        """
        errors = errors + self.epsilon
        clipped_errors = np.minimum(errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.alpha)

        for idx, p in zip(idxs, ps):
            self.tree.update(idx, p)
