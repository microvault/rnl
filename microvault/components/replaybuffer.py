import random
from collections import deque, namedtuple
from typing import Tuple

import numpy as np
import torch


class ReplayBuffer:

    # Memória de replay priorizada
    # Fonte: https://arxiv.org/abs/1511.05952
    def __init__(
        self,
        buffer_size: int = 1048576,
        batch_size: int = 32,
        gamma: float = 0.99,
        nstep: int = 10,
        state_dim: int = 14,
        action_dim: int = 2,
        device: str = "cpu",
    ):
        """
        A árvore é composta por uma árvore de soma que contém as pontuações de prioridade em sua folha e também uma matriz de dados.
        """

        self.batch_size = batch_size
        self.gamma = gamma
        self.n_step = nstep
        self.n_step_buffer = deque(maxlen=nstep)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )

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
        done: np.bool_ = np.bool_(False),
    ):
        assert isinstance(
            state[0], np.float32
        ), f"State is not of type (np.float32) in REPLAY BUFFER -> state type: {type(state)}."

        # assert isinstance(action[0], np.float32), "Action is not of type (np.float32) in REPLAY BUFFER -> action type: {}.".format(type(action))
        assert isinstance(
            reward, (np.float32, np.float64)
        ), f"Reward is not of type (np.float64 / np.float32) in REPLAY BUFFER -> reward: {type(reward)}."

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

        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        - First, to sample a minibatch of size k the range [0, priority_total] is divided into k ranges.
        - Then a value is uniformly sampled from each range.
        - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
        - Then, we calculate IS weights for each minibatch element.
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = (
            torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None]))
            .float()
            .to(self.device)
        )
        actions = (
            torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None]))
            .float()
            .to(self.device)
        )
        rewards = (
            torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None]))
            .float()
            .to(self.device)
        )
        next_states = (
            torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None]))
            .float()
            .to(self.device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)
            )
            .int()
            .to(self.device)
        )

        assert (
            states.dtype == torch.float32
        ), "The (state) tensor elements are not of type torch.float32 in the REPLAY BUFFER -> {}.".format(
            states.dtype
        )
        assert (
            actions.dtype == torch.float32
        ), "The (actions) tensor elements are not of type torch.float32 in the REPLAY BUFFER -> {}.".format(
            actions.dtype
        )
        assert (
            rewards.dtype == torch.float32
        ), "The (rewards) tensor elements are not of type torch.float32 in the REPLAY BUFFER -> {}.".format(
            rewards.dtype
        )
        assert (
            next_states.dtype == torch.float32
        ), "The (next_states) tensor elements are not of type torch.float32 in the REPLAY BUFFER -> {}.".format(
            next_states.dtype
        )
        assert (
            dones.dtype == torch.int
        ), "The (dones) tensor elements are not of type torch.float32 in the REPLAY BUFFER -> {}.".format(
            dones.dtype
        )

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return len(self.memory)
