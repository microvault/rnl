import random
from collections import deque, namedtuple
from typing import Tuple

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:

    def __init__(self, buffer_size, batch_size):

        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )

    def add(self, state, action, reward, next_state, done) -> None:
        """Add experiences to the buffer

        Params
        ======
            state (np.ndarray): agent states
            action (np.ndarray): agent action
            reward (np.float64): agent reward
            next_state (np.ndarray): agent next_state
        """

        assert isinstance(
            state, np.ndarray
        ), "State is not of data structure (np.ndarray) in REPLAY BUFFER -> state: {}.".format(
            type(state)
        )
        assert isinstance(
            action, np.ndarray
        ), "Action is not of data structure (np.ndarray) in REPLAY BUFFER -> action: {}.".format(
            type(action)
        )
        assert isinstance(
            next_state, np.ndarray
        ), "Next State is not of data structure (np.ndarray) in REPLAY BUFFER -> next state: {}.".format(
            type(next_state)
        )

        assert isinstance(
            state[0], np.float32
        ), "State is not of type (np.float32) in REPLAY BUFFER -> state type: {}.".format(
            type(state)
        )
        assert isinstance(
            action[0], np.float32
        ), "Action is not of type (np.float32) in REPLAY BUFFER -> action type: {}.".format(
            type(action)
        )
        assert isinstance(
            reward, (int, np.float64)
        ), "Reward is not of type (np.float64 / int) in REPLAY BUFFER -> reward: {}.".format(
            type(reward)
        )
        assert isinstance(
            next_state[0], np.float32
        ), "Next State is not of type (np.float32) in REPLAY BUFFER -> next state type: {}.".format(
            type(next_state)
        )
        assert isinstance(
            done, bool
        ), "Done is not of type (bool) in REPLAY BUFFER -> done type: {}.".format(
            type(done)
        )

        assert (
            state.shape[0] == 24
        ), "The size of the state is not (24) in REPLAY BUFFER -> state size: {}.".format(
            state.shape[0]
        )
        assert (
            action.shape[0] == 4
        ), "The size of the action is not (4) in REPLAY BUFFER -> action size: {}.".format(
            state.shape[0]
        )
        if isinstance(reward, np.float64):
            assert (
                reward.size == 1
            ), "The size of the reward is not (1) in REPLAY BUFFER -> reward size: {}.".format(
                reward.size
            )
        assert (
            next_state.shape[0] == 24
        ), "The size of the next_state is not (24) in REPLAY BUFFER -> next_state size: {}.".format(
            next_state.shape[0]
        )

        assert (
            state.ndim == 1
        ), "The ndim of the state is not (1) in REPLAY BUFFER -> state ndim: {}.".format(
            state.ndim
        )
        assert (
            action.ndim == 1
        ), "The ndim of the action is not (1) in REPLAY BUFFER -> action ndim: {}.".format(
            state.ndim
        )
        if isinstance(reward, np.float64):
            assert (
                reward.ndim == 0
            ), "The ndim of the reward is not (0) in REPLAY BUFFER -> reward ndim: {}.".format(
                reward.ndim
            )
        assert (
            next_state.ndim == 1
        ), "The ndim of the next_state is not (1) in REPLAY BUFFER -> next_state ndim: {}.".format(
            next_state.ndim
        )

        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = (
            torch.from_numpy(np.vstack([e.state for e in experiences if e is not None]))
            .float()
            .to(DEVICE)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
            .float()
            .to(DEVICE)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(DEVICE)
        )
        next_states = (
            torch.from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(DEVICE)
        )
        dones = (
            torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]))
            .float()
            .to(DEVICE)
        )

        assert isinstance(
            states, torch.Tensor
        ), "State is not of type torch.Tensor in REPLAY BUFFER."
        assert isinstance(
            actions, torch.Tensor
        ), "Actions is not of type torch.Tensor in REPLAY BUFFER."
        assert isinstance(
            rewards, torch.Tensor
        ), "Rewards is not of type torch.Tensor in REPLAY BUFFER."
        assert isinstance(
            next_states, torch.Tensor
        ), "Next states is not of type torch.Tensor in REPLAY BUFFER."
        assert isinstance(
            dones, torch.Tensor
        ), "Dones is not of type torch.Tensor in REPLAY BUFFER."

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
            dones.dtype == torch.float32
        ), "The (dones) tensor elements are not of type torch.float32 in the REPLAY BUFFER -> {}.".format(
            dones.dtype
        )

        # TODO
        # assert all(tensor.device.type == DEVICE for tensor in [states, actions, rewards, next_states, dones]), "Each tensor must be on the same device in REPLAY BUFFER"

        return (states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return len(self.memory)
