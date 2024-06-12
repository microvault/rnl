import numpy as np
import torch
from sumtree import SumTree

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PER:

    def __init__(self, buffer_size: int, batch_size: int):
        """
        The tree is composed of a sum tree that contains the priority scores at his leaf and also a data array.
        """
        self.tree = SumTree(buffer_size)
        self.epsilon = 0.01  # small amount to avoid zero priority
        self.alpha = (
            0.6  # [0..1] convert the importance of TD error to priority, often 0.6
        )
        self.beta = (
            0.4  # importance-sampling, from initial value increasing to 1, often 0.4
        )
        self.beta_increment_per_sampling = 1e-4  # annealing the bias, often 1e-3
        self.absolute_error_upper = 1.0  # clipped abs error
        self.batch_size = batch_size

    def __len__(self):
        return len(self.tree)

    def is_full(self):
        return len(self.tree) >= self.tree.capacity

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.float64,
        next_state: np.float64,
        done: bool,
        error: np.float64 = None,
    ):
        assert isinstance(
            state[0], np.float32
        ), "State is not of type (np.float32) in REPLAY BUFFER -> state type: {}.".format(
            type(state)
        )
        # assert isinstance(action[0], np.float32), "Action is not of type (np.float32) in REPLAY BUFFER -> action type: {}.".format(type(action))
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

        sample = (state, action, reward, next_state, done)
        if error is None:
            priority = np.amax(self.tree.tree[-self.tree.capacity :])
            if priority == 0:
                priority = self.absolute_error_upper
        else:
            priority = min(
                (abs(error) + self.epsilon) ** self.alpha, self.absolute_error_upper
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
            torch.from_numpy(np.vstack([e.state for e in minibatch if e is not None]))
            .float()
            .to(device)
        )
        actions = (
            torch.from_numpy(np.vstack([e.action for e in minibatch if e is not None]))
            .float()
            .to(device)
        )
        rewards = (
            torch.from_numpy(np.vstack([e.reward for e in minibatch if e is not None]))
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(
                np.vstack([e.next_state for e in minibatch if e is not None])
            )
            .float()
            .to(device)
        )
        dones = (
            torch.from_numpy(np.vstack([e.done for e in minibatch if e is not None]))
            .int()
            .to(device)
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
            dones.dtype == torch.int
        ), "The (dones) tensor elements are not of type torch.float32 in the REPLAY BUFFER -> {}.".format(
            dones.dtype
        )

        return idxs, (states, actions, rewards, next_states, dones), is_weights

    def batch_update(self, idxs, errors):
        """
        Update the priorities on the tree
        """
        errors = errors + self.epsilon
        clipped_errors = np.minimum(errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.alpha)

        for idx, p in zip(idxs, ps):
            self.tree.update(idx, p)
