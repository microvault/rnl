from typing import Iterator

from torch.utils.data.dataset import IterableDataset


class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

    >>> RLDataset(ReplayBuffer(5))  # doctest: +ELLIPSIS
    <...reinforce_learn_Qnet.RLDataset object at ...>

    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        """
        Args:
            buffer: replay buffer
            sample_size: number of experiences to sample at a time
        """
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator:
        states, actions, rewards, dones, new_states = self.buffer.sample()
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]


from typing import Callable, Iterable

from torch.utils.data import IterableDataset


class ExperienceSourceDataset(IterableDataset):

    def __init__(self, generate_batch: Callable):
        self.generate_batch = generate_batch

    def __iter__(self) -> Iterable:
        iterator = self.generate_batch()
        return iterator
