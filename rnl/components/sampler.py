from typing import List, Optional, Union

from rnl.components.replay_buffer import MultiStepReplayBuffer, PrioritizedReplayBuffer


class Sampler:
    """Sampler class to handle both standard and distributed training."""

    def __init__(
        self,
        per: bool,
        n_step: bool,
        memory: Union["PrioritizedReplayBuffer", "MultiStepReplayBuffer"],
    ):
        self.per = per
        self.n_step = n_step
        self.memory = memory

    def sample(
        self,
        batch_size: Optional[int] = None,
        beta: Optional[float] = None,
        idxs: Optional[List[int]] = None,
    ):
        if batch_size is None:
            if self.n_step:
                if idxs is None:
                    raise ValueError("Para sampling N-step, 'idxs' deve ser fornecido.")
                return self._sample_n_step(idxs)
            else:
                raise ValueError("Batch size deve ser fornecido se não for N-step.")
        else:
            if self.per:
                if beta is None:
                    raise ValueError("Para sampling PER, 'beta' deve ser fornecido.")
                return self._sample_per(batch_size, beta)
            else:
                return self._sample_standard(batch_size)

    def _sample_standard(self, batch_size: int, return_idx: bool = False):
        return self.memory.sample(batch_size, return_idx)

    def _sample_per(self, batch_size: int, beta: float):
        # Use isinstance para checar o tipo do buffer
        if isinstance(self.memory, PrioritizedReplayBuffer):
            return self.memory.sample_per(batch_size, beta)
        else:
            raise TypeError("Memory não é uma instância de PrioritizedReplayBuffer.")

    def _sample_n_step(self, idxs: List[int]):
        return self.memory.sample_from_indices(idxs)
