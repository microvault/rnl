import random

import numpy as np
import torch


class Engine:

    def __init__(self, seed: int = 42, device: str = "cpu"):
        self.seed = seed
        self.device = device

    def seed_everything(self):
        # torch.manual_seed(self.seed)
        # np.random.seed(self.seed)
        # random.seed(self.seed)

        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed_all(self.seed)
        #     torch.backends.cudnn.deterministic = True
        #     torch.backends.cudnn.benchmark = False

        # Metal Performance Shaders
        if torch.backends.mps.is_available():
            print("MPS available")

    def set_device(self):
        # clear the cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Set the device globally
        torch.set_default_device(self.device)
