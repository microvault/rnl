import copy
import random
import numpy as np
from rnl.components.noise import OUNoise

class Randomization:
    def __init__(
        self,
        robot_config,
        sensor_config,
        env_config,
        seed=42,
        theta=0.15,
        sigma=0.2
    ):
        self.configs = {
            'robot_config': robot_config,
            'sensor_config': sensor_config,
            'env_config': env_config,
        }
        self.noises = {}
        self.seed = seed
        self.theta = theta
        self.sigma = sigma
        self._initialize_noises()

    def _initialize_noises(self):
        for config_name, config in self.configs.items():
            self.noises[config_name] = {}
            for attr, value in vars(config).items():
                min_attr = f"{attr}_min"
                max_attr = f"{attr}_max"
                if hasattr(config, min_attr) and hasattr(config, max_attr):
                    if isinstance(value, (int, float)):
                        size = 1
                    elif isinstance(value, (list, np.ndarray)):
                        size = len(value)
                    else:
                        continue  # Ignora atributos que não são numéricos ou listas
                    noise = OUNoise(size=size, seed=self.seed, theta=self.theta, sigma=self.sigma)
                    self.noises[config_name][attr] = noise

    def randomize(self):
        for config_name, config in self.configs.items():
            for attr, noise in self.noises[config_name].items():
                sampled_noise = noise.sample()
                current_value = getattr(config, attr)
                min_value = getattr(config, f"{attr}_min")
                max_value = getattr(config, f"{attr}_max")
                
                if isinstance(current_value, list):
                    noise_value = sampled_noise.tolist()
                    new_values = [max(min_val, min(val + n, max_val)) 
                                  for val, n, min_val, max_val in zip(current_value, noise_value, 
                                                                     getattr(config, f"{attr}_min"), 
                                                                     getattr(config, f"{attr}_max"))]
                    setattr(config, attr, new_values)
                elif isinstance(current_value, (int, float, np.number)):
                    noise_value = sampled_noise[0]
                    new_value = current_value + noise_value
                    new_value = max(min_value, min(new_value, max_value))
                    setattr(config, attr, new_value)
                    
    def reset(self):
        """Redefinir todos os processos de ruído para seus estados iniciais."""
        for config_name, noise_attrs in self.noises.items():
            for attr, noise in noise_attrs.items():
                noise.reset()