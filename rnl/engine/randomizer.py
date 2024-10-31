import copy
import random
import numpy as np

class DomainRandomization:
    def __init__(
        self,
        robot_config: RobotConfig,
        sensor_config: SensorConfig,
        env_config: EnvConfig,
        robot_config_mins: dict,
        robot_config_maxs: dict,
        sensor_config_mins: dict,
        sensor_config_maxs: dict,
        env_config_mins: dict,
        env_config_maxs: dict,
    ):
        self.robot_config = robot_config
        self.sensor_config = sensor_config
        self.env_config = env_config
        self.robot_config_mins = robot_config_mins
        self.robot_config_maxs = robot_config_maxs
        self.sensor_config_mins = sensor_config_mins
        self.sensor_config_maxs = sensor_config_maxs
        self.env_config_mins = env_config_mins
        self.env_config_maxs = env_config_maxs

    def randomize_config(self, config, min_values, max_values):
        randomized_config = copy.deepcopy(config)
        for field_name in config.__dataclass_fields__:
            value = getattr(config, field_name)
            min_value = min_values[field_name]
            max_value = max_values[field_name]
            if isinstance(value, (float, int)):
                # Randomize within min and max
                new_value = random.uniform(min_value, max_value)
                setattr(randomized_config, field_name, type(value)(new_value))
            elif isinstance(value, list):
                # Randomize each element in the list
                new_value = []
                for idx, v in enumerate(value):
                    v_min = min_value[idx]
                    v_max = max_value[idx]
                    rand_val = random.uniform(v_min, v_max)
                    new_value.append(type(v)(rand_val))
                setattr(randomized_config, field_name, new_value)
            else:
                # For other types, keep the original value
                setattr(randomized_config, field_name, value)
        return randomized_config

    def get_randomized_configs(self):
        randomized_robot_config = self.randomize_config(
            self.robot_config, self.robot_config_mins, self.robot_config_maxs
        )
        randomized_sensor_config = self.randomize_config(
            self.sensor_config, self.sensor_config_mins, self.sensor_config_maxs
        )
        randomized_env_config = self.randomize_config(
            self.env_config, self.env_config_mins, self.env_config_maxs
        )
        return randomized_robot_config, randomized_sensor_config, randomized_env_config