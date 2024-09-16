from rnl.configs.config import EnvConfig, RobotConfig, SensorConfig
from rnl.training.learn import training, inference


def robot(radius, vel_linear, val_angular):
    return RobotConfig(radius, vel_linear, val_angular)


def sensor(fov, num_rays, min_range, max_range):
    return SensorConfig(fov, num_rays, min_range, max_range)


def make(map, mode, timestep, fps, threshold, grid_lenght, physical, rgb_array=False):
    return EnvConfig(
        map, mode, timestep, fps, threshold, grid_lenght, physical, rgb_array
    )


class Trainer:
    def __init__(
        self,
        robot_config: RobotConfig,
        sensor_config: SensorConfig,
        env_config: EnvConfig,
        pretrained_model=False,
    ):
        self.robot_config = robot_config
        self.sensor_config = sensor_config
        self.env_config = env_config
        self.pretrained_model = pretrained_model

    def learn(
        self,
        max_timestep=800000,
        use_mutation=False,
        freq_evolution=10000,
        log=False,
        batch_size=64,
        lr=0.0001,
        pop_size=6,
        hidden_size=[800, 600],
        no_mut=0.4,
        arch_mut=0.2,
        new_layer=0.2,
        param_mut=0.2,
        act_mut=0,
        hp_mut=0.2,
        mut_strength=0.1,
        seed=1,
        num_envs=16,
        device="mps",
        learn_step=10,
        n_step=3,
        memory_size=1000000,
        target_score=200.0,
    ):
        training(
            max_timestep,
            use_mutation,
            freq_evolution,
            log,
            batch_size,
            lr,
            pop_size,
            hidden_size,
            no_mut,
            arch_mut,
            new_layer,
            param_mut,
            act_mut,
            hp_mut,
            mut_strength,
            seed,
            num_envs,
            device,
            learn_step,
            n_step,
            memory_size,
            target_score,
            self.robot_config,
            self.sensor_config,
            self.env_config,
            self.pretrained_model,
        )

    def run(self):

        inference(
            self.robot_config,
            self.sensor_config,
            self.env_config,
            self.pretrained_model,
        )
