import microvault as vault
import numpy as np

if __name__ == "__main__":
    # 1.step -> config robot
    param_robot = vault.robot(
        radius=40,  # (centimeters)
        vel_linear=[0.0, 2.0],  # [min, max]
        val_angular=[1.0, 2.0],  # [min, max]
    )

    # 2.step -> config sensors [for now only lidar sensor!!]
    param_sensor = vault.sensor(
        fov=4 * np.pi, num_rays=20, min_range=0.0, max_range=6.0  # int
    )
    # 3.step -> config env
    param_env = vault.make(
        map="None",
        mode="normal",  # hard (muda tudo), normal (mapa fixo)
        timestep=1000,
        fps=100,
        threshold=0.05,
        grid_lenght=5,
        physical="normal",
    )

    # 4.step -> config train robot
    model = vault.Trainer(param_robot, param_sensor, param_env, pretrained_model=False)
    # 5.step -> train robot
    model.learn(
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
        num_envs=1,
        device="mps",
        learn_step=10,
        n_step=3,
        memory_size=1000000,
        target_score=200.0,
    )
