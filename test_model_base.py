import argparse

import numpy as np

import rnl as vault

if __name__ == "__main__":
    # Setup argparse
    parser = argparse.ArgumentParser(description="Train or setup environment.")
    parser.add_argument(
        "mode", choices=["train", "run"], help="Mode to run: 'train' or 'run'"
    )

    # Parse arguments
    args = parser.parse_args()

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
        rgb_array=True,
    )

    if args.mode == "train":
        # 4.step -> config train robot
        model = vault.Trainer(
            param_robot, param_sensor, param_env, pretrained_model=False
        )
        # 5.step -> train robot
        model.learn()

    elif args.mode == "run":
        # 4.step -> config train robot
        model = vault.Trainer(
            param_robot, param_sensor, param_env, pretrained_model=False
        )
        # 5.step -> train robot
        model.run()
