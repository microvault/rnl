# Robot Navigation Learning

<div align="center">
     <img src="https://raw.githubusercontent.com/microvault/rnl/main/docs/_static/img.png" alt="MicroVault">
</div>

<p align="center">
  <a href='https://microvault.readthedocs.io/en/latest/?badge=latest'><img src='https://readthedocs.org/projects/microvault/badge/?version=latest' alt='Documentation Status' /></a>
  <a href="https://pypi.org/project/microvault/"><img alt="PyPI" src="https://img.shields.io/pypi/v/microvault"></a>
  <a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
  <a href="https://codecov.io/gh/microvault/microvault"><img alt="codecov" src="https://codecov.io/gh/microvault/microvault/graph/badge.svg?token=WRTOBP06AW"></a>
  <a href="https://github.com/microvault/microvault/actions/workflows/main.yaml"><img alt="CI" src="https://github.com/microvault/microvault/actions/workflows/main.yaml/badge.svg"></a>
  <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://codeclimate.com/github/microvault/microvault/maintainability"><img src="https://api.codeclimate.com/v1/badges/f121e3b57214eac38280/maintainability" /></a>

</p>

<div align="center">

**End-to-end Deep Reinforcement Learning for Real-World Robotics Navigation in Pytorch**

</div>

This project uses Deep Reinforcement Learning (DRL) to train a robot to move in unfamiliar environments. The robot learns to make decisions on its own, interacting with the environment, and gradually becomes better and more efficient at navigation.

- [About the Project](#About)
- [Dependencies](#Dependencies)
- [Installation](#Installation)
- [How to Use](#How-to-Use)
- [License](#license)
- [Contact](#Contact)

## About the Project
<a name="About"></a>

This project uses RainbowDQN to train a robot to navigate autonomously without needing heavy installations like ROS, Gazebo, or the entire navigation stack. The focus is to create a lightweight and easy-to-use agent that learns to make decisions on its own by interacting with the environment and chooses the best way to navigate safely and stably. Over time, the robot will improve its navigation skills, becoming more efficient.

## Dependencies:
<a name="Dependencies"></a>
- Python 3.12.4
- PyTorch

## Installation
<a name="Installation"></a>
1. Clone or repository:
```bash
git clone https://github.com/microvault/rnl.git
cd rnl
```
2. Create a virtual environment (optional, but recommended), and install the dependencies:
```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry install
```
3. Activate the virtual environment:
```bash
poetry shell
```
## How to Use
<a name="How-to-Use"></a>

You have two options to get started: you can use the `train_model_base.py` file to get started quickly, or you can create your own script from scratch if you prefer more customization.

1. Using the ready script:
```bash
make train
make run
```

2.	Adding in python `train`:
```python
import numpy as np
import rnl as vault

# 1.step -> config robot
param_robot = vault.robot(
    base_radius=0.033,  # (cm)
    vel_linear=[0.0, 2.0],  # [min, max]
    val_angular=[1.0, 2.0],  # [min, max]
    wheel_distance=0.16,  # (cm)
    weight=1.0,  # robot (kg)
    threshold=0.01,  # distance for obstacle avoidance (cm)
)

# 2.step -> config sensors [for now only lidar sensor!!]
param_sensor = vault.sensor(
    fov=4 * np.pi,
    num_rays=20,
    min_range=0.0,
    max_range=6.0,
)

# 3.step -> config env
param_env = vault.make(
    map_file="None", # map file yaml (Coming soon)
    random_mode="normal",  # hard or normal (Coming soon)
    timestep=1000,  # max timestep
    grid_dimension=5,  # size grid
    friction=0.4,  # grid friction
    porcentage_obstacles=0.1
)

# 4.step -> config train robot
model = vault.Trainer(
    param_robot, param_sensor, param_env, pretrained_model=False
)

# 5.step -> train robot
model.learn(
    batch_size=64,
    lr=0.0001,
    seed=1,
    num_envs=2,
    device="cpu",
    target_score=200,
    checkpoint=100,
    checkpoint_path="checkpoints",
    hidden_size=[800, 600],
)

```

3.	Adding in python `inference`:
```python
import numpy as np
import rnl as vault

# 1.step -> config robot
param_robot = vault.robot(
    base_radius=0.033,  # (cm)
    vel_linear=[0.0, 2.0],  # [min, max]
    val_angular=[1.0, 2.0],  # [min, max]
    wheel_distance=0.16,  # (cm)
    weight=1.0,  # robot (kg)
    threshold=0.01,  # distance for obstacle avoidance (cm)
)

# 2.step -> config sensors [for now only lidar sensor!!]
param_sensor = vault.sensor(
    fov=4 * np.pi,
    num_rays=20,
    min_range=0.0,
    max_range=6.0,
)

# 3.step -> config env
param_env = vault.make(
    map_file="None", # map file yaml (Coming soon)
    random_mode="normal",  # hard or normal (Coming soon)
    timestep=1000,  # max timestep
    grid_dimension=5,  # size grid
    friction=0.4,  # grid friction
    porcentage_obstacles=0.1
)

# 4.step -> config render
param_render = vault.render(fps=100, controller=True, rgb_array=True)


# 5.step -> config train robot
model = vault.Trainer(
    param_robot, param_sensor, param_env, pretrained_model=False
)

# 6.step -> run robot
model.run()
```

## License
<a name="License"></a>
This project is licensed under the MIT license - see archive [LICENSE](https://github.com/microvault/rnl/blob/main/LICENSE) for details.

## Contact and Contribution
<a name="Contact"></a>
The project is still under development and may have some bugs. If you encounter any problems or have suggestions, feel free to open an `issue` or send an `email` to:
Nicolas Alan - **grottimeireles@gmail.com**.


## TODO:
- [ ] Add map file yaml
- [ ] Add random mode (hard or normal)
- [ ] Create Integration ROS and (Gazebo, webots)
- [ ] Create Integration with OpenAI o1-preview


## Acknowledgments

```bibtex
@software{Ustaran-Anderegg_AgileRL,
  author = {Ustaran-Anderegg, Nicholas and Pratt, Michael},
  license = {Apache-2.0},
  title = {{AgileRL}},
  url = {https://github.com/AgileRL/AgileRL}
}
```
