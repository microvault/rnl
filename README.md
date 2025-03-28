<div align="center">
     <img src="https://raw.githubusercontent.com/microvault/rnl/main/docs/images/rnl.png" alt="MicroVault">
</div>

<p align="center">
  <a href='https://microvault.readthedocs.io/en/latest/?badge=latest'><img src='https://readthedocs.org/projects/microvault/badge/?version=latest' alt='Documentation Status' /></a>
  <a href="https://pypi.org/project/rnl/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rnl"></a>
  <a href="https://codecov.io/gh/microvault/microvault"><img alt="codecov" src="https://codecov.io/gh/microvault/microvault/graph/badge.svg?token=WRTOBP06AW"></a>
  <a href="https://github.com/microvault/microvault/actions/workflows/main.yaml"><img alt="CI" src="https://github.com/microvault/microvault/actions/workflows/main.yaml/badge.svg"></a>
<a href="https://codeclimate.com/github/microvault/microvault/maintainability"><img src="https://api.codeclimate.com/v1/badges/f121e3b57214eac38280/maintainability" /></a>

</p>

<div align="center">

**End-to-end Deep Reinforcement Learning for Real-World Robotics Navigation in Pytorch**

</div>

> **Warning** :
> This project is still in progress and not yet finalized for release for use.

This project uses Deep Reinforcement Learning (DRL) to train a robot to move in unfamiliar environments. The robot learns to make decisions on its own, interacting with the environment, and gradually becomes better and more efficient at navigation.

### How to Use

Installation and usage mode.

* **Install with pip**:
```bash
pip install rnl
```

*	**Use** `train`:
```python
import numpy as np
import rnl as vault

# 1.step -> config robot
param_robot = vault.robot(
    base_radius=0.105,  # (m)
    vel_linear=[0.0, 0.22],  # [min, max]
    vel_angular=[1.0, 2.84],  # [min, max]
    wheel_distance=0.16,  # (m)
    weight=1.0,  # robot (kg)
    threshold=1.0,  # distance for obstacle avoidance (m)
    collision=0.5,
    path_model="None",
)

# 2.step -> config sensors [for now only lidar sensor!!]
param_sensor = vault.sensor(
    fov=2 * np.pi,
    num_rays=20,
    min_range=0.0,
    max_range=6.0,
)

# 3.step -> config env
param_env = vault.make(
    scale=100,
    folder_map="None",
    name_map="None",
    max_timestep=10000,
    mode="easy-01",
)

# 4. step -> config render
param_render = vault.render(controller=False, debug=True, plot=False)

# 5.step -> config train robot
model = vault.Trainer(
    param_robot, param_sensor, param_env, param_render
)

# 6.step -> train robot
model.learn(
  algorithm="PPO",
  max_timestep_global=3000000,
  seed=1,
  buffer_size=1000000,
  hidden_size=[20, 10],
  activation="ReLu",
  batch_size=1024,
  num_envs=4,
  device="cuda",
  checkpoint="model",
  use_wandb=True,
  wandb_api_key="",
  lr=0.0003,
  learn_step=512,
  gae_lambda=0.95,
  action_std_init=0.6,
  clip_coef=0.2,
  ent_coef=0.0,
  vf_coef=0.5,
  max_grad_norm=0.5,
  update_epochs=10,
  name="models",
)

```

*	**Use** `inference`:
```python
import numpy as np
import rnl as vault

# 1.step -> config robot
param_robot = vault.robot(
    base_radius=0.105,  # (m)
    vel_linear=[0.0, 0.22],  # [min, max]
    vel_angular=[1.0, 2.84],  # [min, max]
    wheel_distance=0.16,  # (m)
    weight=1.0,  # robot (kg)
    threshold=1.0,  # distance for obstacle avoidance (m)
    collision=0.5,
    path_model="None",
)

# 2.step -> config sensors [for now only lidar sensor!!]
param_sensor = vault.sensor(
    fov=2 * np.pi,
    num_rays=20,
    min_range=0.0,
    max_range=6.0,
)

# 3.step -> config env
param_env = vault.make(
    scale=100,
    folder_map="None",
    name_map="None",
    max_timestep=10000,
    mode="easy-01",
)

# 4.step -> config render
param_render = vault.render(controller=False, debug=True, plot=False)

# 5.step -> config train robot
vault.Simulation(param_robot, param_sensor, param_env, param_render)

# 6.step -> run robot
model.run()
```

* **Use** `demo`:
```bash
python main.py -m sim
```

## License
This project is licensed under the MIT license - see archive [LICENSE](https://github.com/microvault/rnl/blob/main/LICENSE) for details.

## Contact and Contribution
The project is still under development and may have some bugs. If you encounter any problems or have suggestions, feel free to open an `issue` or send an `email` to:
Nicolas Alan - **grottimeireles@gmail.com**.
