# Robot Navigation Learning

<div align="center">
     <img src="https://raw.githubusercontent.com/microvault/rnl/main/docs/_static/img.png" alt="MicroVault">
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

This project uses Deep Reinforcement Learning (DRL) to train a robot to move in unfamiliar environments. The robot learns to make decisions on its own, interacting with the environment, and gradually becomes better and more efficient at navigation.

<details>
  <summary>How to Use</summary>

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
    base_radius=0.033,  # (m)
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

*	**Use** `inference`:
```python
import numpy as np
import rnl as vault

# 1.step -> config robot
param_robot = vault.robot(
    base_radius=0.033,  # (m)
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

* **Use** `demo`:
```bash
python train.py
```
</details>

## License
This project is licensed under the MIT license - see archive [LICENSE](https://github.com/microvault/rnl/blob/main/LICENSE) for details.

## Contact and Contribution
The project is still under development and may have some bugs. If you encounter any problems or have suggestions, feel free to open an `issue` or send an `email` to:
Nicolas Alan - **grottimeireles@gmail.com**.


## TODO:
- [ ] Add map file yaml
- [ ] Add random mode (hard or normal)
- [ ] Create Integration ROS and (Gazebo, webots)
- [ ] Create Integration with OpenAI o1-preview


## Acknowledgments
* [AgileRL](https://github.com/AgileRL/AgileRL)
