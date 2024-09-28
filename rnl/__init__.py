# from gymnasium.envs.registration import register

# register(
#     id="microvault/NaviEnv-v0",
#     entry_point="microvault.environment.environment_navigation:NaviEnv",
# )

from rnl.training.interface import Trainer, make, render, robot, sensor

text = [
    r"+--------------------+",
    r" ____  _   _ _",
    r"|  _ \| \ | | |",
    r"| |_) |  \| | |",
    r"|  _ <| |\  | |___",
    r"|_| \_\_| \_|_____|",
    r"_____________________",
]

for line in text:
    print(line)

__all__ = ["robot", "sensor", "render", "make", "Trainer"]
