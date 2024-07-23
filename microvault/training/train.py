import gym
import matplotlib.pyplot as plt

import microvault

# Criação do ambiente
env = gym.make("microvault/NaviEnv-v0", render=True)

# Interação com o ambiente
state = env.reset()
done = False

# for timestep in range(100):
#     print(f"iteração {timestep}")
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action, timestep)
ani = env.render()
plt.show()
env.close()
