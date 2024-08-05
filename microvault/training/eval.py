import gymnasium as gym

env = gym.make("microvault/NaviEnv-v0", controller=True)

env.reset()
env.render()
