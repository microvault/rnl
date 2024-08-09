from microvault.environment.environment_navigation import NaviEnv
import gymnasium as gym

env = gym.make("NaviEnv-v0", rgb_array=True, controller=True)

env.reset()
env.render()
