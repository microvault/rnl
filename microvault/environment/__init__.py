from gymnasium.envs.registration import register

register(
    id="NaviEnv-v0",
    entry_point="microvault.environment.environment_navigation:NaviEnv",  # Adjust this if NaviEnv is in a different module
    max_episode_steps=500000,
)
