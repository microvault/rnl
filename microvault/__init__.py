from gym.envs.registration import register

register(
    id="microvault/NaviEnv-v0",
    entry_point="microvault.environment.environment_navigation:NaviEnv",
)
