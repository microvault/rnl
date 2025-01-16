from gymnasium.envs.registration import register

register(
    id="NaviEnv-v0",
    entry_point="rnl.environment.env:NaviEnv",
    max_episode_steps=1000000,
)
