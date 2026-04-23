from gymnasium.envs.registration import register

register(
    id="RandomMario-v0",
    entry_point="mario_env_gymnasium:MarioEnv",
    kwargs={"skip": 4},
    disable_env_checker=True,
)
