import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GymnasiumSpaceCompatibility(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # Convert action space
        act = env.action_space
        if hasattr(act, "n"):  # old gym.spaces.Discrete
            self.action_space = spaces.Discrete(int(act.n))
        elif hasattr(act, "shape") and hasattr(act, "dtype"):
            low = getattr(act, "low", None)
            high = getattr(act, "high", None)
            if low is not None and high is not None:
                self.action_space = spaces.Box(
                    low=np.array(low),
                    high=np.array(high),
                    shape=act.shape,
                    dtype=act.dtype,
                )
            else:
                self.action_space = act
        else:
            self.action_space = act

        # Convert observation space
        obs = env.observation_space
        if hasattr(obs, "shape") and hasattr(obs, "dtype"):
            low = getattr(obs, "low", None)
            high = getattr(obs, "high", None)

            if low is not None and high is not None:
                self.observation_space = spaces.Box(
                    low=np.array(low),
                    high=np.array(high),
                    shape=obs.shape,
                    dtype=obs.dtype,
                )
            else:
                self.observation_space = spaces.Box(
                    low=0,
                    high=255,
                    shape=obs.shape,
                    dtype=obs.dtype if obs.dtype is not None else np.uint8,
                )
        else:
            self.observation_space = obs
