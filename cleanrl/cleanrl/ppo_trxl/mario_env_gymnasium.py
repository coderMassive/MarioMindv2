import random
import gymnasium as gym
from gym_super_mario_bros.smb_env import SuperMarioBrosEnv

class MarioEnv(SuperMarioBrosEnv):
    metadata = {"render_modes": ["human", "rgb_array", "debug_rgb_array"], "render_fps": 60}

    def __init__(self, skip=4, render_mode=None, **kwargs):
        if render_mode == "debug_rgb_array":
            render_mode = "rgb_array"
        self.render_mode = render_mode
        super().__init__(target=self._random_target())
        self._skip = skip

    @staticmethod
    def _random_target():
        world = random.randint(1, 8)
        stage = random.randint(1, 4)
        return (world, stage)

    def _write_stage(self):
        world, stage = self._random_target()
        self._target_world = world
        self._target_stage = stage
        self._target_area = stage
        super()._write_stage()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
        self._write_stage()
        obs = super().reset()
        return obs, {}

    def _did_reset(self):
        super()._did_reset()
        self._time_stop = None
        self._x_stop = 0
        self._score_last = 0
        self.ram[0x075A] = 0
        self._action_current = 0
        self._action_last = 0

    @property
    def _stuck(self):
        if self._x_position - self._x_stop > 1:
            self._time_stop = None
            self._x_stop = self._x_position
            return False
        if self._time_stop is None:
            self._time_stop = self._time
            self._x_stop = self._x_position
            return False
        return self._time_stop - self._time > 10

    def step(self, action):
        obs = None
        reward = 0
        terminated = False
        truncated = False
        info = {}

        self._action_current = action

        for _ in range(self._skip):
            obs, temp, done, info = super().step(action)
            reward += temp
            if done:
                terminated = True
                break

        if self._stuck and not terminated:
            truncated = True

        return obs, reward, terminated, truncated, info

    @property
    def _score_reward(self):
        reward = self._score - self._score_last
        self._score_last = self._score
        return reward if reward > 0 else 0

    @property
    def _jump_penalty(self):
        reward = 0
        if self._action_last < 128 and self._action_current >= 128:
            reward = -30
        self._action_last = self._action_current
        return reward

    def _get_reward(self):
        return self._x_reward + self._score_reward + self._jump_penalty
