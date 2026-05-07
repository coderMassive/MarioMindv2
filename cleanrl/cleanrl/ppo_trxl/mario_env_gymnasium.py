import random
import gymnasium as gym
from gymnasium import spaces
from gym_super_mario_bros.smb_env import SuperMarioBrosEnv

class MarioEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "debug_rgb_array"],
        "render_fps": 60,
    }

    def __init__(self, skip=4, render_mode=None, **kwargs):
        if render_mode == "debug_rgb_array":
            render_mode = "rgb_array"

        self.skip = skip
        self.render_mode = render_mode
        self.reward_range = (-30, 100)

        self.env = None
        self._target_world = None
        self._target_stage = None

        self._make_new_env()

    def _random_target(self):
        world = random.randint(1, 8)
        stage = random.randint(1, 4)
        return world, stage

    def _make_new_env(self):
        if self.env is not None:
            self.env.close()

        world, stage = self._random_target()
        self._target_world = world
        self._target_stage = stage

        self.env = SuperMarioBrosEnv(target=(world, stage))

        self.action_space = spaces.Discrete(int(self.env.action_space.n))

        obs = self.env.observation_space
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=obs.shape,
            dtype=obs.dtype,
        )

        self._time_stop = None
        self._x_stop = 0
        self._score_last = 0
        self._action_current = 0
        self._action_last = 0

        print(f"[MarioEnv] New level: {world}-{stage}")

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            random.seed(seed)

        self._make_new_env()

        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        return obs, {
            "world": self._target_world,
            "stage": self._target_stage,
        }

    def step(self, action):
        obs = None
        reward = 0
        done = False
        info = {}

        self._action_current = int(action)

        for _ in range(self.skip):
            obs, temp, done, info = self.env.step(self._action_current)
            reward += temp

            if done:
                break

        if self._stuck:
            done = False
            truncated = True
        else:
            truncated = False

        reward += self._score_reward
        reward += self._jump_penalty

        info["world"] = self._target_world
        info["stage"] = self._target_stage

        terminated = bool(done)

        return obs, reward, terminated, truncated, info

    def render(self):
        frame = self.env.render(mode="rgb_array")
        return frame

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    @property
    def _x_position(self):
        return self.env._x_position

    @property
    def _time(self):
        return self.env._time

    @property
    def _score(self):
        return self.env._score

    @property
    def _x_reward(self):
        return self.env._x_reward

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
