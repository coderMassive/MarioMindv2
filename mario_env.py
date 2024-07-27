from gym_super_mario_bros.smb_env import SuperMarioBrosEnv

class MarioEnv(SuperMarioBrosEnv):
    def __init__(self, skip=1):
        super(MarioEnv, self).__init__()
        self._skip = skip
        self.reward_range = (-3 * skip, 100)

    def reset(self):
        self._time_stop = None
        self._x_stop = 0
        self._score_last = 0
        return super(MarioEnv, self).reset()

    def step(self, action):
        if self._stuck():
            self._kill_mario()

        obs = None
        reward = 0
        done = False
        info = {}
        for _ in range(self._skip):
            obs, temp, done, info = super(MarioEnv, self).step(action)
            reward += temp
            if done:
                break

        return obs, reward, done, info

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
    
    @property
    def _score_reward(self):
        reward = self._score - self._score_last
        self._score_last = self._score
        return reward if reward > 0 else 0
    
    def _get_reward(self):
        return self._x_reward + self._score_reward

