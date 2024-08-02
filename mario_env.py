from gym_super_mario_bros.smb_env import SuperMarioBrosEnv

class MarioEnv(SuperMarioBrosEnv):
    def __init__(self, skip=1):
        super(MarioEnv, self).__init__()
        self._skip = skip
        self.reward_range = (-30, 100)

    def _did_reset(self):
        super(MarioEnv, self)._did_reset()
        self._time_stop = None
        self._x_stop = 0
        self._score_last = 0
        self.ram[0x075a] = 0
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
        done = False
        info = {}

        self._action_current = action

        for _ in range(self._skip):
            obs, temp, done, info = super(MarioEnv, self).step(action)
            reward += temp
            if done:
                break

        if self._stuck:
            info["truncated"] = True
            
        return obs, reward, done, info
    
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
