from utils import process_results
import gymnasium as gym
from gymnasium import spaces
import cv2
import numpy as np
from ultralytics import YOLO

class YOLOEnvironment(gym.Env):
    def __init__(self, env, max_boxes):
        super(YOLOEnvironment, self).__init__()
        self.env = env
        self.model = YOLO('best.engine')
        self.max_boxes = max_boxes

        # Observation space: (x, y, w, h, confidence, class_id) * max_boxes
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.max_boxes, 6),
            dtype=np.float32
        )

        # Action space (example): Continuous space with two actions
        self.action_space = spaces.Discrete(env.action_space.n)

        self.reward_range = self.env.reward_range

    def reset(self, seed=None, options=None):
        self.env.reset()
        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        obs = self._get_observation()
        truncated = False
        return obs, reward, done, truncated, info

    def _get_observation(self):
        frame = self.env.render(mode='rgb_array')
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = self.model(image, verbose=False)
        obs = process_results(results, self.max_boxes)
        return obs
    
