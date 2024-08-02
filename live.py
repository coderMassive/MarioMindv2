from utils import process_results
import cv2
import numpy as np
import time
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from stable_baselines3 import PPO
from nes_py.wrappers import JoypadSpace
from inference import get_model

MAX_BOXES = 10
FPS = 60
SKIP = 4

env = SuperMarioBrosEnv()
env = JoypadSpace(env, SIMPLE_MOVEMENT)

model = PPO.load("./tmp/best_model.zip")
img_model = get_model(model_id="mario-ibyfv/2", api_key="ITukAND4XqHSos8UA9me")

images = []
prev_obs = None

env.reset()
while True:
    action = None
    done = False
    for i in range(SKIP):
        frameStart = time.time()
        # performs action on environment
        if i == 0:
            # processes frame into bounding boxes
            frame = env.render(mode='rgb_array')
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            results = img_model.infer(image)
            obs = process_results(results, MAX_BOXES)
            prev_obs = obs if prev_obs is None else prev_obs

            action, _states = model.predict([prev_obs, obs])
            prev_obs = obs

        env.render()
        _, _, done, _ = env.step(action.item())

        if done:
            env.reset()
        
        frameEnd = time.time()
        if frameEnd - frameStart < 1/FPS:
            time.sleep(frameEnd - frameStart)

    if done:
      env.reset()