from utils import process_results
from mario_env import MarioEnv
import cv2
import numpy as np
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from nes_py.wrappers import JoypadSpace
from inference import get_model

MAX_BOXES = 10
FPS = 60
SKIP = 4

env = MarioEnv()
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
        # processes frame into bounding boxes
        frame = env.render(mode='rgb_array')
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = img_model.infer(image)
        obs = process_results(results, MAX_BOXES)
        prev_obs = obs if prev_obs is None else prev_obs

        # save image to images
        for prediction in results[0].predictions:
            # Get bounding box coordinates
            x1 = int(prediction.x - prediction.width / 2)
            y1 = int(prediction.y - prediction.height / 2)
            x2 = int(prediction.x + prediction.width / 2)
            y2 = int(prediction.y + prediction.height / 2)

            # Get class label and confidence score
            label = prediction.class_id
            confidence = prediction.confidence

            # Draw bounding box and label on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{label}: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        images.append(image)
        
        # performs action on environment
        if i == 0:
            action, _states = model.predict([prev_obs, obs])
            prev_obs = obs

        _, _, done, _ = env.step(action.item())

        if done:
            break

    if done:
      break
env.close()

# write to mp4 file
height, width, _ = images[0].shape
vidwriter = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))
for image in images:
    vidwriter.write(image)
vidwriter.release()