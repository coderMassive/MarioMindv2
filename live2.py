from utils import process_results
import cv2
import time
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.smb_random_stages_env import SuperMarioBrosRandomStagesEnv
from stable_baselines3 import PPO
from nes_py.wrappers import JoypadSpace
from ultralytics import YOLO

MAX_BOXES = 10
FPS = 60
SKIP = 4
WINDOW_NAME = "Mario Detection"

env = SuperMarioBrosRandomStagesEnv()
env = JoypadSpace(env, SIMPLE_MOVEMENT)

model = PPO.load("./tmp/best_model.zip")
img_model = YOLO("best.onnx")

prev_obs = None
should_quit = False

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

env.reset()

while True:
    action = None
    done = False

    for i in range(SKIP):
        frame_start = time.time()

        if i == 0:
            frame = env.render(mode="rgb_array")
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            results = img_model(image, device="cpu")
            obs = process_results(results, MAX_BOXES)
            prev_obs = obs if prev_obs is None else prev_obs

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                label = results[0].names.get(cls_id, str(cls_id))

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    image,
                    f"{label}: {confidence:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow(WINDOW_NAME, image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                should_quit = True
                break

            action, _states = model.predict([prev_obs, obs])
            prev_obs = obs

        _, _, done, info = env.step(action.item())

        if done:
            prev_obs = None
            env.reset()
            break

        elapsed = time.time() - frame_start
        remaining = (1 / FPS) - elapsed
        if remaining > 0:
            time.sleep(remaining)

    if should_quit:
        break

cv2.destroyAllWindows()
env.close()
