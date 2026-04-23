from utils import process_results
import cv2
import time
import numpy as np
import torch
import shap
import torch.nn as nn

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.smb_random_stages_env import SuperMarioBrosRandomStagesEnv
from stable_baselines3 import PPO
from nes_py.wrappers import JoypadSpace
from ultralytics import YOLO

MAX_BOXES = 10
FPS = 60
SKIP = 4
TOP_K = 5

BACKGROUND_SAMPLES = 32
SHAP_EVERY = 6
SMOOTHING = 0.85

# Heatmap settings
HEATMAP_ALPHA = 0.5
BLUR_SIZE = 41
MIN_DISPLAY_THRESHOLD = 0.1

env = SuperMarioBrosRandomStagesEnv()
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# SHAP is usually more stable on CPU
model = PPO.load("./tmp/best_model.zip", device="cpu")
img_model = YOLO("best.onnx", task="detect")

device = next(model.policy.parameters()).device

prev_obs = None
should_quit = False
last_detection_frame = None

smoothed_importance = None
cached_importance = None
shap_counter = 0

cv2.namedWindow("Smooth", cv2.WINDOW_NORMAL)
cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)


class PolicyProbabilityWrapper(nn.Module):
    def __init__(self, sb3_model):
        super().__init__()
        self.policy = sb3_model.policy
        self.device = next(self.policy.parameters()).device

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.to(self.device, dtype=torch.float32)

        features = self.policy.extract_features(x)
        latent_pi, _ = self.policy.mlp_extractor(features)
        dist = self.policy._get_action_dist_from_latent(latent_pi)
        return dist.distribution.probs


def get_processed_mapping(results, max_boxes):
    """
    Match the same ordering used by process_results:
    1) sort by confidence descending
    2) keep top max_boxes
    3) sort by distance to Mario if Mario exists
    """
    detections = []

    result = results[0]
    if result.boxes is None:
        return []

    boxes_xywh = result.boxes.xywh.cpu().numpy()
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy()

    for i in range(len(boxes_xywh)):
        x, y, w, h = boxes_xywh[i]
        x1, y1, x2, y2 = boxes_xyxy[i]
        conf = float(confs[i])
        cls = int(clss[i])
        label = result.names.get(cls, str(cls))

        detections.append({
            "xywh": [float(x), float(y), float(w), float(h)],
            "xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "conf": conf,
            "cls": cls,
            "label": label,
        })

    detections = sorted(detections, key=lambda d: d["conf"], reverse=True)
    detections = detections[:max_boxes]

    mario = next((d for d in detections if d["cls"] == 8), None)

    if mario is not None:
        mx, my = mario["xywh"][0], mario["xywh"][1]

        def dist(d):
            dx = d["xywh"][0] - mx
            dy = d["xywh"][1] - my
            return dx * dx + dy * dy

        detections = sorted(detections, key=dist)

    return detections


def build_background_dataset(env, img_model, num_samples):
    background = []
    prev = None

    env.reset()

    while len(background) < num_samples:
        frame = env.render(mode="rgb_array")
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results = img_model(image, device="cpu", verbose=False)
        obs = process_results(results, MAX_BOXES)

        if prev is None:
            prev = obs.copy()

        stacked = np.array([prev, obs], dtype=np.float32)
        background.append(stacked)

        action, _ = model.predict([prev, obs])
        _, _, done, _ = env.step(action.item())

        prev = obs.copy()

        if done:
            prev = None
            env.reset()

    return np.array(background, dtype=np.float32)


def make_explainer(policy_model, background_np):
    background_tensor = torch.tensor(
        background_np,
        dtype=torch.float32,
        device=policy_model.device
    )
    return shap.GradientExplainer(policy_model, background_tensor)


def compute_shap_magnitude(explainer, policy_model, prev_obs, obs):
    """
    Returns magnitude-only per-box SHAP scores.
    This version handles multiple possible SHAP output shapes safely.
    """
    stacked = np.array([prev_obs, obs], dtype=np.float32)

    x = torch.tensor(
        stacked,
        dtype=torch.float32,
        device=policy_model.device
    ).unsqueeze(0)

    shap_values = explainer.shap_values(x)

    # SHAP may return:
    # 1) list of arrays, one per output class
    # 2) a single ndarray/tensor with output dim included
    if isinstance(shap_values, list):
        shap_arr = np.stack([
            sv.detach().cpu().numpy() if torch.is_tensor(sv) else np.array(sv)
            for sv in shap_values
        ], axis=0)
        # magnitude-only: average absolute attribution over outputs
        shap_arr = np.mean(np.abs(shap_arr), axis=0)
    else:
        if torch.is_tensor(shap_values):
            shap_arr = shap_values.detach().cpu().numpy()
        else:
            shap_arr = np.array(shap_values)

        # Example possible shape: [1, 2, MAX_BOXES, 6, num_actions]
        if shap_arr.ndim >= 5:
            shap_arr = np.mean(np.abs(shap_arr), axis=-1)

    shap_arr = np.squeeze(shap_arr)

    # Expected final shape: [2, MAX_BOXES, 6]
    if shap_arr.ndim != 3:
        print("Unexpected SHAP shape after squeeze:", shap_arr.shape)
        return np.zeros(MAX_BOXES, dtype=np.float32)

    if shap_arr.shape[0] != 2:
        print("Unexpected frame dimension in SHAP:", shap_arr.shape)
        return np.zeros(MAX_BOXES, dtype=np.float32)

    current_frame = shap_arr[1]

    # Magnitude only
    box_scores = np.sum(np.abs(current_frame), axis=1).astype(np.float32)

    return box_scores


def add_heatmap(image, mapping, importance):
    h, w = image.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)

    if len(importance) == 0:
        return image

    max_val = float(np.max(importance))
    if max_val < 1e-12:
        return image

    ranked = np.argsort(-importance)[:TOP_K]

    for idx, det in enumerate(mapping):
        if idx >= len(importance):
            continue
        if idx not in ranked:
            continue

        score = float(importance[idx])
        norm = score / (max_val + 1e-12)

        if norm < MIN_DISPLAY_THRESHOLD:
            continue

        x1, y1, x2, y2 = det["xyxy"]

        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        heat[y1:y2, x1:x2] += norm

    heat = cv2.GaussianBlur(heat, (BLUR_SIZE, BLUR_SIZE), 0)

    peak = float(np.max(heat))
    if peak > 1e-12:
        heat /= peak

    overlay = np.zeros_like(image, dtype=np.float32)
    overlay[:, :, 2] = heat * 255.0  # red only

    alpha = (heat * HEATMAP_ALPHA)[..., None]
    blended = image.astype(np.float32) * (1.0 - alpha) + overlay * alpha

    return np.clip(blended, 0, 255).astype(np.uint8)


policy_model = PolicyProbabilityWrapper(model).to(device)
background = build_background_dataset(env, img_model, BACKGROUND_SAMPLES)
explainer = make_explainer(policy_model, background)

env.reset()

while True:
    action = None
    done = False

    for i in range(SKIP):
        frame_start = time.time()

        frame = env.render(mode="rgb_array")
        smooth = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if i == 0:
            image = smooth.copy()

            results = img_model(image, device="cpu", verbose=False)
            obs = process_results(results, MAX_BOXES)

            if prev_obs is None:
                prev_obs = obs.copy()

            if cached_importance is None or shap_counter <= 0:
                try:
                    importance = compute_shap_magnitude(
                        explainer,
                        policy_model,
                        prev_obs,
                        obs
                    )
                    cached_importance = importance.copy()
                    shap_counter = SHAP_EVERY
                except Exception as e:
                    print("SHAP error:", e)
                    importance = np.zeros(MAX_BOXES, dtype=np.float32)
                    cached_importance = importance.copy()
                    shap_counter = SHAP_EVERY
            else:
                importance = cached_importance.copy()
                shap_counter -= 1

            if smoothed_importance is None:
                smoothed_importance = importance.copy()
            else:
                smoothed_importance = (
                    SMOOTHING * smoothed_importance
                    + (1.0 - SMOOTHING) * importance
                )

            importance = smoothed_importance.copy()
            mapping = get_processed_mapping(results, MAX_BOXES)

            image = add_heatmap(image, mapping, importance)
            last_detection_frame = image

            action, _ = model.predict([prev_obs, obs])
            prev_obs = obs.copy()

        if last_detection_frame is not None:
            cv2.imshow("Detections", last_detection_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            should_quit = True
            break

        if action is None:
            action, _ = model.predict([prev_obs, prev_obs])

        _, _, done, _ = env.step(action.item())

        if done:
            prev_obs = None
            smoothed_importance = None
            cached_importance = None
            shap_counter = 0
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
