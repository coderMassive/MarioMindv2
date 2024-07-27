from mario_env import MarioEnv
from roboflow_env import RoboflowEnvironment
from save_best_training import SaveOnBestTrainingRewardCallback
from nes_py.wrappers import JoypadSpace
from gymnasium.wrappers import FrameStack
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3 import PPO


log_dir = './tmp/'

def make_env():
    mario_env = MarioEnv(skip=8)
    mario_env = JoypadSpace(mario_env, SIMPLE_MOVEMENT)
    env = RoboflowEnvironment(mario_env, "mario-ibyfv/2", api_key="ITukAND4XqHSos8UA9me", max_boxes=10)
    env = FrameStack(env, 2)
    return env

env = DummyVecEnv([make_env for _ in range(2)])
env = VecMonitor(env, log_dir)

model = PPO('MlpPolicy', env, learning_rate=0.00003, verbose=1)
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir='tmp/')
model.learn(total_timesteps=500000, callback=callback)
model.save("./mario_model.zip")