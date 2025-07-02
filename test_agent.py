import gym
import importlib
import os
from stable_baselines3 import PPO
from PIL import Image

# ==== 參數設定 ====
MODEL_PATH = 'checkpoints/rl_model_60000_steps.zip'  # 你的 checkpoint 路徑
FRAME_DIR = 'test_frames'                            # 圖片存放資料夾
ENV_NAME = 'donkey-generated-track-v0'

# ==== 載入 gym-donkeycar config（可根據你的 config.py 調整） ====
import donkeycar as dk
cfg = dk.load_config()

# ==== 建立環境 ====
gym_donkeycar = importlib.import_module('gym_donkeycar')
env = gym.make(ENV_NAME, conf=cfg.GYM_CONF, sim_path=cfg.DONKEY_SIM_PATH)

# ==== 載入模型 ====
model = PPO.load(MODEL_PATH, env=env)

# ==== 跑一個測試回合並存圖 ====
os.makedirs(FRAME_DIR, exist_ok=True)
obs = env.reset()
done = False
total_reward = 0
step = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    # 存圖片
    img = Image.fromarray(obs)
    img.save(f"{FRAME_DIR}/frame_{step:05d}.jpg")
    step += 1

print(f"測試回合總 reward: {total_reward:.2f}，總步數: {step}")
print(f"所有畫面已存於 {FRAME_DIR}/ 目錄下。")
print("你可以用以下指令合成影片：")
print(f"ffmpeg -framerate 15 -i {FRAME_DIR}/frame_%05d.jpg -c:v libx264 -pix_fmt yuv420p test_run.mp4")

env.close() 