#!/usr/bin/env python3
"""
Scripts to train a keras model using tensorflow.
Basic usage should feel familiar: train.py --tubs data/ --model models/mypilot.h5

Usage:
    train.py [--tubs=tubs] (--model=<model>)
    [--type=(linear|inferred|tensorrt_linear|tflite_linear)]
    [--comment=<comment>]

Options:ㄧ
    -h --help              Show this screen.
"""

print("=== DEBUG: train.py 啟動 ===")
from docopt import docopt
import donkeycar as dk
from donkeycar.pipeline.training import train
import importlib
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
import os

# 強化學習相關 import
try:
    import gym
    import stable_baselines3 as sb3
    from stable_baselines3.common.env_util import make_vec_env
except ImportError:
    pass  # 如果沒裝，RL_MODE 會自動失敗

class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_rewards = 0.0

    def _on_step(self) -> bool:
        # SB3 會在每個 step 呼叫這個
        rewards = self.locals.get('rewards')
        dones = self.locals.get('dones')
        if rewards is not None and dones is not None:
            for reward, done in zip(rewards, dones):
                self.current_rewards += reward
                if done:
                    self.episode_rewards.append(self.current_rewards)
                    self.current_rewards = 0.0
        return True

def train_rl(cfg):
    import gym
    import stable_baselines3 as sb3
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import CheckpointCallback
    import importlib
    import os
    import matplotlib.pyplot as plt

    env_name = getattr(cfg, 'RL_ENV_NAME', 'donkey-generated-track-v0')
    total_timesteps = getattr(cfg, 'RL_TOTAL_TIMESTEPS', 100_000)
    algo = getattr(cfg, 'RL_ALGO', 'PPO')
    model_path = getattr(cfg, 'RL_MODEL_PATH', 'models/rl_model.zip')
    checkpoint_dir = getattr(cfg, 'RL_CHECKPOINT_DIR', 'checkpoints')
    checkpoint_freq = getattr(cfg, 'RL_CHECKPOINT_FREQ', 10_000)

    # 檢查 gym-donkeycar 是否存在
    try:
        gym_donkeycar = importlib.import_module('gym_donkeycar')
    except ImportError:
        print('請先安裝 gym-donkeycar: pip install gym-donkeycar')
        return

    # 建立 checkpoint 目錄
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 建立環境
    env = gym.make(env_name, conf=cfg.GYM_CONF, sim_path=cfg.DONKEY_SIM_PATH)
    env = make_vec_env(lambda: env, n_envs=1)

    # 選擇演算法
    algo_class = getattr(sb3, algo, None)
    if algo_class is None:
        print(f'不支援的 RL 演算法: {algo}')
        return

    # 嘗試載入現有模型前 debug print
    print("DEBUG: model_path =", model_path)
    print("DEBUG: 絕對路徑 =", os.path.abspath(model_path))
    print("DEBUG: 檔案是否存在？", os.path.exists(model_path))
    if os.path.exists(model_path):
        print(f"載入現有模型: {model_path}")
        model = algo_class.load(model_path, env=env, tensorboard_log="./sb3_tensorboard/")
    else:
        print("建立新模型")
        model = algo_class('CnnPolicy', env, verbose=1, tensorboard_log="./sb3_tensorboard/")

    # 設定 checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix='rl_model'
    )
    # 設定 reward logger callback
    reward_logger = RewardLoggerCallback()

    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, reward_logger])
    model.save(model_path)
    print(f'RL 訓練完成，模型已儲存到 {model_path}')

    # 訓練結束後畫 reward 曲線
    if len(reward_logger.episode_rewards) > 0:
        plt.figure()
        plt.plot(reward_logger.episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode Reward Curve')
        plt.savefig('reward_curve.png')
        plt.close()
        print('reward_curve.png 已儲存！')

def main():
    import donkeycar as dk
    cfg = dk.load_config()
    print("config file path:", cfg.__file__ if hasattr(cfg, '__file__') else "unknown")
    print("RL_MODE =", getattr(cfg, 'RL_MODE', False))
    if getattr(cfg, 'RL_MODE', False):
        print("=== DEBUG: 進入 RL 訓練 ===")
        train_rl(cfg)
        return
    # 只有不是 RL_MODE 才跑 docopt
    args = docopt(__doc__)
    print("=== DEBUG: args =", args)
    tubs = args['--tubs']
    model = args['--model']
    model_type = args['--type']
    comment = args['--comment']
    train(cfg, tubs, model, model_type, comment)

if __name__ == "__main__":
    main()