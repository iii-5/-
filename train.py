import gymnasium as gym
from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
from torch import nn
import numpy as np
import os
import datetime
import json

import donkeycar as dk
from donkeycar.parts.dgym import DonkeyGymEnv

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[2]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),

            nn.Flatten(),
        )

        with th.no_grad():
            dummy_input = th.as_tensor(observation_space.sample()[None]).float().permute(0, 3, 1, 2)
            n_flatten = self.cnn(dummy_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations.permute(0, 3, 1, 2)))


class CustomDataLoggerCallback(EvalCallback):
    def __init__(self, eval_env: gym.Env, callback_on_eval_freq: int, log_path: str,
                 n_eval_episodes: int = 5, verbose: int = 1):
        super().__init__(eval_env, best_model_save_path=None, log_path=log_path,
                         eval_freq=callback_on_eval_freq, n_eval_episodes=n_eval_episodes,
                         verbose=verbose)
        self.custom_log_data = []
        self.log_file_path = os.path.join(log_path, 'custom_metrics.json')

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            avg_speed = []
            total_collisions = 0
            obs, info = self.eval_env.reset()
            done = False
            episode_speed = []
            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, trunc, info = self.eval_env.step(action)

                if isinstance(info, dict):
                    episode_speed.append(info.get('speed', 0))
                    if info.get('hit', 'none') != 'none':
                        total_collisions += 1
                elif isinstance(info, list):
                    for env_info in info:
                        episode_speed.append(env_info.get('speed', 0))
                        if env_info.get('hit', 'none') != 'none':
                            total_collisions += 1
                
                done = done or trunc
                
            avg_episode_speed = np.mean(episode_speed) if episode_speed else 0

            log_entry = {
                "timesteps": self.num_timesteps,
                "mean_reward": np.mean(self.episode_rewards),
                "std_reward": np.std(self.episode_rewards),
                "mean_ep_length": np.mean(self.episode_lengths),
                "avg_speed": float(avg_episode_speed),
                "total_collisions": total_collisions,
                "timestamp": str(datetime.datetime.now())
            }
            self.custom_log_data.append(log_entry)
            print(f"Custom Log - Timesteps: {log_entry['timesteps']}, Avg Speed: {log_entry['avg_speed']:.2f}, Collisions: {log_entry['total_collisions']}")

            if self.n_calls % (self.eval_freq * 5) == 0:
                with open(self.log_file_path, 'w') as f:
                    json.dump(self.custom_log_data, f, indent=4)
        return True

    def _on_training_end(self) -> None:
        with open(self.log_file_path, 'w') as f:
            json.dump(self.custom_log_data, f, indent=4)
        print(f"Custom training metrics saved to {self.log_file_path}")


if __name__ == "__main__":
    cfg = dk.load_config(myconfig='config.py')

    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)

    env = DonkeyGymEnv(cfg.DONKEY_SIM_PATH,
                       host=cfg.SIM_HOST,
                       env_name=cfg.GYM_ENV_NAME,
                       conf=cfg.GYM_CONF,
                       port=cfg.SIM_PORT)

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]
    )

    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs
    )

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=100.0, verbose=1)

    data_logger_callback = CustomDataLoggerCallback(env, callback_on_eval_freq=5000, log_path=log_dir, n_eval_episodes=3)

    callback = data_logger_callback

    print(f"Starting training for {cfg.TRAIN_RL_TOTAL_STEPS} timesteps...")
    model.learn(total_timesteps=cfg.TRAIN_RL_TOTAL_STEPS, callback=callback)

    model_save_path = os.path.join("models", f"donkey_rl_model_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.zip")
    model.save(model_save_path)
    print(f"Training finished. Model saved to {model_save_path}")

    env.close()