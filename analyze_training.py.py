import json
import pandas as pd
import matplotlib.pyplot as plt
import os

LOG_DIR = r'C:\Users\ioyt8\myvirtualcar\logs\20250701-133000'

CUSTOM_METRICS_FILE = os.path.join(LOG_DIR, 'custom_metrics.json')
TENSORBOARD_EVENTS_DIR = LOG_DIR

def plot_custom_metrics(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    with open(file_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 6))
    plt.plot(df['timesteps'], df['mean_reward'], label='Mean Episode Reward')
    plt.fill_between(df['timesteps'], df['mean_reward'] - df['std_reward'],
                     df['mean_reward'] + df['std_reward'], alpha=0.2, label='Std Dev of Reward')
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.title('Mean Episode Reward during Training')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, 'mean_reward.png'))
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(df['timesteps'], df['avg_speed'], label='Average Speed (m/s)')
    plt.xlabel('Timesteps')
    plt.ylabel('Speed (m/s)')
    plt.title('Average Speed during Training')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, 'avg_speed.png'))
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(df['timesteps'], df['total_collisions'], label='Total Collisions per Eval Period')
    plt.xlabel('Timesteps')
    plt.ylabel('Collisions')
    plt.title('Collisions during Training')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, 'total_collisions.png'))
    plt.show()

if __name__ == "__main__":
    print(f"Analyzing data from: {CUSTOM_METRICS_FILE}")
    plot_custom_metrics(CUSTOM_METRICS_FILE)

    print("\n----------------------------------------------------------")
    print("For more detailed logs (e.g., policy loss, value loss),")
    print(f"run TensorBoard from your Anaconda Prompt in the {LOG_DIR} directory:")
    print(f"cd {os.path.abspath(os.path.join(LOG_DIR, '..'))}")
    print(f"tensorboard --logdir {os.path.abspath(LOG_DIR)}")
    print("Then open your browser to http://localhost:6006")
    print("----------------------------------------------------------")