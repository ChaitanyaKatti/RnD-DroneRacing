from racing_env import RacingEnv
from racing_policy import FeatureEncoder, FeatureExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import numpy as np
from constants import *
import torch

# Create the environment
env = RacingEnv(gui=False)
eval_env = Monitor(env)

# Define the reward threshold at which training should stop.
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=REWARD_THRESHOLD, verbose=1)
eval_callback = EvalCallback(
    env,
    best_model_save_path='./logs/best_model/',
    log_path='./logs/results/',
    eval_freq=4096,
    deterministic=True,
    render=False,
    callback_on_new_best=stop_callback  # Stop training if new best meets threshold.
)

# Create the model
model = PPO(
    "MlpPolicy",
    env,
    device="cpu",
    learning_rate=LEARNING_RATE,
    n_steps=N_STEPS,
    batch_size=MINI_BATCH_SIZE,
    n_epochs=N_EPOCHS,
    policy_kwargs=dict(
        net_arch=NET_ARCH,
        # features_extractor_class=FeatureExtractor,
        # activation_fn=lambda: torch.nn.LeakyReLU(negative_slope=0.2),
        # optimizer_class=torch.optim.Adam,
    ),
    verbose=1,
)

# Train the model
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, progress_bar=True)

 # Evaluate the trained model
for episode in range(10):
    obs, _ = eval_env.reset()
    done = False
    truncated = False
    total_reward = 0
    while not done and not truncated:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = eval_env.step(action)
        total_reward += reward
        eval_env.render()
    print(f"Episode {episode} Reward: {total_reward:.4f} Info: {info.get('done_reason', 'N/A')}")
eval_env.close()