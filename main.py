from env import QuadrotorEnv
from agent import ManualAgent
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from utils import linear_schedule
from constants import *

if __name__ == "__main__":
    # Create a vectorized training environment with several instances of the QuadrotorEnv.
    vec_env = DummyVecEnv([lambda: QuadrotorEnv() for _ in range(NUM_ENVS)])
    
    # Create a single evaluation environment for callbacks and testing.
    eval_env = Monitor(QuadrotorEnv())

    # Define the reward threshold at which training should stop.
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=REWARD_THRESHOLD, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./logs/best_model/',
        log_path='./logs/results/',
        eval_freq=1000,
        deterministic=True,
        render=False,
        callback_on_new_best=stop_callback  # Stop training if new best meets threshold.
    )

    # Initialize the PPO model using the vectorized environment.
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        device="cpu",
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=MINI_BATCH_SIZE,
        n_epochs=N_EPOCHS,
        policy_kwargs=dict(
            net_arch=NET_ARCH,
            activation_fn=lambda: torch.nn.LeakyReLU(negative_slope=0.2),
            optimizer_class=torch.optim.Adam,
        ),
    )

    # Train the model.
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, progress_bar=True)
    
    # Load the best saved model.
    model = PPO.load("./logs/best_model/best_model.zip", env=vec_env, device="cpu")

    input("Training Complete\nPress Enter to continue...")
    
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


## Manual control example, connect a controller
# if __name__ == "__main__":
#     env = QuadrotorEnv()
#     model = ManualAgent(env)
#     while True:
#         done = False
#         truncated = False
#         obs, _ = env.reset()
#         while True or not done and not truncated:
#             action, _ = model(obs)
#             obs, reward, done, truncated, info = env.step(action, dt = 0.001)
#             env.render()
#     env.close()
