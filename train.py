from env.hover import HoverEnv
from policy import AsymmetricActorCritic
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from hyperparams import *
import time
import os

def train():
    # Create the environment
    env = HoverEnv(gui=False)
    eval_env = Monitor(env)

    # Define the reward threshold at which training should stop.
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=REWARD_THRESHOLD, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(SAVE_PATH, 'best_model'),
        log_path=os.path.join(SAVE_PATH, 'eval_logs'),
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=False,
        callback_on_new_best=stop_callback  # Stop training if new best meets threshold.
    )

    # Create the model
    model = PPO(
        AsymmetricActorCritic,
        env,
        device="cpu",
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=MINI_BATCH_SIZE,
        n_epochs=N_EPOCHS,
        verbose=1,
    )

    # Train the model
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, progress_bar=True)
    env.close()

def test():
    env = HoverEnv(gui=True)
    model = PPO(
        AsymmetricActorCritic,
        env,
        device="cpu",
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=MINI_BATCH_SIZE,
        n_epochs=N_EPOCHS,
        verbose=1,
    )
    model.load(os.path.join(SAVE_PATH, 'best_model', 'best_model.zip'))

    # Evaluate the trained model
    for episode in range(10):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        while not done and not truncated:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            env.render()
            time.sleep(0.05)
        print(f"Episode {episode} Reward: {total_reward:.4f} Info: {info.get('done_reason', 'N/A')}")
    env.close()

if __name__ == "__main__":
    train()
    test()