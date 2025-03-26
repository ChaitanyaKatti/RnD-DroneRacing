from env.hover import HoverEnv
from policy import AsymmetricActorCritic
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from hyperparams import *
import time
import os

def train(log_dir, learning_rate, mini_batch_size, n_epochs, n_steps, policy_latent_dim):
    # Save the hyperparameters
    os.makedirs(log_dir, exist_ok=True)
    with open(f"{log_dir}/hyperparams.txt", "w") as f:
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Mini Batch Size: {mini_batch_size}\n")
        f.write(f"Number of Epochs: {n_epochs}\n")
        f.write(f"Number of Steps: {n_steps}\n")
        f.write(f"Policy Latent Dimension: {policy_latent_dim}\n")
        f.write(f"Total Timesteps: {TOTAL_TIMESTEPS}\n")
        f.write(f"Reward Threshold: {REWARD_THRESHOLD}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Create the environment
    env = HoverEnv(gui=False)
    eval_env = Monitor(env)

    # Define the reward threshold at which training should stop.
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=REWARD_THRESHOLD, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
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
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=mini_batch_size,
        n_epochs=n_epochs,
        verbose=1,
    )

    # Train the model
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, progress_bar=True)

    # Load the best model
    best_model_path = os.path.join(log_dir, 'best_model.zip')
    if os.path.exists(best_model_path):
        model = PPO.load(best_model_path)
        print(f"Loaded best model from {best_model_path}")

        # Evaluate the model
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)

        # Save evaluation results to a text file
        with open(os.path.join(log_dir, 'evaluation_results.txt'), 'w') as f:
            f.write(f"Mean reward: {mean_reward:.4f}\n")
            f.write(f"Standard deviation: {std_reward:.4f}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"Evaluation completed - Mean reward: {mean_reward:.4f} ± {std_reward:.4f}")
    else:
        print("No best model found to evaluate")

    env.close()

def test(log_dir, gui=False):
    env = HoverEnv(gui=gui)
    model = PPO(
        AsymmetricActorCritic,
        env,
        device="cpu",
    )
    model.load(os.path.join(log_dir, 'best_model.zip'))

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
            if gui:
                env.render()
                time.sleep(0.02)
        print(f"Episode {episode} Reward: {total_reward:.4f} Info: {info.get('done_reason', 'N/A')}")
    env.close()

if __name__ == "__main__":
    train(
        log_dir=LOG_DIR,
        learning_rate=LEARNING_RATE,
        mini_batch_size=MINI_BATCH_SIZE,
        n_epochs=N_EPOCHS,
        n_steps=N_STEPS,
        policy_latent_dim=POLICY_LATENT_DIM,
    )

    test(log_dir=LOG_DIR, gui=True)
