import random
from hyperparams import *
from train import train
import time

learning_rate = [1e-3, 1e-4]
mini_batch_size = [16, 32, 64, 128]
n_epochs = [4, 8, 16, 32]
n_steps = [1024, 2048, 4096, 8192]
policy_latent_dim = [64, 128, 256]


def random_search():
    this_run = time.strftime('%Y_%m_%d_%H_%M_%S')
    for i in range(10):
        lr = random.choice(learning_rate)
        mbs = random.choice(mini_batch_size)
        ne = random.choice(n_epochs)
        ns = random.choice(n_steps)
        pld = random.choice(policy_latent_dim)
        log_dir = f"logs/random_search_{this_run}/{i+1}"
        
        print(f"Trial {i+1}:")
        print(f"Learning Rate: {lr}")
        print(f"Mini Batch Size: {mbs}")
        print(f"Number of Epochs: {ne}")
        print(f"Number of Steps: {ns}")
        print(f"Policy Latent Dimension: {pld}")

        # Train the model
        train(log_dir, lr, mbs, ne, ns, pld)
        print("\n\n\n")


if __name__ == "__main__":
    random_search()
