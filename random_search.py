import random
from hyperparams import *
from train import train

learning_rate = [1e-3, 1e-4]
mini_batch_size = [16, 32, 64, 128]
n_epochs = [4, 8, 16, 32]
n_steps = [1024, 2048, 4096, 8192]
policy_latent_dim = [64, 128, 256]


def random_search():
    for i in range(10):
        lr = random.choice(learning_rate)
        mbs = random.choice(mini_batch_size)
        ne = random.choice(n_epochs)
        ns = random.choice(n_steps)
        pld = random.choice(policy_latent_dim)

        print(f"Trial {i+1}:")
        print(f"Learning Rate: {lr}")
        print(f"Mini Batch Size: {mbs}")
        print(f"Number of Epochs: {ne}")
        print(f"Number of Steps: {ns}")
        print(f"Policy Latent Dimension: {pld}")

        global LEARNING_RATE
        global MINI_BATCH_SIZE
        global N_EPOCHS
        global N_STEPS
        global POLICY_LATENT_DIM
        global SAVE_PATH

        LEARNING_RATE = lr
        MINI_BATCH_SIZE = mbs
        N_EPOCHS = ne
        N_STEPS = ns
        POLICY_LATENT_DIM = pld
        SAVE_PATH = f"logs/random_search_{i+1}"

        # Save the hyperparameters
        with open(f"{SAVE_PATH}/hyperparams.txt", "w") as f:
            f.write(f"Learning Rate: {lr}\n")
            f.write(f"Mini Batch Size: {mbs}\n")
            f.write(f"Number of Epochs: {ne}\n")
            f.write(f"Number of Steps: {ns}\n")
            f.write(f"Policy Latent Dimension: {pld}\n")

        # Train the model
        train()