# Implementation Using PyBullet

## Conda use conda
```
conda activate drones
```

## Dataset
```bash
python3 pretrain/dataset_generator.py
```

## Pretrain
run `pretrain/autoencoder.ipynb`


## RL Training
```bash
python3 learn.py
```

## To-Do
- [x] PyBullet Drone Environment
- [x] Asynchronous Actor-Critic PPO
- [x] Feature Encoder
- [x] Dataset Generator along trajectory with noise
- [ ] Hyperparameter Tuning
- [ ] Full PID Rate Controller

## Reference
- [Demonstrating Agile Flight from Pixels without State Estimation](https://rpg.ifi.uzh.ch/docs/RSS24_Geles.pdf)
- [Gym-Pybullet-Drones](https://github.com/utiasDSL/gym-pybullet-drones)
