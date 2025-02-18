import pygame
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy

class Agent():
    def __init__(self, env):
        pass

    def __call__(self, state):
        pass


class ManualAgent(Agent):
    def __init__(self, env):
        super().__init__(env)
        pygame.init()
        self.env = env
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

    def __call__(self, state):
        pygame.event.get()
        thrust = 0.5 * (1 + self.joystick.get_axis(2))
        target_pitch = self.joystick.get_axis(1) # Pitch back is positive
        target_roll = self.joystick.get_axis(3) # Roll right is positive
        target_yaw = -self.joystick.get_axis(0) # Yaw left is positive
        target_rate = np.array([target_roll, target_pitch, target_yaw])
        
        # Deadzone
        if abs(target_pitch) < 0.05:
            target_pitch = 0.0
        if abs(target_roll) < 0.05:
            target_roll = 0.0
        if abs(target_yaw) < 0.05:
            target_yaw = 0.0
        
        # print(f"Thrust: {thrust:.3f}, Roll Rate: {target_roll:.3f}, Pitch Rate: {target_pitch:.3f}, Yaw Rate: {target_yaw:.3f}")
        return np.array([thrust, target_roll, target_pitch, target_yaw]), {}
