import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_device
from gymnasium import spaces
import numpy as np

# # Feature Encoder
# class FeatureEncoder(nn.Module):
#     def __init__(self):
#         super(FeatureEncoder, self).__init__()
#         self.cnn_enc = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
        
#         self.fc_enc = nn.Sequential(
#             nn.Linear(8 * 16 * 16, 1024),  # Adjusted based on input (3, 64, 64)
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU()
#         )

#     def forward(self, x):
#         x = self.cnn_enc(x)
#         x = x.view(x.size(0), -1)  # Flatten
#         x = self.fc_enc(x)
#         return x

# # Custom Feature Extractor
# class FeatureExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space: spaces.Tuple, features_dim: int = 256):
#         super().__init__(observation_space, features_dim)
        
#         # Load the feature encoder
#         self.feature_encoder = FeatureEncoder()
#         self.feature_encoder.load_state_dict(torch.load("weights/feature_encoder.pth"))
#         self.feature_encoder.eval()
        
#         # Ensure model is on the right device
#         self.device = get_device("auto")
#         self.feature_encoder.to(self.device)

#         # Compute feature size (256 from encoder + 12 from last_3_actions + 15 from privileged state)
#         self.features_dim = features_dim + 12 + 15

#     def forward(self, observations):
#         """
#         observations: (segmentation, last_3_actions, privileged_state)
#         """
#         seg, last_3_actions, privileged_state = observations

#         # Ensure tensor is on the correct device
#         seg = torch.permute(seg, (0, 3, 1, 2)).float().to(self.device)  # Convert to (B, C, H, W)
#         last_3_actions = last_3_actions.float().to(self.device)
#         privileged_state = privileged_state.float().to(self.device)

#         # Extract image features
#         encoding = self.feature_encoder(seg)  # Shape: (B, 256)

#         # Concatenate all inputs
#         return torch.cat([encoding, last_3_actions, privileged_state], dim=-1)

# # Custom Actor-Critic Policy
# class RacingPolicy(ActorCriticPolicy):
#     def __init__(self,
#                  observation_space,
#                  action_space,
#                  lr_schedule,
#                  **kwargs):
#         super().__init__(
#             observation_space,
#             action_space,
#             lr_schedule,
#             features_extractor_class=FeatureExtractor,
#             features_extractor_kwargs={"features_dim": 256},
#             **kwargs
#         )

#     def _build_mlp_extractor(self):
#         """
#         Build separate MLPs for the actor and critic.
#         """
#         # Actor network
#         self.actor_mlp = nn.Sequential(
#             nn.Linear(256 + 12, 256),  # Encoding (256) + last_3_actions (12)
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, self.action_space.shape[0])
#         )

#         # Critic network
#         self.critic_mlp = nn.Sequential(
#             nn.Linear(256 + 12 + 15, 256),  # Encoding (256) + last_3_actions (12) + privileged_state (15)
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )

#     def forward(self, obs):
#         """
#         Forward pass.
#         """
#         features = self.extract_features(obs)  # Shape: (B, 256 + 12 + 15)

#         # Split features for actor and critic
#         encoding = features[:, :256]  # First 256 dims (CNN features)
#         last_3_actions = features[:, 256:256+12]  # Next 12 dims (last actions)
#         privileged_state = features[:, 256+12:]  # Remaining 15 dims (privileged state)

#         # Actor gets encoding + last_3_actions
#         actor_input = torch.cat([encoding, last_3_actions], dim=-1)

#         # Critic gets encoding + last_3_actions + privileged_state
#         critic_input = torch.cat([encoding, last_3_actions, privileged_state], dim=-1)

#         return self.actor_mlp(actor_input), self.critic_mlp(critic_input)


#     def critic(self, obs):
#         """
#         Compute value function with privileged state.
#         """
#         features = self.extract_features(obs)
#         return self.critic_mlp(features)

# # Define PPO policy
# policy = RacingPolicy(
#     observation_space=spaces.Tuple(
#         (spaces.Box(low=0, high=1, shape=(64, 64, 3), dtype=np.float32),  # Segmentation image
#          spaces.Box(low=-1, high=1, shape=(3, 4), dtype=np.float32),  # Last 3 actions
#          spaces.Box(low=-1, high=1, shape=(15,), dtype=np.float32))  # Privileged state
#     ),
#     action_space=spaces.Box(
#         low=np.array([0.0, -1.0, -1.0, -1.0]),
#         high=np.array([1.0, 1.0, 1.0, 1.0]),
#     ),
#     lr_schedule=0.001
# )






















# Feature Encoder
class FeatureEncoder(nn.Module):
    def __init__(self):
        super(FeatureEncoder, self).__init__()
        self.cnn_enc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc_enc = nn.Sequential(
            nn.Linear(8 * 16 * 16, 256),  # Adjusted based on input (3, 64, 64)
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

    def forward(self, x):
        # duplicate to 3 channels
        # x = torch.cat((x, x, x), 1)
        x = self.cnn_enc(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_enc(x)
        return x

# Custom Feature Extractor
class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Tuple, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Load the feature encoder
        self.feature_encoder = FeatureEncoder()
        # self.feature_encoder.load_state_dict(torch.load("weights/feature_encoder.pth"))
        # self.feature_encoder.eval()
        
        # Ensure model is on the right device
        self.device = get_device("cpu")
        self.feature_encoder.to(self.device)

    def forward(self, observations):
        """
        observations: segmentation
        """
        seg = observations

        # Ensure tensor is on the correct device
        seg = torch.permute(seg, (0, 1, 2, 3)).float().to(self.device)  # Convert to (B, C, H, W)

        # Extract image features
        with torch.no_grad():
            encoding = self.feature_encoder(seg)

        return encoding
