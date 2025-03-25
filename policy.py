from typing import Callable
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from hyperparams import *

# Feature Encoder
class FeatureEncoder(nn.Module):
    def __init__(self):
        super(FeatureEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),   #(1, 84, 84) -> (16, 84, 84)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  #(16, 84, 84) -> (16, 42, 42)
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=0),   #(16, 42, 42) -> (8, 40, 40)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  #(8, 40, 40) -> (8, 20, 20)
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1),    #(8, 20, 20) -> (8, 20, 20)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                   #(8, 20, 20) -> (8, 10, 10)
        )

        self.fc_enc = nn.Sequential(
            nn.Linear(4*10*10, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc_enc(x)
        return x


# Custom Feature Extractor
class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Tuple, features_dim: int = 256):
        super().__init__(observation_space, features_dim=features_dim)

        # Load the feature encoder
        self.feature_encoder = FeatureEncoder()
        self.feature_encoder.load_state_dict(torch.load("pretrain/weights/encoder.pth"))
        self.feature_encoder.eval()

    def forward(self, observations) -> torch.Tensor:
        batch_size = observations['img'].size(0)
        seg = observations['img']
        last_3_actions = observations['last_3_actions'].view(batch_size, -1)
        privileged_state = observations['kinematics'].view(batch_size, -1)

        with torch.no_grad(): # Don't train the encoder
            encoding = self.feature_encoder(seg)

        return torch.cat([encoding, last_3_actions, privileged_state], dim=-1)


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()

        self.latent_dim_pi = POLICY_LATENT_DIM
        self.latent_dim_vf = POLICY_LATENT_DIM

        self.policy_net = nn.Sequential(
            nn.Linear(256 + 12, self.latent_dim_pi),  # Encoding (256) + last_3_actions (12)
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.latent_dim_pi, self.latent_dim_pi),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.value_net = nn.Sequential(
            nn.Linear(
                256 + 12 + 15, self.latent_dim_vf
            ),  # Encoding (256) + last_3_actions (12) + privileged_state (15)
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.latent_dim_vf, self.latent_dim_vf),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        return self.policy_net(features[:, : 256 + 12])

    def forward_critic(self, features):
        return self.value_net(features)


# Custom Actor-Critic Policy
class AsymmetricActorCritic(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=FeatureExtractor,
            features_extractor_kwargs={"features_dim": 256},
            **kwargs,
        )

    def _build_mlp_extractor(self):
        self.mlp_extractor = PolicyNetwork()

if __name__ == "__main__":
    from torchsummary import summary

    # Test the policy network
    policy = PolicyNetwork().to("cuda")

    print("Policy Network")
    summary(policy.policy_net, (256 + 12,))

    print("value Network")
    summary(policy.value_net, (256 + 12 + 15,))