from stable_baselines3.td3.policies import TD3Policy, register_policy, Actor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import BaseModel
import gym
from typing import Any, Dict, List, Optional, Tuple, Type
from torch import nn
from stable_baselines3.common.preprocessing import get_action_dim
import torch as th


class CustomActor(Actor):
    """
    Actor network (policy) for TD3.
    """
    def __init__(self, *args, **kwargs):
        layers = kwargs.pop("layers")
        layer_width = kwargs.pop("layer_width")
        activation_fun = kwargs.pop("activation_fun")
        end_activation_fun = kwargs.pop("end_activation_fun")
        super(CustomActor, self).__init__(*args, **kwargs)
        # Define custom network with Dropout
        # WARNING: it must end with a tanh activation to squash the output
        action_dim = get_action_dim(self.action_space)

        net_dict = []
        if layers > 0:
            net_dict.append(nn.Linear(self.features_dim, layer_width, bias=False))
            net_dict.append(activation_fun)
            for _ in range(layers-1):
                net_dict.append(nn.Linear(layer_width, layer_width, bias=False))
                net_dict.append(activation_fun)
            net_dict.append(nn.Linear(layer_width, action_dim, bias=False))
            net_dict.append(end_activation_fun)
        else:
            net_dict.append(nn.Linear(self.features_dim, action_dim, bias=False))
            net_dict.append(end_activation_fun)

        self.mu = nn.Sequential(*net_dict)
        print("Actor Net")
        print(self.mu)


class CustomContinuousCritic(BaseModel):
    """
    Critic network(s) for DDPG/SAC/TD3.
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        critic_net_props: Dict = None
    ):

        layers = critic_net_props["layers"]
        layer_width = critic_net_props["layer_width"]
        activation_fun = critic_net_props["activation_fun"]
        bias = critic_net_props["bias"]

        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            # q_net = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
            # Define critic with Dropout here
            action_dim = get_action_dim(self.action_space)

            net_dict = []
            if layers > 0:
                net_dict.append(nn.Linear(features_dim + action_dim, layer_width, bias=bias))
                net_dict.append(activation_fun)
                for _ in range(layers - 1):
                    net_dict.append(nn.Linear(layer_width, layer_width, bias=bias))
                    net_dict.append(activation_fun)
                net_dict.append(nn.Linear(layer_width, 1, bias=bias))
            else:
                net_dict.append(nn.Linear(features_dim + action_dim, 1, bias=bias))

            q_net = nn.Sequential(*net_dict)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)
        print(self.q_networks[0])

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)
        qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features = self.extract_features(obs)
        return self.q_networks[0](th.cat([features, actions], dim=1))


class CustomTD3Policy(TD3Policy):
    def __init__(self, *args, **kwargs):
        actor_layers = kwargs.pop("actor_layers", 0)
        actor_layer_width = kwargs.pop("actor_layer_width", 5)
        actor_activation_fun = kwargs.pop("actor_activation_fun", nn.Tanh())
        actor_end_activation_fun = kwargs.pop("actor_end_activation_fun", nn.Hardtanh())

        self.custom_actor_kwargs = {"layers": actor_layers,
                                    "layer_width": actor_layer_width,
                                    "activation_fun": actor_activation_fun,
                                    "end_activation_fun": actor_end_activation_fun}
        
        critic_layers = kwargs.pop("critic_layers", 2)
        critic_layer_width = kwargs.pop("critic_layer_width", 200)
        critic_activation_fun = kwargs.pop("critic_activation_fun", nn.Tanh())
        critic_bias = kwargs.pop("critic_bias", True)

        self.custom_critic_kwargs = {"layers": critic_layers,
                                     "layer_width": critic_layer_width,
                                     "activation_fun": critic_activation_fun,
                                     "bias": critic_bias}
        super(CustomTD3Policy, self).__init__(*args, **kwargs)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor_kwargs = {**actor_kwargs, **self.custom_actor_kwargs}
        return CustomActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        critic_kwargs["critic_net_props"] = self.custom_critic_kwargs
        return CustomContinuousCritic(**critic_kwargs).to(self.device)


register_policy("CustomTD3Policy", CustomTD3Policy)