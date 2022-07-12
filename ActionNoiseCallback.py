from stable_baselines3.common.callbacks import EventCallback, BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np


class ActionNoiseCallback(BaseCallback):
    """
    Callback for linear decreasing action noise with:
    max(end_noise, start_noise - (start_noise - end_noise) / steps * current_step)

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param start_noise: start_noise
    :param end_noise: noise if linear decrease stops
    :param steps: steps from start to end noise
    :param verbose:
    """

    def __init__(
            self,
            start_noise,
            end_noise,
            steps,
            verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.evaluate = False
        self.noise_in_step = np.linspace(start_noise, end_noise, steps)
        self.best_reward = -np.inf

    def _on_step(self) -> bool:
        n_actions = self.model.action_space.shape[-1]
        if self.num_timesteps < self.noise_in_step.shape[0]:
            self.model.action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(self.noise_in_step[self.num_timesteps]) * np.ones(n_actions))
        else:
            self.model.action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(self.noise_in_step[-1]) * np.ones(n_actions))
        return True
