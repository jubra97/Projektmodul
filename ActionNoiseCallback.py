from stable_baselines3.common.callbacks import EventCallback, BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np


class ActionNoiseCallback(BaseCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
            self,
            start_noise,
            end_noise,
            steps,
            n_eval_episodes: int = 1,
            eval_freq: int = 10000,
            deterministic: bool = True,
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
