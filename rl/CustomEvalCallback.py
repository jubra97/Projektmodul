import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import EventCallback, BaseCallback
from stable_baselines3.common.logger import Figure
import numpy as np

class CustomEvalCallback(BaseCallback):
    """
    Callback for evaluating an agent. Creates a figure and the current RMSE that is logged to the tensorboard.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param early_stopping: Stopp the agent early if the reached rmse is too bad.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param verbose:
    """

    def __init__(
            self,
            eval_env,
            best_model_save_path,
            n_eval_episodes: int = 1,
            eval_freq: int = 10000,
            early_stopping: bool = False,
            deterministic: bool = True,
            verbose: int = 1,
    ):
        super(CustomEvalCallback, self).__init__(verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.evaluate = False
        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.best_reward = -np.inf
        self.early_stopping = early_stopping
        self.early_stopping_counter = 0

    def _init_callback(self):
        # with the out commented code below you could initialize your network with your own starting weights
        import torch as th
        # import torch.nn as nn
        # def init_with_zero(m):
        #     if type(m) == nn.Linear:
        #         # print(m.weight)
        #         # nn.init.uniform_(m.weight, -0.00001, 0.00001)
        #         # print(m.weight)
        #         nn.init.zeros_(m.weight)
        # self.model.actor.mu.apply(init_with_zero)
        # self.model.actor_target.mu.apply(init_with_zero)

        # self.model.critic.mu.apply(init_with_zero)
        # self.model.critic_target.mu.apply(init_with_zero)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.evaluate = True

        # stop after three bad test episodes
        if self.early_stopping_counter < 3 or not self.early_stopping:
            return True
        else:
            return False

    def _on_rollout_end(self) -> None:
        # only evaluate after a finished episode
        if self.evaluate:
            self.evaluate = False

            # run evaluation on eval env
            ob = self.eval_env.reset()
            done = False
            rewards = []
            while done is not True:
                action, _states = self.model.predict(ob, deterministic=self.deterministic)
                ob, reward, done, info = self.eval_env.step(action)
                rewards.append(reward)

            # log metrics
            print(f"Logging Image Call Nr: {self.n_calls}")
            fig, _, rmse, smoothness = self.eval_env.create_eval_plot()
            self.logger.record("Overview/A", Figure(fig, close=True), exclude=("stdout", "log", "json", "csv"))
            plt.close()
            self.logger.record("rollout/ep_rew_mean_eval", np.sum(rewards))
            self.logger.record("rollout/ep_rmse_eval", rmse)
            self.logger.record("rollout/ep_smoothness_eval", smoothness)

            # update early stopping counter; only use it if enough training steps are over
            if self.num_timesteps > 120_000:
                if rmse > 0.4:
                    self.early_stopping_counter += 1
                else:
                    self.early_stopping_counter = 0

            # save model if it's a new best model
            mean_reward = np.mean(rewards)
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.model.save(self.best_model_save_path)
                print("New best model!")