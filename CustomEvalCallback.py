import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import EventCallback, BaseCallback
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

class CustomEvalCallback(BaseCallback):
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
        print(type(self.model.actor))
        import torch as th
        # self.logger.output_formats[1].writer.add_graph(self.model.actor.mu, th.Tensor([1] * 7).to(self.model.device))
        # print("A")
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
        # print("A")
        # print("B")

    def _on_step(self) -> bool:
        # n_actions = self.model.action_space.shape[-1]
        # if self.n_calls % 10_000 == 0:
        #     print("Called")
        #     self.model.action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions))
        # if self.n_calls % 500 == 0:
        #     if self.model.action_noise:
        #         last_noise = self.model.action_noise._sigma
        #         new_noise = (last_noise - 0.01) if (last_noise - 0.01) > 0 else None
        #         if new_noise is not None:
        #             self.model.action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(new_noise) * np.ones(n_actions))
        #         else:
        #             self.model.action_noise = None
        #     print(f"Action Noise: {self.model.action_noise}")


        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.evaluate = True
        #     # crete eval env
        #     print("Start Log")
        #
        #     ob = self.eval_env.reset()
        #     done = False
        #     while done is not True:
        #         action, _states = self.model.predict(ob, deterministic=self.deterministic)
        #         ob, reward, done, info = self.eval_env.step(action)
        #
        #     print(f"Logging Image Call Nr: {self.n_calls}")
        #     fig = self.eval_env.create_eval_plot()
        #     self.logger.record("Overview/A", Figure(fig, close=True), exclude=("stdout", "log", "json", "csv"))
        #     plt.close()
        #     # first_layer_weight = self.model.actor.mu._modules["0"].weight
        #     # self.logger.output_formats[1].writer.add_histogram("test", first_layer_weight, self.n_calls)
        if self.early_stopping_counter < 3 or not self.early_stopping:
            return True
        else:
            return False

    def _on_rollout_end(self) -> None:
        if self.evaluate:
            self.evaluate = False
            ob = self.eval_env.reset()
            done = False
            rewards = []
            while done is not True:
                action, _states = self.model.predict(ob, deterministic=self.deterministic)
                ob, reward, done, info = self.eval_env.step(action)
                rewards.append(reward)

            print(f"Logging Image Call Nr: {self.n_calls}")
            fig, _, rmse, smoothness = self.eval_env.create_eval_plot()
            self.logger.record("Overview/A", Figure(fig, close=True), exclude=("stdout", "log", "json", "csv"))
            plt.close()
            self.logger.record("rollout/ep_rew_mean_eval", np.sum(rewards))
            self.logger.record("rollout/ep_rmse_eval", rmse)
            self.logger.record("rollout/ep_smoothness_eval", smoothness)
            if self.num_timesteps > 120_000:
                if rmse > 0.4:
                    self.early_stopping_counter += 1
                else:
                    self.early_stopping_counter = 0
            mean_reward = np.mean(rewards)
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.model.save(self.best_model_save_path)
                print("New best model!")

            # first_layer_weight = self.model.actor.mu._modules["0"].weight
            # self.logger.output_formats[1].writer.add_histogram("test", first_layer_weight, self.n_calls)