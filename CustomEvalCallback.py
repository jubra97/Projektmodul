import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import EventCallback, BaseCallback
from stable_baselines3.common.logger import Figure

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
            n_eval_episodes: int = 1,
            eval_freq: int = 10000,
            deterministic: bool = True,
            verbose: int = 1,
    ):
        super(CustomEvalCallback, self).__init__(verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.deterministic = deterministic

        self.eval_env = eval_env

    # def _init_callback(self):
    #     print(type(self.model.actor))
    #     self.logger.output_formats[1].writer.add_graph(self.model.actor.mu)
    #     print("A")
    #     # import torch.nn as nn
    #     # def init_with_zero(m):
    #     #     if type(m) == nn.Linear:
    #     #         nn.init.zeros_(m.weight)
    #     #         # nn.init.zeros_(m.bias)
    #     # self.model.actor.mu.apply(init_with_zero)
    #     # print("A")
    #     # print("B")

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # crete eval env
            print("Start Log")

            ob = self.eval_env.reset()
            done = False
            while done is not True:
                action, _states = self.model.predict(ob, deterministic=self.deterministic)
                ob, reward, done, info = self.eval_env.step(action)

            print(f"Logging Image Call Nr: {self.n_calls}")
            fig = self.eval_env.create_eval_plot()
            self.logger.record("Overview/A", Figure(fig, close=True), exclude=("stdout", "log", "json", "csv"))
            plt.close()

        return True
