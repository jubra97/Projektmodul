import matplotlib; matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3.common.callbacks import EventCallback, BaseCallback
from stable_baselines3.common.logger import Figure

class CustomEvalCallback(EventCallback):
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


    def _init_callback(self) -> None:
        pass

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # crete eval env
            print("Start Log")
            env = self.eval_env()

            ob = env.reset()
            done = False
            while done is not True:
                action, _states = self.model.predict(ob, deterministic=self.deterministic)
                ob, reward, done, info = env.step(action)

            print(f"Logging Image Call Nr: {self.n_calls}")
            samples_per_episode = int(env.simulation_time * env.controller_sample_frequency)
            sim_time = np.linspace(0, env.simulation_time, samples_per_episode)

            fig, ax = plt.subplots(2, 3, figsize=(20, 10))

            # plot obs axes
            ax[0][0].set_title("Obs")
            obs = np.array(env.observations_log)
            ax[0][0].plot(sim_time, obs[-samples_per_episode:, 0], label="Error")
            ax[0][0].plot(sim_time, obs[-samples_per_episode:, 1], label="Integrated Error")
            ax[0][0].plot(sim_time, obs[-samples_per_episode:, 2] * 100, label="Derived Error (*100)")
            ax[0][0].grid()
            ax[0][0].legend()

            # plt action axes
            action_data = pd.DataFrame(env.actions_log[-samples_per_episode:]).to_dict(orient="list")
            ax[0][1].set_title("Action")
            ax[0][1].plot(sim_time, action_data["reference_value"][-samples_per_episode:], label="Reference Value")
            ax[0][1].plot(sim_time, action_data["action"][-samples_per_episode:], label="Action")
            ax[0][1].grid()
            ax[0][1].legend()

            # go back from list of dicts to dict of lists
            reward_data = pd.DataFrame(env.rewards_log[-samples_per_episode:]).to_dict(orient="list")

            # plt reward axes
            ax[1][0].set_title("Reward")
            ax[1][0].plot(sim_time, reward_data["reward"], label="Reward")
            ax[1][0].grid()
            ax[1][0].legend()

            # plt reward shares
            ax[1][1].set_title("Reward Shares")
            ax[1][1].plot(sim_time, reward_data["pen_error"], label="Error Share")
            ax[1][1].plot(sim_time, reward_data["pen_integrated"], label="Integrated Error Share")
            ax[1][1].plot(sim_time, reward_data["pen_action"], label="Action Share")
            ax[1][1].grid()
            ax[1][1].legend()

            # plt u and out
            ax[0][2].set_title("Function")
            ax[0][2].plot(env.t, env.u, label="Set Point")
            ax[0][2].plot(env.t, env.out, label="Output")
            ax[0][2].grid()
            ax[0][2].legend()

            fig.tight_layout()
            self.logger.record("Overview/A", Figure(fig, close=True), exclude=("stdout", "log", "json", "csv"))
            plt.close()

            del env

        return True
