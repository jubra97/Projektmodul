from stable_baselines3.common.callbacks import BaseCallback
from envs.DirectControllerPT2 import DirectControllerPT2
from stable_baselines3.common.logger import Figure
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class TensorboardCallback(BaseCallback):
    def __init__(self, env:DirectControllerPT2, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.env = env

    def _on_step(self) -> bool:
        for key, value in self.env.tensorboard_log.items():
            self.logger.record(key, value)
        self.logger.dump(self.num_timesteps)
        # if self.env.simulation_time_steps >= len(self.env.t)-100 and len(self.model.ep_info_buffer) > 0:
        #     print("Logging Image")
        #     samples_per_episode = int(self.env.simulation_time * self.env.controller_sample_frequency)
        #     sim_time = np.linspace(0, self.env.simulation_time, samples_per_episode)
        #
        #     fig, ax = plt.subplots(2, 3, figsize=(20, 10))
        #
        #     # plot obs axes
        #     ax[0][0].set_title("Obs")
        #     obs = np.array(self.env.observations_log)
        #     ax[0][0].plot(sim_time, obs[-samples_per_episode:, 0], label="Error")
        #     ax[0][0].plot(sim_time, obs[-samples_per_episode:, 1], label="Integrated Error")
        #     ax[0][0].plot(sim_time, obs[-samples_per_episode:, 2] * 100, label="Derived Error (*100)")
        #     ax[0][0].grid()
        #     ax[0][0].legend()
        #
        #     # plt action axes
        #     ax[0][1].set_title("Action")
        #     ax[0][1].plot(sim_time, self.env.actions_log[-samples_per_episode:], label="Action")
        #     ax[0][1].grid()
        #     ax[0][1].legend()
        #
        #     # go back from list of dicts to dict of lists
        #     reward_data = pd.DataFrame(self.env.rewards_log[-samples_per_episode:]).to_dict(orient="list")
        #
        #     # plt reward axes
        #     ax[1][0].set_title("Reward")
        #     ax[1][0].plot(sim_time, reward_data["reward"], label="Reward")
        #     ax[1][0].grid()
        #     ax[1][0].legend()
        #
        #     # plt reward shares
        #     ax[1][1].set_title("Reward Shares")
        #     ax[1][1].plot(sim_time, reward_data["pen_error"], label="Error Share")
        #     ax[1][1].plot(sim_time, reward_data["pen_integrated"], label="Integrated Error Share")
        #     ax[1][1].plot(sim_time, reward_data["pen_action"], label="Action Share")
        #     ax[1][1].grid()
        #     ax[1][1].legend()
        #
        #     # plt u and out
        #     ax[0][2].set_title("Function")
        #     ax[0][2].plot(self.env.t, self.env.u, label="Set Point")
        #     ax[0][2].plot(self.env.t[:-100], self.env.out, label="Output")
        #     ax[0][2].grid()
        #     ax[0][2].legend()
        #
        #     fig.tight_layout()
        #     self.logger.record("Overview/A", Figure(fig, close=True), exclude=("stdout", "log", "json", "csv"))
        #     plt.close()
        return True
