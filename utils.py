import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def create_eval_plot(env):
    samples_per_episode = int(env.simulation_time * env.controller_sample_frequency)
    sim_time = np.linspace(0, env.simulation_time, samples_per_episode)

    fig, ax = plt.subplots(2, 3, figsize=(20, 10))

    # plot obs axes
    ax[0][0].set_title("Obs")
    obs = np.array(env.observations_log)
    ax[0][0].plot(sim_time, obs[-samples_per_episode:, 0], label="current_set_point")
    ax[0][0].plot(sim_time, obs[-samples_per_episode:, 1], label="current_system_input")
    ax[0][0].plot(sim_time, obs[-samples_per_episode:, 2], label="current_system_output")
    ax[0][0].plot(sim_time, obs[-samples_per_episode:, 3], label="current_system_output_dot")
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

    ax[1][2].text(0.5, 0.5, f"Integrated Reward: {np.sum(reward_data['reward'])}")

    fig.tight_layout()
    return fig
