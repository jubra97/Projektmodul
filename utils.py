import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pathlib
import json

# def create_eval_plot(env):
#     samples_per_episode = env.sim.n_sample_points
#     sim_time = np.linspace(0, env.sim.simulation_time, samples_per_episode)
#
#     fig, ax = plt.subplots(2, 3, figsize=(20, 10))
#
#     # plot obs axes
#     ax[0][0].set_title("Obs")
#     obs = np.array(env.observations_log)
#     ax[0][0].plot(sim_time, obs[-samples_per_episode:, 0], label="current_set_point")
#     ax[0][0].plot(sim_time, obs[-samples_per_episode:, 1], label="current_system_input")
#     ax[0][0].plot(sim_time, obs[-samples_per_episode:, 2], label="current_system_output")
#     ax[0][0].plot(sim_time, obs[-samples_per_episode:, 3], label="current_system_output_dot")
#     ax[0][0].grid()
#     ax[0][0].legend()
#
#     # plt action axes
#     action_data = pd.DataFrame(env.actions_log[-samples_per_episode:]).to_dict(orient="list")
#     ax[0][1].set_title("Action")
#     ax[0][1].plot(sim_time, action_data["reference_value"][-samples_per_episode:], label="Reference Value")
#     ax[0][1].plot(sim_time, action_data["action"][-samples_per_episode:], label="Action")
#     ax[0][1].grid()
#     ax[0][1].legend()
#
#     # go back from list of dicts to dict of lists
#     reward_data = pd.DataFrame(env.rewards_log[-samples_per_episode:]).to_dict(orient="list")
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
#     ax[0][2].plot(env.t, env.u, label="Set Point")
#     ax[0][2].plot(env.t, env.out, label="Output")
#     ax[0][2].grid()
#     ax[0][2].legend()
#
#     ax[1][2].text(0.5, 0.5, f"Integrated Reward: {np.sum(reward_data['reward'])}")
#
#     fig.tight_layout()
#     return fig


def eval(env, model, folder_name):
    steps = range(-10, 11)
    slopes = np.linspace(0, 0.5, 3)
    import time
    start = time.perf_counter()
    i = 1
    pathlib.Path(f"eval\\{folder_name}").mkdir(exist_ok=True)
    rewards = []
    rmse = []
    extra_info = {}
    for step in steps:
        for slope in slopes:
            # slope = slope * 0.1

            # create env
            done = False
            obs = env.reset(step, slope)
            while not done:
                action, _ = model.predict(obs)
                obs, reward, done, info = env.step(action)
                rewards.append(reward)
            fig = env.create_eval_plot()
            plt.savefig(f"eval\\{folder_name}\\{i}_{step}_{slope}.png")
            plt.close()
            i += 1
            # rmse_episode = np.square(np.array(env.w) - np.array(env.sim._sim_out))
            rmse_episode = np.square(np.array(env.w[0, :]) - np.array(env.sim._sim_out[1, :]))
            rmse.append(rmse_episode)
    mean_episode_reward = np.sum(rewards)/env.n_episodes
    extra_info["mean_episode_reward"] = mean_episode_reward
    extra_info["rmse"] = np.mean(rmse)

    with open(f"eval\\{folder_name}\\extra_info.json", 'w+') as f:
        json.dump(extra_info, f)
    print(extra_info["rmse"])
    print(f"Time taken: {time.perf_counter() - start}")