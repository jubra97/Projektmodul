import json
import pathlib
import abc
from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq


class DirectController(gym.Env, abc.ABC):

    def __init__(self, log=False, output_freq=100, sensor_freq=4000, reward_function="discrete", observation_function="error_with_vel",
                 oscillation_pen_gain=0.01,
                 oscillation_pen_fun=np.sqrt, error_pen_fun=None):
        """
        Create a gym environment to directly control the actuating value (u) of a system.
        :param log: Log the simulation outcomes.
        """
        super(DirectController, self).__init__()

        # set reward function
        if reward_function == "discrete":
            self.reward_function = self._create_reward_discrete
        elif reward_function == "normal":
            self.reward_function = self._create_reward
        else:
            raise ValueError(
                "No corresponding reward function could be found. If you want to add your own reward function add the"
                " function to this if else block.")

        # set observation function
        if observation_function == "obs_with_vel":
            self.observation_function = self._create_obs_with_vel
        elif observation_function == "last_states":
            self.observation_function = self._create_obs_last_states
        elif observation_function == "error_with_vel":
            self.observation_function = self._create_obs_errors_with_vel
        elif observation_function == "last_errors":
            self.observation_function = self._create_obs_last_errors
        else:
            raise ValueError(
                "No corresponding observation function could be found. If you want to add your own observation"
                "function add the function to this if else block.")

        # self.oscillation_pen_gain_raising = np.linspace(0, 20, 670)
        self.oscillation_pen_gain = oscillation_pen_gain
        self.oscillation_pen_fun = oscillation_pen_fun
        self.error_pen_fun = error_pen_fun

        self.sensor_freq = sensor_freq  # Hz
        self.output_freq = output_freq  # Hz
        assert sensor_freq % output_freq == 0, "sensor_freq must be a multiple ot output_freq!"
        self.measurements_per_output_update = int(sensor_freq / output_freq)
        self.nbr_measurements_to_keep = self.measurements_per_output_update * 2

        # create fifo lists for latest measurement points, new data is inserted from the right side. Because of this the
        # most recent value is the in e.g. self.last_y[-1].
        self.last_u = deque([0] * self.nbr_measurements_to_keep, maxlen=self.nbr_measurements_to_keep)
        self.last_y = deque([0] * self.nbr_measurements_to_keep, maxlen=self.nbr_measurements_to_keep)
        self.last_w = deque([0] * self.nbr_measurements_to_keep, maxlen=self.nbr_measurements_to_keep)
        self.last_t = deque([0] * self.nbr_measurements_to_keep, maxlen=self.nbr_measurements_to_keep)

        self.integrated_error = 0

        # create variables for logging
        self.episode_log = {"obs": {}, "rewards": {}, "action": {}, "function": {}}
        self.log = log
        self.n_episodes = 0
        self.log_all_episodes = []

        # define size of observation space
        test_obs = self.reset()
        obs_size = test_obs.shape[0]
        self.observation_space = gym.spaces.Box(low=np.array([-100] * obs_size, dtype=np.float32),
                                                high=np.array([100] * obs_size, dtype=np.float32),
                                                shape=(obs_size,),
                                                dtype=np.float32)

        # action space size is always one as u is calculated directly
        self.action_space = gym.spaces.Box(low=np.array([-1], dtype=np.float32),
                                           high=np.array([1], dtype=np.float32),
                                           shape=(1,),
                                           dtype=np.float32)

    def reset(self, step_start=None, step_end=None, step_slope=None, custom_w=None):
        """
        Reset the environment. Called before every start of a new episode. If no custom reference value (custom_w) is
        given a ramp (or step) is used.
        :param step_height: Height of step/ramp.
        :param step_slope: Slope of Step. If slope is 0 a step is generated.
        :param custom_w:
        :return:
        """

        # add log of last episode to a list of logs of all episodes
        if self.n_episodes != 0:
            self.log_all_episodes.append(self.episode_log)
        self.n_episodes += 1

        if self.log:
            self.episode_log["obs"]["last_set_points"] = []
            self.episode_log["obs"]["last_system_inputs"] = []
            self.episode_log["obs"]["last_system_outputs"] = []
            self.episode_log["obs"]["errors"] = []
            self.episode_log["obs"]["error"] = []
            self.episode_log["obs"]["error_vel"] = []
            self.episode_log["obs"]["set_point"] = []
            self.episode_log["obs"]["system_input"] = []
            self.episode_log["obs"]["system_output"] = []
            self.episode_log["obs"]["set_point_vel"] = []
            self.episode_log["obs"]["input_vel"] = []
            self.episode_log["obs"]["outputs_vel"] = []
            self.episode_log["obs"]["error_integrated"] = []
            self.episode_log["rewards"]["summed"] = []
            self.episode_log["rewards"]["pen_error"] = []
            self.episode_log["rewards"]["pen_action"] = []
            self.episode_log["action"]["value"] = []
            self.episode_log["action"]["change"] = []
            self.episode_log["function"]["w"] = None
            self.episode_log["function"]["y"] = None

        self.integrated_error = 0
        self.custom_reset()
        obs = self.observation_function()
        return obs

    @abc.abstractmethod
    def custom_reset(self, *args, **kwargs):
        raise NotImplementedError()

    def sensor_data_processing(self):
        pass

    def _create_obs_with_vel(self):
        """
        Create observation consisting of: set point (w), system output (y), system input (u) and their derivations.
        :param first: True if first call in an episode (normally in reset()).
        :return:
        """
        set_points = np.array(list(self.last_w))
        system_outputs = np.array(list(self.last_y))
        system_inputs = np.array(list(self.last_u))

        outputs_vel = (system_outputs[-2] - system_outputs[-1]) * 1 / self.measurements_per_output_update
        input_vel = (system_inputs[-3] - system_inputs[-1]) * 1 / self.measurements_per_output_update
        set_point_vel = (set_points[-2] - set_points[-1]) * 1 / self.measurements_per_output_update

        obs = [set_points[-1], system_inputs[-1], system_outputs[-1], set_point_vel, input_vel, outputs_vel]

        if self.log:
            self.episode_log["obs"]["set_point"].append(obs[0])
            self.episode_log["obs"]["system_input"].append(obs[1])
            self.episode_log["obs"]["system_output"].append(obs[2])
            self.episode_log["obs"]["set_point_vel"].append(obs[3])
            self.episode_log["obs"]["input_vel"].append(obs[4])
            self.episode_log["obs"]["outputs_vel"].append(obs[5])

        return np.array(obs)

    def _create_obs_last_states(self):
        """
        Create observation consisting of: last 3 set points (w), last 3 system outputs (y), last 3 system inputs (u).
        :param first: True if first call in an episode (normally in reset()).
        :return:
        """

        obs = list(self.last_w) + list(self.last_u) + list(self.last_y)

        if self.log:
            self.episode_log["obs"]["last_set_points"].append(list(obs[0:3]))
            self.episode_log["obs"]["last_system_inputs"].append(list(obs[3:6]))
            self.episode_log["obs"]["last_system_outputs"].append(list(obs[6:9]))

        return np.array(obs)

    def _create_obs_errors_with_vel(self):
        """
        Create observation consisting of: system error (e), system input (u) and their derivations.
        :param first: True if first call in an episode (normally in reset()).
        :return:
        """
        set_points = np.array(list(self.last_w))
        system_outputs = np.array(list(self.last_y))
        system_inputs = np.array(list(self.last_u))
        errors = (set_points - system_outputs).tolist()

        error_smooth = [0, 0]
        error_smooth[-1] = np.mean(errors[-10:])
        error_smooth[-2] = np.mean(errors[-20:-10])

        self.integrated_error += error_smooth[-1] * 1 / self.measurements_per_output_update

        error_vel = (error_smooth[-2] - error_smooth[-1]) * 1 /self.measurements_per_output_update
        input_vel = (system_inputs[-41] - system_inputs[-1]) * 1 / self.measurements_per_output_update

        # old_obs = self._create_obs_with_vel()

        obs = [error_smooth[-1],  error_vel * 10, self.integrated_error, input_vel*10]
        # obs = old_obs.tolist() + [error_smooth[-1], system_inputs[-1], error_vel, input_vel]
        # obs = [error_smooth[-1], error_vel]

        if self.log:
            self.episode_log["obs"]["error"].append(obs[0])
            # self.episode_log["obs"]["system_input"].append(obs[3])
            self.episode_log["obs"]["error_vel"].append(obs[1])
            self.episode_log["obs"]["input_vel"].append(obs[3])
            self.episode_log["obs"]["error_integrated"].append(obs[2])
            # self.episode_log["obs"]["system_output"].append(obs[4])
        return np.array(obs)

    def _create_obs_last_errors(self):
        """
        Create observation consisting of: last 3 system errors (e), last 3 system inputs (u).
        :param first: True if first call in an episode (normally in reset()).
        :return:
        """
        set_points = np.array(list(self.last_w))
        system_outputs = np.array(list(self.last_y))
        errors = (set_points - system_outputs).tolist()

        obs = errors + list(self.last_u)

        if self.log:
            self.episode_log["obs"]["errors"].append(obs[0:3])
            self.episode_log["obs"]["last_system_inputs"].append(obs[3:6])

        return np.array(obs)

    def _create_reward(self):
        """
        Create reward as a combination of the current error e between y and w and the change of of the action (to
        smaller the oscillation in u).
        :return: reward
        """
        # get latest system attributes and calculate error/ integrated error
        y = np.array(list(self.last_y)[-self.measurements_per_output_update:])
        w = np.array(list(self.last_w)[-self.measurements_per_output_update:])
        e = np.mean(w - y)

        # calculate action change
        action_change = (self.last_u[-(self.measurements_per_output_update + 1)]
                         - self.last_u[-self.measurements_per_output_update]) \
                         * (1 / self.measurements_per_output_update)

        pen_error = np.square(e)
        pen_action = np.square(action_change) * 0.01
        # pen_integrated = np.square(self.integrated_error) * 0

        reward = 0
        reward -= pen_error
        reward -= pen_action

        if self.log:
            self.episode_log["rewards"]["summed"].append(reward)
            self.episode_log["rewards"]["pen_error"].append(-pen_error)
            self.episode_log["rewards"]["pen_action"].append(-pen_action)

        return reward

    def _create_reward_discrete(self):
        # get latest system attributes and calculate error/ integrated error
        y = np.array(list(self.last_y)[-self.measurements_per_output_update:])
        w = np.array(list(self.last_w)[-self.measurements_per_output_update:])
        e = np.mean(w - y)

        # calculate action change
        action_change = (self.last_u[-(self.measurements_per_output_update + 1)]
                         - self.last_u[-self.measurements_per_output_update]) \
                         * (1 / self.measurements_per_output_update)

        abs_error = abs(e)

        if self.error_pen_fun:
            pen_error = self.error_pen_fun(abs(e))
        else:
            pen_error = abs(e)

        if self.oscillation_pen_fun:
            pen_action = self.oscillation_pen_fun(abs((1/pen_error) * action_change)) * self.oscillation_pen_gain
        else:
            pen_action = abs(action_change) * self.oscillation_pen_gain

        reward = 0
        if abs_error < 0.5:
            reward += 1
        if abs_error < 0.1:
            reward += 2
        if abs_error < 0.05:
            reward += 3
        if abs_error < 0.02:
            reward += 4
        if abs_error < 0.01:
            reward += 5
        if abs_error < 0.005:
            reward += 10
        if abs_error < 0.0005:
            reward += 10

        reward -= pen_error
        reward -= pen_action

        if self.log:
            self.episode_log["rewards"]["summed"].append(reward)
            self.episode_log["rewards"]["pen_action"].append(-pen_action)
            self.episode_log["rewards"]["pen_error"].append(-pen_error)
        return reward

    @abc.abstractmethod
    def step(self, action):
        """
        Step the environment for one update step of the action. Collect sensor values of the simulation and compute
        reward and next observation from them.
        :param action: Next u for simulation.
        :return:
        """
        raise NotImplementedError()

    def render(self, mode="human"):
        """
        Must be here to implement abstract methods of super class.
        :param mode:
        :return:
        """
        ...

    def eval_fft(self):
        """
        Calculate fft of the action signal for one episode. Also compute the action smoothness value in
        https: // arxiv.org / pdf / 2012.06644.pdf.
        :return:
        """
        N = 150
        T = 1 / 100

        actions = self.episode_log["action"]["value"]
        actions = actions - np.mean(actions)
        actions_fft = fft(actions)
        actions_fftfreq = fftfreq(N, T)[:N // 2]

        # https: // arxiv.org / pdf / 2012.06644.pdf Smoothness Measurement
        sm = (2 / actions_fftfreq.shape[0]) * np.sum(actions_fftfreq * 2 / N * np.abs(actions_fft[0:N // 2]))
        return actions_fft, actions_fftfreq, N, sm

    def create_eval_plot(self):
        """
        Crate a plot for tensorboard while training and afterwards for evaluation.
        :return:
        """
        fig, ax = plt.subplots(2, 3, figsize=(20, 12))
        timestamps = np.linspace(0, self.last_t[-1], int(self.last_t[-1] * self.output_freq))

        ax[0][0].set_title("Obs")
        for key, value in self.episode_log["obs"].items():
            if len(value) > 0:
                ax[0][0].plot(timestamps, value[:-1], label=key)  # verschoben zum Reset; eine Obs mehr als Reward!
        ax[0][0].grid()
        ax[0][0].legend()

        ax[1][0].set_title("Rewards")
        for key, value in self.episode_log["rewards"].items():
            if len(value) > 0:
                ax[1][0].plot(timestamps, value, label=key)
        ax[1][0].plot([0], [0], label=f"Sum: {np.sum(self.episode_log['rewards']['summed']):.2f}")
        ax[1][0].grid()
        ax[1][0].legend()

        ax[0][1].set_title("Action")
        for key, value in self.episode_log["action"].items():
            if len(value) > 0:
                ax[0][1].plot(timestamps, value, label=key)
        ax[0][1].grid()
        ax[0][1].legend()

        ax[1][1].set_title("Function")
        for key, value in self.episode_log["function"].items():
            if len(value) > 0:
                ax[1][1].plot(self.sim.t, value, label=key)
        ax[1][1].grid()
        ax[1][1].legend()

        ax[1][2].set_title("FFT")
        actions_fft, actions_fftfreq, N, sm = self.eval_fft()
        ax[1][2].plot(actions_fftfreq, 2 / N * np.abs(actions_fft[0:N // 2]))
        ax[1][2].text(0.5, 0.9, f"Smoothness: {sm}", transform=ax[1][2].transAxes)
        ax[1][2].grid()

        fig.tight_layout()
        return fig

    def eval(self, model, folder_name):
        """
        Run an evaluation with different steps and ramps. Create a plot for every run and save it. Also save a json file
        with some statistics of a run.
        :param model: Model to be used for action prediction.
        :param folder_name: Folder to save evaluation in.
        """
        steps = np.linspace(0, 0.5, 20)
        slopes = np.linspace(0, 0.5, 3)
        i = 1
        pathlib.Path(f"eval\\{folder_name}").mkdir(exist_ok=True)
        rewards = []
        rmse = []
        sms = []
        extra_info = {}
        for step in steps:
            for slope in slopes:
                # slope = slope * 0.1
                actions = []
                # create env
                done = False
                obs = self.reset(0, step, slope)
                while not done:
                    action, _ = model.predict(obs)
                    obs, reward, done, info = self.step(action)
                    rewards.append(reward)
                    actions.append(action)
                _, _, _, smoothness = self.eval_fft()
                sms.append(smoothness)
                fig = self.create_eval_plot()
                plt.savefig(f"eval\\{folder_name}\\{i}_{step}_{slope}.png")
                plt.close()
                i += 1
                rmse_episode = np.sqrt(np.square(np.array(self.w) - np.array(self.sim._sim_out)))
                rmse.append(rmse_episode)
        mean_episode_reward = np.sum(rewards) / self.n_episodes
        extra_info["mean_episode_reward"] = mean_episode_reward
        extra_info["rmse"] = np.mean(rmse)
        extra_info["smoothness"] = np.mean(sms)

        with open(f"eval\\{folder_name}\\extra_info.json", 'w+') as f:
            json.dump(extra_info, f)
        print(f"Eval Info: RMSE: {np.mean(rmse)} --- Smoothness: {np.mean(sms)}")
