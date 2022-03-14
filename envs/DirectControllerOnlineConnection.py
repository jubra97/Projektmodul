import pyads
import time
import numpy as np
from ctypes import sizeof, Structure
import matplotlib.pyplot as plt
import threading
import copy

BUFFER_SIZE = 20


class AdsBuffer(Structure):
    """
    Class that represents the output buffer that can be read via ads.
    """

    _fields_ = [

        ("t", pyads.PLCTYPE_LREAL * BUFFER_SIZE),
        ("w", pyads.PLCTYPE_LREAL * BUFFER_SIZE),
        ("u", pyads.PLCTYPE_LREAL * BUFFER_SIZE),
        ("y", pyads.PLCTYPE_LREAL * BUFFER_SIZE)

    ]


class DirectControllerOnlineConnection:
    def __init__(self):
        """
        Connection via ADS to read and write data to a simulink model. TODO: Add model name and add model to git repo
        The model variables are pulled with 500Hz while the model on the SPS is running with 4kHz. Because the SPS model
        uses an output buffer every timestep can be read. Because the data is updated via a callback function a mutex is
        needed to read and write to sample data.
        """
        self.last_timestamps = []
        self.last_timestamp = 0

        self.ads_buffer_mutex = threading.Lock()
        self.last_t = []
        self.last_w = []
        self.last_y = []
        self.last_u = []

        self.last_runs = []
        self.reset_triggered = True

        self.sample_freq = 4000
        self.n_updates = 0

        self.plc = pyads.Connection('192.168.10.200.1.1', 352)
        self.plc.open()

        # if you add a new in/output you can get a list of all ads variables here.
        # symbols = self.plc.get_all_symbols()
        # for sym in symbols:
        #     print(sym)

        self.ads_buffer_sps = self.plc.get_symbol("Object3 (RL_Learn_DirectControl).Output.ads_buffer", array_size=20)
        attr = pyads.NotificationAttrib(length=sizeof(AdsBuffer), trans_mode=pyads.ADSTRANS_SERVERCYCLE, max_delay=1.0, cycle_time=2.0)
        self.ads_buffer_sps.add_device_notification(self.update_obs, attr)

        self.input_u = self.plc.get_symbol("Object3 (RL_Learn_DirectControl).Input.goal_torque")
        self.reset_trigger = self.plc.get_symbol("Object3 (RL_Learn_DirectControl).Input.In1")

    def reset(self):
        """
        Reset saved samples and also trigger reset of the model on the sps.
        :return:
        """
        print("reset called")
        self.reset_triggered = True
        with self.ads_buffer_mutex:
            if self.last_t:
                self.last_runs.append({"t": copy.copy(self.last_t),
                                       "w": copy.copy(self.last_w),
                                       "y": copy.copy(self.last_y),
                                       "u": copy.copy(self.last_u)
                                       })
                self.last_t = []
                self.last_w = []
                self.last_y = []
                self.last_u = []

        self.reset_trigger.write(1)
        self.reset_trigger.write(0)
        self.reset_triggered = False

    def update_obs(self, notification, data):
        """
        Callback when new data is received. Save data two class variables. The received data possesses a field with
        timestamps. Those are used two only save new data.
        :param notification:
        :param data:
        :return:
        """
        if self.reset_triggered:
            return

        _handle, _datetime, value = self.plc.parse_notification(
            notification, self.ads_buffer_sps.plc_type
        )

        t = value[:BUFFER_SIZE]
        w = value[BUFFER_SIZE:BUFFER_SIZE*2]
        u = value[BUFFER_SIZE*2:BUFFER_SIZE*3]
        y = value[BUFFER_SIZE*3:BUFFER_SIZE*4]
        self.n_updates += 1

        with self.ads_buffer_mutex:
            if not self.last_t:
                first_t_non_zero = (np.array(t) != 0).tolist().index(1)

                self.last_t = t[first_t_non_zero - 1:]
                self.last_w = w[first_t_non_zero - 1:]
                self.last_u = u[first_t_non_zero - 1:]
                self.last_y = y[first_t_non_zero - 1:]

            else:
                # check which data is new
                is_new_timestamp = (np.array(t) - self.last_t[-1]) > 0
                if is_new_timestamp.all():
                    raise ValueError("Lost Step")
                try:
                    index_first_new_value = is_new_timestamp.tolist().index(1)
                except ValueError:
                    index_first_new_value = 0
                    print("maybe lost step")
                    # print(t)
                    # print(self.last_t[-50:])
                 # add new data to lists
                self.last_t += t[index_first_new_value:]
                self.last_w += w[index_first_new_value:]
                self.last_u += u[index_first_new_value:]
                self.last_y += y[index_first_new_value:]

    def set_u(self, u):
        """
        Write u to sps model.
        :param u: U
        :return:
        """
        self.input_u.write(u)


if __name__ == "__main__":
    env = DirectControllerOnlineConnection()

    for i in range(1):
        env.reset()
        # print(i * 50)
        # env.set_pi(0.3 * i, 0.1 * i)
        time.sleep(10)
    for run in env.last_runs:
        print(run["t"][0])

    with env.ads_buffer_mutex:
        plt.plot(env.last_t, env.last_w)
        plt.plot(env.last_t, env.last_u)
        plt.plot(env.last_t, env.last_y)
        plt.show()

    with env.ads_buffer_mutex:
        plt.plot(env.last_t)
        plt.show()

    with env.ads_buffer_mutex:
        plt.plot(env.last_t, env.last_w)
        plt.plot(env.last_t, env.last_u)
        plt.plot(env.last_t, env.last_y)
        plt.show()