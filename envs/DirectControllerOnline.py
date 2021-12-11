import pyads
import time
import numpy as np
from ctypes import sizeof, Structure
import matplotlib.pyplot as plt
import threading


BUFFER_SIZE = 20


class AdsBuffer(Structure):

    _fields_ = [

        ("t", pyads.PLCTYPE_LREAL * BUFFER_SIZE),
        ("w", pyads.PLCTYPE_LREAL * BUFFER_SIZE),
        ("u", pyads.PLCTYPE_LREAL * BUFFER_SIZE),
        ("y", pyads.PLCTYPE_LREAL * BUFFER_SIZE)

    ]


class DirectControllerOnline:
    def __init__(self):
        self.last_timestamps = []
        self.last_timestamp = 0

        self.ads_buffer_mutex = threading.Lock()
        self.last_t = []
        self.last_w = []
        self.last_y = []
        self.last_u = []

        self.plc = pyads.Connection('192.168.10.200.1.1', 352)
        self.plc.open()
        # symbols = self.plc.get_all_symbols()
        # for sym in symbols:
        #     print(sym)

        self.ads_buffer_sps = self.plc.get_symbol("Object3 (RL_Learn_DirectControl).Output.ads_buffer", array_size=20)
        attr = pyads.NotificationAttrib(length=sizeof(AdsBuffer), trans_mode=pyads.ADSTRANS_SERVERCYCLE, max_delay=1.0, cycle_time=5.0)
        self.ads_buffer_sps.add_device_notification(self.update_obs, attr)

        self.input_u = self.plc.get_symbol("Object3 (RL_Learn_DirectControl).Input.goal_torque")

    def update_obs(self, notification, data):
        _handle, _datetime, value = self.plc.parse_notification(
            notification, self.ads_buffer_sps.plc_type
        )

        t = value[:BUFFER_SIZE]
        w = value[BUFFER_SIZE:BUFFER_SIZE*2]
        u = value[BUFFER_SIZE*2:BUFFER_SIZE*3]
        y = value[BUFFER_SIZE*3:BUFFER_SIZE*4]

        with self.ads_buffer_mutex:
            if not self.last_t:
                self.last_t = t
                self.last_w = w
                self.last_u = u
                self.last_y = y

            else:
                # check witch data is new
                is_new_timestamp = (np.array(t) - self.last_t[-1]) > 0
                if is_new_timestamp.all():
                    raise ValueError("Lost Step")
                index_first_new_value = is_new_timestamp.tolist().index(1)
                 # add new data to lists
                self.last_t += t[index_first_new_value:]
                self.last_w += w[index_first_new_value:]
                self.last_u += u[index_first_new_value:]
                self.last_y += y[index_first_new_value:]

    def set_u(self, u):
        self.input_u.write(u)


if __name__ == "__main__":
    env = DirectControllerOnline()

    for i in range(1):
        print(i)
        time.sleep(60)

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