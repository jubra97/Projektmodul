import pyads
import time
import numpy as np

# connect to plc and open connection
plc = pyads.Connection('192.168.10.200.1.1', 353)
with plc:
    # # vars = plc.get_all_symbols()
    # # for var in vars:
    # #     print(var)
    #
    # start_time = time.time()
    # print(start_time-start_time)
    # plc.write_by_name("Object3 (ads_test).Input.timestamp_in", start_time)
    # print(time.time()-start_time)
    # return_time = plc.read_by_name("Object3 (ads_test).Input.timestamp_in")
    # print(time.time()-start_time)
    #
    # var_handle = plc.get_handle("Object3 (ads_test).Input.timestamp_in")
    #
    # times = []
    # for _ in range(1000):
    #     start_time = time.time()
    #     plc.write_by_name("", start_time, pyads.PLCTYPE_LREAL, handle=var_handle)
    #     return_time = plc.read_by_name("", pyads.PLCTYPE_LREAL, handle=var_handle)
    #     times.append(time.time()-start_time)
    #
    # print(np.mean(times))
    # print(max(times))
    # print(min(times))
    # plc.release_handle(var_handle)
    timestamps = []
    @plc.notification(pyads.PLCTYPE_LREAL)
    def callback(handle, name, timestamp, value):
        # print(
        #     '{1}: received new notitifiction for variable "{0}", value: {2}'
        #         .format(name, timestamp, value)
        # )
        timestamps.append(timestamp)


    notification_handle, var_handle = plc.add_device_notification("Object3 (ads_test).Input.timestamp_in", pyads.NotificationAttrib(8),
                                callback)

    for i in range(1000):
        plc.write_by_name("", i, pyads.PLCTYPE_LREAL, handle=var_handle)

    time.sleep(1)
    timestamps = np.array(timestamps)
    diffs = (timestamps[1:] - timestamps[:-1])
    print(diffs)
    print(np.mean(diffs))
    print(max(diffs))
    print(min(diffs))

    # plc.release_handle(notification_handle)
    plc.release_handle(var_handle)
