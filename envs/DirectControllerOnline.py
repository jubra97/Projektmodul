import pyads

# connect to plc and open connection
plc = pyads.Connection('192.168.10.200.1.1', 352)
plc.open()
vars = plc.get_all_symbols()
print(vars)
plc.close()