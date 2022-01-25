import scipy.io

def load_data(matfile):
    mat = scipy.io.loadmat('SysIdentPy/data/' + matfile)
    u_train = mat['u_train'][0].reshape(-1,1)
    u_valid = mat['u_valid'][0].reshape(-1,1)
    u2_train = mat['u2_train'][0].reshape(-1,1)
    u2_valid = mat['u2_valid'][0].reshape(-1,1)
    y_train = mat['y_train'][0].reshape(-1,1)
    y_valid = mat['y_valid'][0].reshape(-1,1)
    return u_train, u_valid, u2_train, u2_valid, y_train, y_valid