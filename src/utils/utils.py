import numpy as np
import os

eps = 1e-5

def slerp(p0, p1, t):
    """
    Spherical linear interpolation
    """
    val = np.dot(np.squeeze(p0/np.linalg.norm(p0)), np.squeeze(p1/np.linalg.norm(p1)))
    val = val if val >= -1 else -1
    val = val if val < 1 else 1
    omega = np.arccos(val)
    so = np.sin(omega)
    if so == 0.:
        so += eps
    return np.sin(1.0 - t) * omega / so * p0 + np.sin(t * omega) / so * p1

def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)