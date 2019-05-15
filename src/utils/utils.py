import numpy as np
import os

def slerp(p0, p1, t):
    """
    Spherical linear interpolation
    """
    omega = np.arccos(np.dot(np.squeeze(p0/np.linalg.norm(p0)),
                             np.squeeze(p1/np.linalg.nprm(p1))))
    so = np.sin(omega)
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