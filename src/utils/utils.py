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

def blend(outputs, target):
        """Calculate interpolation between choosen outputs and target
        Parameters:
            outputs (array) -- array of latent codes [len_sequence, z_dim]
            target (array) -- array of latent codes [z_dim]
        Returns:
            transition (array) -- transition frames [len_sequence, z_dim]
        """
        # find output with least L2 distance to target
        o2t = outputs - target
        o2o = outputs[1:] - outputs[:-1]
        o2t = np.linalg.norm(o2t, axis=-1)
        o2o = np.linalg.norm(o2o, axis=-1)
        o2o_max = np.amax(o2o, axis=0)

        o2t_min = np.amin(o2t, axis=0)
        o2t_minidx = np.argmin(o2t, axis=0)

        need_blend = o2t_min > o2o_max
        transition = outputs[:o2t_minidx]
        if need_blend:
            n_steps = int(np.ceil(o2t_min / o2o_max))
            blendings = np.array([slerp(outputs[o2t_minidx], target, t) for t in np.linspace(0, 1, n_steps)])
            transition = np.concatenate((transition, blendings), axis=0)

        return transition

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