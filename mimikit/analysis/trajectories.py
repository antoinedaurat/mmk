import numpy as np
from sklearn.neighbors import NearestNeighbors


def distance_to_nnbrs(x, y, k=4):
    """
    Find k nearest neighbors in y for each frame in x.
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(y)
    nframes = x.shape[0]
    dists = np.empty((nframes, k))
    inds = np.empty((nframes, k), dtype=np.int)
    for k in range(nframes):
        d, i = nbrs.kneighbors(x[k:k+1])
        dists[k] = d[0]
        inds[k] = i[0]
    return dists, inds


def find_short_trajectory(X, original, dists, indices, n_trial_pathes=16):
    """
    For each frame in STFT ``X`` an index into the STFT ``original`` is found such that
    the spectral distance between the two frames is small.  The functions tries to
    keep the sum of the absolute differences of the indices in the resulting sequence small
    (to avoid a jumpy trajectories).

    Parameters
    ----------
    X : ndarray
        STFT for which to find the trajectory
    original : ndarray
        The STFT which is indexed by the output trajectory
    dists : ndarray
        distances computed by :func: analysis.by distance_to_nnbr
    inds : ndarray
        indices computed by :func: analysis.by distance_to_nnbr
    n_trial_pathes : int
        Number of trial trajectories used to find the trajectory with lowest traveled distance

    Returns
    -------
    traj : list
        A list with found trajectories
    dists : list
        Distances traveled for each trajectory

    """
    tdiffs = np.array([(indices[1:] - indices[:-1,k:k+1]) for k in range(indices.shape[-1])])
    tindsdown = np.argmax(np.where(tdiffs < 0, tdiffs, -np.inf), axis=2)
    tindsup = np.argmin(np.where(tdiffs >= 0, tdiffs, np.inf), axis=2)
    tinds = np.moveaxis(np.stack((tindsdown, tindsup)), 0, -1)
    original_log = np.log(np.maximum(abs(original), 0.000001)).T
    X_log = np.log(np.maximum(abs(X), 0.000001))
    traj = [[0]]
    cumdist = [0]
    for k in range(X.shape[-1] - 1):
        while len(cumdist) > n_trial_pathes:
            # find worst trajectory and eliminate
            maxelt = max(cumdist)
            delinds = [i for i, x in enumerate(cumdist) if x == maxelt]
            errors = [((original_log[:, indices[k, traj[i][-1]]] - X_log[:, k])**2).sum() for i in delinds]
            nleft = len(cumdist) - n_trial_pathes
            order = sorted(range(len(errors)), key=lambda k: errors[k])[:nleft]
            for o in sorted(order, reverse=True):
                id = delinds[o]
                del cumdist[id]
                del traj[id]
        # append trajectories
        for i in range(len(traj)):
            # append lower path
            newpath = list.copy(traj[i])
            prev = traj[i][-1]
            curdist = cumdist[i]
            i1 = tinds[prev, k, 0]
            dist = tdiffs[prev, k, i1]
            traj[i].append(i1)
            cumdist[i] += abs(dist)
            traj.append(newpath)
            # upper path
            i1 = tinds[prev, k, 1]
            dist = tdiffs[prev, k, i1]
            newpath.append(i1)
            cumdist.append(curdist + abs(dist))
    return traj, cumdist

