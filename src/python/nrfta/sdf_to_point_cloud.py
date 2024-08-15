import numpy as np
import scipy as sp
import gpytoolbox as gpy
import time
import math

from .outside_points_from_rasterization import outside_points_from_rasterization
from .locally_make_feasible import locally_make_feasible

def sdf_to_point_cloud(U, S,
    rng_seed=3452,
    rasterization_resolution=None,
    n_local_searches=None,
    local_search_iters=20,
    batch_size=10000,
    num_rasterization_spheres=0,
    tol=1e-4,
    parallel=False
    ):
    assert np.min(U)>=0. and np.max(U)<=1.

    d = U.shape[1]
    assert d==2 or d==3, "Only dimensions 2 and 3 supported."

    assert rasterization_resolution>0, "Rasterization resolution must > 0"

    n_sdf = U.shape[0]

    # Pick default values if not supplied.
    if rasterization_resolution is None:
        rasterization_resolution = 64 * math.ceil(n_sdf**(1./d)/16.)
    if n_local_searches is None:
        n_local_searches = math.ceil(2. * n_sdf**(1./d))

    # RNG used to compute random numbers and new seeds during the method.
    rng = np.random.default_rng(seed=rng_seed)
    seed = lambda : rng.integers(0,np.iinfo(np.int32).max)

    P = outside_points_from_rasterization(U, S,
        rng_seed=seed(),
        res=rasterization_resolution, num_spheres=num_rasterization_spheres,
        tol=tol, parallel=parallel)
    
    # If we found no points at all, return empty arrays here.
    if P.size == 0:
        return np.array([], dtype=np.float64), \
            np.array([], dtype=np.float64), \
            np.array([], dtype=np.int32), \
            np.array([], dtype=np.float64)

    Pf, N, f = locally_make_feasible(U, S, P,
        rng_seed=seed(),
        n_local_searches=n_local_searches,
        local_search_iters=local_search_iters,
        batch_size=batch_size,
        tol=tol,
        parallel=parallel)

    return P, N, f, Pf
