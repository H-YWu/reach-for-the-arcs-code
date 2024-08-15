import numpy as np
import scipy as sp
import gpytoolbox as gpy
import platform, os, sys
import math
os_name = platform.system()
if os_name == "Darwin":
    # Get the macOS version
    os_version = platform.mac_ver()[0]
    # print("macOS version:", os_version)

    # Check if the macOS version is less than 14
    if os_version and os_version < "14":
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../build-studio')))
    else:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../build')))
elif os_name == "Windows":
    # For Windows systems
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../build/Debug')))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../build/Release')))
else:
    # For other systems
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../build')))
from rfta_bindings import _fine_tune_point_cloud_iter_cpp_impl

from .point_cloud_to_mesh import point_cloud_to_mesh


def fine_tune_point_cloud(U, S, P, N, f,
    rng_seed=3452,
    fine_tune_iters=10,
    batch_size=10000,
    screening_weight=10.,
    max_points_per_sphere=3,
    n_local_searches=None,
    local_search_iters=20,
    local_search_t=0.01,
    tol=1e-4,
    parallel=False):
    
    d = U.shape[1]
    assert d==2 or d==3, "Only dimensions 2 and 3 supported."
    assert max_points_per_sphere>=1, "There has to be at least one point per sphere."

    n_sdf = U.shape[0]

    # Pick default values if not supplied.
    if n_local_searches is None:
        n_local_searches = math.ceil(2. * n_sdf**(1./d))

    # RNG used to compute random numbers and new seeds during the method.
    rng = np.random.default_rng(seed=rng_seed)
    seed = lambda : rng.integers(0,np.iinfo(np.int32).max)

    for it in range(fine_tune_iters):
        #Generate a random batch of size batch size.
        if batch_size > 0 and batch_size < n_sdf:
            batch = rng.choice(n_sdf, batch_size)
        else:
            batch = np.arange(n_sdf)
            rng.shuffle(batch)

        V,F = point_cloud_to_mesh(P, N,
            screening_weight=screening_weight,
            outer_boundary_type="Neumann",
            parallel=False, verbose=False)

        if(V.size == 0):
            return P, N, f

        P, N, f = fine_tune_point_cloud_iteration(U,
            S,
            V,
            F,
            P,
            N,
            f,
            batch,
            max_points_per_sphere,
            seed(),
            n_local_searches,
            local_search_iters,
            local_search_t,
            tol, parallel)

    return P, N, f


def fine_tune_point_cloud_iteration(U, S, 
    V, F,
    P, N, f,
    batch,
    max_points_per_sphere,
    rng_seed,
    n_local_searches,
    local_search_iters,
    local_search_t,
    tol, 
    parallel):

    P, N, f = _fine_tune_point_cloud_iter_cpp_impl(U.astype(np.float64),
            S.astype(np.float64),
            V.astype(np.float64),
            F.astype(np.int32),
            P.astype(np.float64),
            N.astype(np.float64),
            f.astype(np.int32),
            batch.astype(np.int32),
            max_points_per_sphere,
            rng_seed,
            n_local_searches,
            local_search_iters,
            local_search_t,
            tol, np.inf, parallel, False)

    return P, N, f


