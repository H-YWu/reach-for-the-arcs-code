import numpy as np
import scipy as sp
import gpytoolbox as gpy
import time
import platform, os, sys
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
import math
from rfta_bindings import _locally_make_feasible_cpp_impl

def locally_make_feasible(U, S, P,
    rng_seed=3452,
    n_local_searches=None,
    local_search_iters=20,
    batch_size=10000,
    tol=1e-4,
    parallel=False):

    assert np.min(U)>=0. and np.max(U)<=1.
    assert P.size > 0, "There needs to be at least one point outside the spheres."

    d = U.shape[1]
    assert d==2 or d==3, "Only dimensions 2 and 3 supported."

    n_sdf = U.shape[0]

    # Pick default values if not supplied.
    if n_local_searches is None:
        n_local_searches = math.ceil(2. * n_sdf**(1./d))

    # RNG used to compute random numbers and new seeds during the method.
    rng = np.random.default_rng(seed=rng_seed)
    seed = lambda : rng.integers(0,np.iinfo(np.int32).max)

    # Batching
    if batch_size > 0 and batch_size < n_sdf:
        batch = rng.choice(n_sdf, batch_size)
    else:
        batch = np.arange(n_sdf)
        rng.shuffle(batch)

    Pf, N, f = _locally_make_feasible_cpp_impl(U.astype(np.float64),
        S.astype(np.float64), P.astype(np.float64),
        batch.astype(np.int32),
        seed(), 
        n_local_searches, local_search_iters,
        tol, np.inf, parallel, False)

    return Pf, N, f


