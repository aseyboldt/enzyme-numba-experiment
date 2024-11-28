import numpy as np
import numba
import nuts_py
import cffi
import fastprogress
import subprocess

N = 1000

mu = np.random.randn(N)

@numba.njit()
def logp(x):
    out = 0.
    for i in range(len(x)):
        diff = x[i] - mu[i]
        out += (diff * diff) / 2
    return -out

sig = numba.types.void(
    numba.types.size_t,
    numba.types.CPointer(numba.types.float64),
    numba.types.CPointer(numba.types.float64),
)
@numba.cfunc(sig, nogil=True, boundscheck=False, no_cpython_wrapper=True, forceinline=True)
def logp(n_dim, out_, x_):
    out = numba.carray(out_, ())
    x = numba.carray(x_, (n_dim,))
    out[()] = logp(x)

with open("logp_func.ll", "w") as file:
    file.write(logp.inspect_llvm().replace(logp.native_name, "_logp_val"))


subprocess.check_call(["make"])


ffi = cffi.FFI()

ffi.cdef("void logp_grad(size_t, size_t, size_t, size_t);")
lib = ffi.dlopen("./libgrad.so")
grad_address = int(ffi.cast("intptr_t", lib.logp_grad))

def make_user_data():
    return 0

x = np.random.randn(N)
settings = nuts_py.lib.SamplerArgs()
n_chains = 4
n_draws = 1000
seed = 42
n_try_init = 10
n_tune = 1000

import time

n_loops = 100

start = time.time()
for _ in range(n_loops):
    sampler = nuts_py.lib.ParallelSampler(grad_address, make_user_data, N, x, settings, n_chains=n_chains, n_draws=n_draws, seed=seed, n_try_init=10)
    try:
        draws = np.full((n_chains, n_draws + n_tune, N), np.nan)
        infos = []
        for draw, info in fastprogress.progress_bar(sampler, total=n_chains * (n_draws + n_tune)):
            infos.append(info)
            draws[info.chain, info.draw, :] = draw
    finally:
        sampler.finalize()

end = time.time()
print((end - start) / n_loops)
