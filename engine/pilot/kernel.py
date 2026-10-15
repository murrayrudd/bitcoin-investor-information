"""Optional compiled excess-demand sum; Python remains the reference model.

Only the repeated scalar excess-demand sum is accelerated. Root search,
accounting, choices at the solution, and verification remain unchanged.
"""
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from scipy.special import logit

_function=None


def enable_compiled(required=False):
    global _function
    if _function is not None:return True
    source=Path(__file__).with_name('_excess.cpp')
    library=source.with_name('_excess_kernel.so')
    try:
        if not library.exists() or library.stat().st_mtime<source.stat().st_mtime:
            compiler=shutil.which('g++') or shutil.which('clang++')
            if compiler is None:raise RuntimeError('No C++ compiler; use Python reference kernel')
            temporary=library.with_suffix(f'.{os.getpid()}.tmp.so')
            subprocess.run([compiler,'-O3','-std=c++11','-shared','-fPIC',str(source),'-o',str(temporary)],check=True,capture_output=True)
            temporary.replace(library)
        handle=ctypes.CDLL(str(library))
        f=handle.bitcoin_excess
        ptr=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
        f.argtypes=[ctypes.c_int,ctypes.c_double]+[ptr]*6+[ctypes.c_double]*3
        f.restype=ctypes.c_double
        _function=f
        return True
    except Exception:
        if required:raise
        return False


def disable_compiled():
    global _function
    _function=None


def excess_function(cash,holdings,floor,cap,q,gamma,low,high,stock):
    if _function is None:return None
    arrays=[np.ascontiguousarray(x,dtype=np.float64) for x in [cash,holdings,floor,cap,logit(q),1/gamma]]
    fn=_function
    return lambda price: fn(len(cash),float(price),*arrays,low,high,stock)
