import numpy as np
gpu_enable = True
try:
    import cupy as cp
    cupy = cp
except ImportError:
    gpu_enable = False
from dezero import Variable
import dezero.mlx


def get_array_module(x):
    """Returns the array module for `x`.

    Args:
        x (dezero.Variable or numpy.ndarray or cupy.ndarray or mlx.core.array):
            Values to determine whether NumPy, CuPy, or MLX should be used.

    Returns:
        module: `cupy`, `dezero.mlx.xp_mlx`, or `numpy` is returned based on
            the argument.
    """
    if isinstance(x, Variable):
        x = x.data

    if dezero.mlx.is_mlx_array(x):
        return dezero.mlx.xp_mlx

    if not gpu_enable:
        return np
    xp = cp.get_array_module(x)
    return xp


def as_numpy(x):
    """Convert to `numpy.ndarray`.

    Args:
        x (`numpy.ndarray` or `cupy.ndarray` or `mlx.core.array`): Arbitrary
            object that can be converted to `numpy.ndarray`.
    Returns:
        `numpy.ndarray`: Converted array.
    """
    if isinstance(x, Variable):
        x = x.data

    if np.isscalar(x):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    elif dezero.mlx.is_mlx_array(x):
        return dezero.mlx.as_numpy(x)
    return cp.asnumpy(x)


def as_cupy(x):
    """Convert to `cupy.ndarray`.

    Args:
        x (`numpy.ndarray` or `cupy.ndarray`): Arbitrary object that can be
            converted to `cupy.ndarray`.
    Returns:
        `cupy.ndarray`: Converted array.
    """
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        raise Exception('CuPy cannot be loaded. Install CuPy!')
    if dezero.mlx.is_mlx_array(x):
        x = dezero.mlx.as_numpy(x)
    return cp.asarray(x)


def as_mlx(x):
    """Convert to `mlx.core.array`."""
    if gpu_enable and isinstance(x, cp.ndarray):
        x = cp.asnumpy(x)
    return dezero.mlx.as_mlx(x)