import numpy as np

gpu_enable = True
try:
    import mlx.core as mx
    mlx = mx
except ImportError:
    gpu_enable = False
    mx = None
    mlx = None

from dezero import Variable


def _to_mx_dtype(dtype):
    if not gpu_enable or dtype is None:
        return None
    if isinstance(dtype, mx.Dtype):
        return mx.float32 if dtype == mx.float64 else dtype
    if dtype in (float, np.float64, np.float32, 'float32', 'float64', 'f'):
        return mx.float32
    if dtype in (np.float16, 'float16'):
        return mx.float16
    if dtype in (int, np.int64, np.int32, 'int32', 'int64', 'i'):
        return mx.int32
    if dtype in (np.int16, 'int16'):
        return mx.int16
    if dtype in (np.int8, 'int8'):
        return mx.int8
    if dtype in (np.uint8, 'uint8'):
        return mx.uint8
    if dtype in (bool, np.bool_, 'bool'):
        return mx.bool_
    dtype_str = str(dtype)
    if 'float' in dtype_str:
        return mx.float32
    if 'int' in dtype_str:
        return mx.int32
    if 'bool' in dtype_str:
        return mx.bool_
    return mx.float32


class _MLXRandom:
    @staticmethod
    def randn(*shape, dtype=None):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        mx_dtype = _to_mx_dtype(dtype) or mx.float32
        return mx.random.normal(shape=shape, dtype=mx_dtype)

    @staticmethod
    def rand(*shape, dtype=None):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        mx_dtype = _to_mx_dtype(dtype) or mx.float32
        return mx.random.uniform(shape=shape, dtype=mx_dtype)


class _MLXModuleWrapper:
    """NumPy/CuPy-compatible wrapper around mlx.core for DeZero."""

    random = _MLXRandom()

    def __getattr__(self, name):
        return getattr(mx, name)

    @staticmethod
    def array(obj, dtype=None):
        if isinstance(obj, Variable):
            obj = obj.data
        mx_dtype = _to_mx_dtype(dtype)
        if isinstance(obj, mx.array):
            return obj.astype(mx_dtype) if mx_dtype is not None else obj
        if isinstance(obj, (list, tuple)) and len(obj) > 0 and isinstance(obj[0], mx.array):
            stacked = mx.stack(list(obj))
            return stacked.astype(mx_dtype) if mx_dtype is not None else stacked
        arr = np.asarray(obj)
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        elif arr.dtype == np.int64:
            arr = arr.astype(np.int32)
        res = mx.array(arr)
        if mx_dtype is not None and res.dtype != mx_dtype:
            res = res.astype(mx_dtype)
        return res

    @staticmethod
    def asarray(obj, dtype=None):
        return _MLXModuleWrapper.array(obj, dtype=dtype)

    @staticmethod
    def zeros(shape, dtype=None):
        mx_dtype = _to_mx_dtype(dtype) or mx.float32
        return mx.zeros(shape, dtype=mx_dtype)

    @staticmethod
    def ones(shape, dtype=None):
        mx_dtype = _to_mx_dtype(dtype) or mx.float32
        return mx.ones(shape, dtype=mx_dtype)

    @staticmethod
    def zeros_like(a, dtype=None):
        res = mx.zeros_like(a)
        mx_dtype = _to_mx_dtype(dtype)
        return res.astype(mx_dtype) if mx_dtype is not None else res

    @staticmethod
    def ones_like(a, dtype=None):
        res = mx.ones_like(a)
        mx_dtype = _to_mx_dtype(dtype)
        return res.astype(mx_dtype) if mx_dtype is not None else res

    @staticmethod
    def empty(shape, dtype=None):
        mx_dtype = _to_mx_dtype(dtype) or mx.float32
        return mx.zeros(shape, dtype=mx_dtype)

    @staticmethod
    def eye(n, m=None, k=0, dtype=None):
        mx_dtype = _to_mx_dtype(dtype) or mx.float32
        return mx.eye(n, m, k, dtype=mx_dtype)

    @staticmethod
    def arange(*args, dtype=None):
        mx_dtype = _to_mx_dtype(dtype)
        if mx_dtype is not None:
            return mx.arange(*args, dtype=mx_dtype)
        return mx.arange(*args)

    @staticmethod
    def matmul(a, b):
        return mx.matmul(a, b)

    @staticmethod
    def dot(a, b):
        return mx.matmul(a, b)

    @staticmethod
    def rollaxis(a, axis, start=0):
        axes = list(range(a.ndim))
        if axis < 0:
            axis += a.ndim
        if start < 0:
            start += a.ndim
        axes.pop(axis)
        axes.insert(start if start <= axis else start - 1, axis)
        return mx.transpose(a, axes)

    @staticmethod
    def scatter_add(gx, slices, gy):
        return gx.at[slices].add(gy)


xp_mlx = _MLXModuleWrapper() if gpu_enable else None


def is_mlx_array(x):
    if not gpu_enable:
        return False
    if isinstance(x, Variable):
        x = x.data
    return isinstance(x, mx.array)


def get_array_module(x):
    """Returns the array module (`xp_mlx` or `numpy`) for `x`."""
    if isinstance(x, Variable):
        x = x.data
    if gpu_enable and isinstance(x, mx.array):
        return xp_mlx
    return np


def as_numpy(x):
    """Convert `mlx.core.array` or `numpy.ndarray` to `numpy.ndarray`."""
    if isinstance(x, Variable):
        x = x.data
    if np.isscalar(x):
        return np.array(x)
    if isinstance(x, np.ndarray):
        return x
    if gpu_enable and isinstance(x, mx.array):
        return np.array(x)
    raise TypeError('Unsupported type for as_numpy: {}'.format(type(x)))


def as_mlx(x):
    """Convert `numpy.ndarray` or scalar to `mlx.core.array`."""
    if isinstance(x, Variable):
        x = x.data
    if not gpu_enable:
        raise Exception('MLX cannot be loaded. Install mlx (`pip install mlx`)!')
    return xp_mlx.array(x)


def eval_arrays(*arrays):
    """Force evaluation of lazy MLX arrays on Apple Silicon GPU."""
    if not gpu_enable:
        return
    mx_list = []
    for a in arrays:
        if isinstance(a, Variable):
            a = a.data
        if isinstance(a, mx.array):
            mx_list.append(a)
        elif isinstance(a, (list, tuple)):
            for item in a:
                if isinstance(item, Variable):
                    item = item.data
                if isinstance(item, mx.array):
                    mx_list.append(item)
    if mx_list:
        mx.eval(*mx_list)
