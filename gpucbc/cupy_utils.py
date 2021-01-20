import cupy
import cupyx
import numpy as np

from ._kernels import i0e_kernel


def _kernel_array_wrapper(func):

    def wrapped_kernel(array, *args, **kwargs):
        array = cupy.atleast_1d(array)
        output = cupy.empty_like(array)
        threads_per_block = min(
            i0e_kernel.max_threads_per_block // 4, array.size
        )
        grid_size = int(np.ceil(array.size / threads_per_block))
        func((grid_size, ), (threads_per_block, ), (array, output))
        return output

    return wrapped_kernel


i0e = _kernel_array_wrapper(i0e_kernel)


def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
    """
    Copied almost verbatim from
    https://github.com/scipy/scipy/blob/v1.5.2/scipy/special/_logsumexp.py#L7-L127
    """
    xp = cupy.get_array_module(a)
    a_max = xp.amax(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~xp.isfinite(a_max)] = 0
    elif not xp.isfinite(a_max):
        a_max = 0

    if b is not None:
        b = xp.asarray(b)
        tmp = b * xp.exp(a - a_max)
    else:
        tmp = xp.exp(a - a_max)

    if xp == np:
        errstate = np.errstate
    else:
        errstate = cupyx.errstate
    # suppress warnings about log of zero
    with errstate(divide='ignore'):
        s = xp.sum(tmp, axis=axis, keepdims=keepdims)
        if return_sign:
            sgn = xp.sign(s)
            s *= sgn  # /= makes more sense but we need zero -> zero
        out = xp.log(s)

    if not keepdims:
        a_max = xp.squeeze(a_max, axis=axis)
    out += a_max

    if return_sign:
        return out, sgn
    else:
        return out

