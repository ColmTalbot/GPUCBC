import cupy
import numpy as np

from .kernels import i0e_kernel


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

