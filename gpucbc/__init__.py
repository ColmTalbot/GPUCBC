from . import likelihood, pn, waveforms
from .backend import Backend


def set_backend(numpy):

    scipy = dict(
        numpy="scipy",
        cupy="cupyx.scipy",
        jax="jax.scipy",
    ).get(numpy, None)
    numpy = dict(jax="jax.numpy").get(numpy, numpy)
    BACKEND = Backend(numpy=numpy, scipy=scipy)
    likelihood.BACKEND = BACKEND
    pn.BACKEND = BACKEND
    waveforms.BACKEND = BACKEND
