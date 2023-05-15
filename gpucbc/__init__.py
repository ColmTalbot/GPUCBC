from . import backend, likelihood, pn, waveforms


from ._version import __version__


def set_backend(numpy):

    scipy = dict(
        numpy="scipy",
        cupy="cupyx.scipy",
        jax="jax.scipy",
    ).get(numpy, None)
    numpy = dict(jax="jax.numpy").get(numpy, numpy)
    BACKEND = backend.Backend(numpy=numpy, scipy=scipy)
    backend.BACKEND = BACKEND
    likelihood.B = BACKEND
    pn.B = BACKEND
    waveforms.B = BACKEND
