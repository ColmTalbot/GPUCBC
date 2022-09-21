from . import likelihood, waveforms, pn


def disable_cupy():
    import numpy as xp

    likelihood.xp = xp
    waveforms.xp = xp


def enable_cupy():
    try:
        import cupy as xp
    except ImportError:
        import numpy as xp

        print("Cannot import cupy")
    likelihood.xp = xp
    waveforms.xp = xp


def activate_jax():
    from jax.config import config
    from jax import numpy
    config.update("jax_enable_x64", True)  # noqa
    likelihood.np = numpy
    likelihood.xp = numpy
    waveforms.np = numpy
    waveforms.xp = numpy
    pn.np = numpy


def deactivate_jax():
    import numpy
    likelihood.np = numpy
    waveforms.np = numpy
    pn.np = numpy
