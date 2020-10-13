from . import likelihood, waveforms


def disable_cupy():
    import numpy
    from scipy.special import i0e
    likelihood.xp = numpy
    likelihood.i0e = i0e
    waveforms.xp = numpy


def enable_cupy():
    try:
        import cupy
        from .cupy_utils import i0e
        likelihood.xp = cupy
        likelihood.i0e = i0e
        waveforms.xp = cupy
    except ImportError:
        print("Cannot import cupy")
        disable_cupy()


enable_cupy()

