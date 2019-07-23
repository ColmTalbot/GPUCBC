from . import likelihood, waveforms


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
