from importlib import import_module


class Backend:

    def __init__(self, numpy, scipy=None):
        self.module = numpy
        try:
            self.np = import_module(numpy)
            if scipy is not None:
                self.special = import_module(f"{scipy}.special")
            else:
                self.special = None
        except ImportError as e:
            raise ImportError(f"Cannot initialize backend for {numpy} {scipy}.\n{e}")

    def to_numpy(self, array):
        if self.module == "numpy":
            return array
        elif self.module == "jax.numpy":
            import numpy as np
            return np.asarray(array)
        elif self.module == "cupy":
            return self.np.asnumpy(array)
        else:
            return array


BACKEND = Backend("numpy", "scipy")
