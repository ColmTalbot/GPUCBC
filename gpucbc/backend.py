from importlib import import_module


class Backend:

    def __init__(self, numpy, scipy=None):
        try:
            self.np = import_module(numpy)
            if scipy is not None:
                self.special = import_module(f"{scipy}.special")
            else:
                self.special = None
        except ImportError as e:
            raise ImportError(f"Cannot initialize backend for {numpy} {scipy}.\n{e}")

BACKEND = Backend("numpy", "scipy")
