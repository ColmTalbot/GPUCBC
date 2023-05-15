import unittest

import bilby
import gpucbc
import jax.numpy as jnp
from jax.scipy import special as jsp


class TestBackend(unittest.TestCase):

    def test_setting_jax(self):
        gpucbc.set_backend("jax")
        self.assertEqual(gpucbc.pn.B.np, jnp)
        self.assertEqual(gpucbc.pn.B.special, jsp)

    def test_unknown_backend_raises_error(self):
        with self.assertRaises(ImportError):
            gpucbc.set_backend("unknown")

    def test_no_scipy_backend(self):
        gpucbc.set_backend("bilby")
        self.assertEqual(gpucbc.pn.B.np, bilby)
        self.assertEqual(gpucbc.pn.B.special, None)
