import unittest

import numpy as np
import scipy.sparse as sp

from main import asm, atm, msh


class FemMiniTest(unittest.TestCase):
    def test_mesh_counts(self):
        p, t = msh(3)
        self.assertEqual(p.shape, (16, 2))
        self.assertEqual(t.shape, (18, 3))

    def test_neumann_constant_kernel(self):
        p, t, k, m, mv = asm(4)
        one = np.ones(len(p))
        self.assertLess(np.linalg.norm(k @ one), 1e-12)
        self.assertAlmostEqual(float(mv.sum()), 1.0, places=12)
        self.assertEqual(m.shape, k.shape)

    def test_atm_small_spd(self):
        a = sp.csr_matrix(np.array([[4.0, -1.0], [-1.0, 3.0]]))
        b = np.array([1.0, 2.0])
        x, st = atm(a, b, tol=1e-12, imax=500)
        self.assertLess(st["res"], 1e-10)
        self.assertTrue(np.allclose(a @ x, b, atol=1e-10))


if __name__ == "__main__":
    unittest.main()
