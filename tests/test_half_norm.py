"""Tests half norm."""
import unittest
import numpy as np
import bhatta_bound
import sampcomp


class TestHalfNorm(unittest.TestCase):
    """Tests half norm."""

    def test_half_norm(self):
        vec = np.array([1, 2, 3])
        self.assertAlmostEqual(bhatta_bound.half_norm(vec), (1 + np.sqrt(2) + np.sqrt(3)) ** 2)


class TestBhatta(unittest.TestCase):
    """Tests Bhattacharyya coefficients."""

    def test_bhatta(self):
        dist = [
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([0.2, 0.3, 0.4, 0.1]),
            np.array([0.3, 0.4, 0.1, 0.2]),
            np.array([0.2, 0.3, 0.1, 0.4]),
        ]
        coin_prob = 0.2
        bc2 = bhatta_bound.rho_sq(coin_prob, *dist)
        bc2_bar = bhatta_bound.rho_sq_bar(coin_prob, *dist)
        bc_bar = bhatta_bound.rho_bar(coin_prob, *dist)
        print(bc2, bc2_bar, bc_bar ** 2)

    def test_er_graph(self):
        adj_mat_ter = sampcomp.erdos_renyi_ternary(3, 0.5)
        print(adj_mat_ter)
