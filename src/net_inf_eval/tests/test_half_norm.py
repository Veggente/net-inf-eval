"""Tests half norm."""

import unittest

import numpy as np

from net_inf_eval import bhatta_bound


class TestHalfNorm(unittest.TestCase):
    """Tests half norm."""

    def test_half_norm(self):
        vec = np.array([1, 2, 3])
        self.assertAlmostEqual(
            bhatta_bound.half_norm(vec), (1 + np.sqrt(2) + np.sqrt(3)) ** 2
        )
