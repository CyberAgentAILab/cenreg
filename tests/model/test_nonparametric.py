import numpy as np
import unittest

from cenreg.model.copula_np import IndependenceCopula
from cenreg.model.nonparametric import (
    compute_empirical_cdf,
    kaplan_meier_estimator,
    zheng_klein_estimator,
)


class TestComputeEmpiricalCDF(unittest.TestCase):
    def test1(self):
        a = np.array([1, 2, 2, 4, 3])
        ecdf = compute_empirical_cdf(a)

        self.assertEqual(ecdf.bins.shape, (4,))
        self.assertEqual(ecdf.cum_p.shape, (5,))
        self.assertTrue(np.allclose(ecdf.bins, np.array([1, 2, 3, 4])))
        self.assertTrue(np.allclose(ecdf.cum_p, np.array([0.0, 0.2, 0.6, 0.8, 1.0])))
        self.assertTrue(
            np.allclose(
                ecdf.cdf(np.array([0, 1, 2, 2.5, 3, 4, 5])),
                [0.0, 0.2, 0.6, 0.6, 0.8, 1.0, 1.0],
            )
        )
        self.assertTrue(
            np.allclose(
                ecdf.icdf(np.array([0.0, 0.1, 0.5, 0.7, 0.9, 1.0])),
                [1, 1, 2, 3, 4, 4],
            )
        )

    def test2(self):
        a = np.array([5, 5, 5, 5])
        ecdf = compute_empirical_cdf(a)

        self.assertEqual(ecdf.bins.shape, (1,))
        self.assertEqual(ecdf.cum_p.shape, (2,))
        self.assertTrue(np.allclose(ecdf.bins, np.array([5])))
        self.assertTrue(np.allclose(ecdf.cum_p, np.array([0.0, 1.0])))
        self.assertTrue(
            np.allclose(
                ecdf.cdf(np.array([4, 5, 6])),
                [0.0, 1.0, 1.0],
            )
        )
        self.assertTrue(
            np.allclose(
                ecdf.icdf(np.array([0.0, 0.5, 1.0])),
                [5, 5, 5],
            )
        )


class TestKaplanMeierEstimator(unittest.TestCase):
    def test1(self):
        observed_times = np.array([1, 2, 2, 4, 3])
        uncensored = np.array([True, True, False, True, True])
        ecdf = kaplan_meier_estimator(observed_times, uncensored)

        self.assertEqual(ecdf.confidence_interval, None)
        self.assertEqual(ecdf.bins.shape, (5,))
        self.assertEqual(ecdf.cum_p.shape, (6,))
        self.assertTrue(np.allclose(ecdf.bins, np.array([0, 1, 2, 3, 4])))
        self.assertTrue(
            np.allclose(ecdf.cum_p, np.array([0.0, 0.0, 0.2, 0.4, 0.7, 1.0]))
        )
        self.assertTrue(
            np.allclose(
                ecdf.cdf(np.array([0, 1, 2, 2.5, 3, 4, 5])),
                [0.0, 0.2, 0.4, 0.4, 0.7, 1.0, 1.0],
            )
        )
        self.assertTrue(
            np.allclose(
                ecdf.icdf(np.array([0.0, 0.1, 0.5, 0.7, 0.9, 1.0])),
                [0, 1, 3, 4, 4, 4],
            )
        )

    def test2(self):
        observed_times = np.array([5, 5, 5, 5])
        uncensored = np.array([False, False, True, False])
        ecdf = kaplan_meier_estimator(observed_times, uncensored)

        self.assertEqual(ecdf.confidence_interval, None)
        self.assertEqual(ecdf.bins.shape, (2,))
        self.assertEqual(ecdf.cum_p.shape, (3,))
        self.assertTrue(np.allclose(ecdf.bins, np.array([0, 5])))
        self.assertTrue(np.allclose(ecdf.cum_p, np.array([0.0, 0.0, 1.0])))
        self.assertTrue(
            np.allclose(
                ecdf.cdf(np.array([4, 5, 6])),
                [0.0, 1.0, 1.0],
            )
        )
        self.assertTrue(
            np.allclose(
                ecdf.icdf(np.array([0.0, 0.5, 1.0])),
                [0, 5, 5],
            )
        )

    def test3(self):
        observed_times = np.array([5, 5, 5, 6])
        uncensored = np.array([True, True, True, False])
        ecdf = kaplan_meier_estimator(observed_times, uncensored)

        self.assertEqual(ecdf.confidence_interval, None)
        self.assertEqual(ecdf.bins.shape, (3,))
        self.assertEqual(ecdf.cum_p.shape, (4,))
        self.assertTrue(np.allclose(ecdf.bins, np.array([0, 5, 6])))
        self.assertTrue(np.allclose(ecdf.cum_p, np.array([0.0, 0.0, 0.75, 1.0])))
        self.assertTrue(
            np.allclose(
                ecdf.cdf(np.array([4, 5, 6])),
                [0.0, 0.75, 1.0],
            )
        )
        self.assertTrue(
            np.allclose(
                ecdf.icdf(np.array([0.0, 0.5, 1.0])),
                [0, 5, 6],
            )
        )


class ZhengKleinEstimator(unittest.TestCase):
    def test1(self):
        observed_times = np.array([1, 2, 2, 4, 3])
        uncensored = np.array([True, True, False, True, True])
        copula = IndependenceCopula()
        ecdf = zheng_klein_estimator(observed_times, uncensored, copula)

        self.assertEqual(ecdf.confidence_interval, None)
        self.assertEqual(ecdf.bins.shape, (5,))
        self.assertEqual(ecdf.cum_p.shape, (6,))
        self.assertTrue(np.allclose(ecdf.bins, np.array([0, 1, 2, 3, 4])))
        self.assertTrue(
            np.allclose(ecdf.cum_p, np.array([0.0, 0.0, 0.2, 0.4, 0.7, 1.0]), rtol=0.01)
        )
        self.assertTrue(
            np.allclose(
                ecdf.cdf(np.array([0, 1, 2, 2.5, 3, 4, 5])),
                [0.0, 0.2, 0.4, 0.4, 0.7, 1.0, 1.0],
                rtol=0.01,
            )
        )
        self.assertTrue(
            np.allclose(
                ecdf.icdf(np.array([0.0, 0.1, 0.5, 0.9, 1.0])),
                [0, 1, 3, 4, 4],
            )
        )

    def test2(self):
        observed_times = np.array([5, 5, 5, 5])
        uncensored = np.array([False, False, True, False])
        copula = IndependenceCopula()
        ecdf = zheng_klein_estimator(observed_times, uncensored, copula)

        self.assertEqual(ecdf.confidence_interval, None)
        self.assertEqual(ecdf.bins.shape, (2,))
        self.assertEqual(ecdf.cum_p.shape, (3,))
        self.assertTrue(np.allclose(ecdf.bins, np.array([0, 5])))
        self.assertTrue(np.allclose(ecdf.cum_p, np.array([0.0, 0.0, 1.0]), rtol=0.01))
        self.assertTrue(
            np.allclose(
                ecdf.cdf(np.array([4, 5, 6])),
                [0.0, 1.0, 1.0],
                rtol=0.01,
            )
        )
        self.assertTrue(
            np.allclose(
                ecdf.icdf(np.array([0.0, 0.5, 1.0])),
                [0, 5, 5],
            )
        )

    def test3(self):
        observed_times = np.array([5, 5, 5, 6])
        uncensored = np.array([True, True, True, False])
        copula = IndependenceCopula()
        ecdf = zheng_klein_estimator(observed_times, uncensored, copula)

        self.assertEqual(ecdf.confidence_interval, None)
        self.assertEqual(ecdf.bins.shape, (3,))
        self.assertEqual(ecdf.cum_p.shape, (4,))
        self.assertTrue(np.allclose(ecdf.bins, np.array([0, 5, 6])))
        self.assertTrue(
            np.allclose(ecdf.cum_p, np.array([0.0, 0.0, 0.75, 1.0]), rtol=0.01)
        )
        self.assertTrue(
            np.allclose(
                ecdf.cdf(np.array([4, 5, 6])),
                [0.0, 0.75, 1.0],
                rtol=0.01,
            )
        )
        self.assertTrue(
            np.allclose(
                ecdf.icdf(np.array([0.0, 0.5, 1.0])),
                [0, 5, 6],
            )
        )


if __name__ == "__main__":
    unittest.main()
