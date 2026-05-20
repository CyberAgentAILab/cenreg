import unittest

import numpy as np

from cenreg.model.copula_np import IndependenceCopula
from cenreg.model.nonparametric import (
    empirical_cdf_estimator,
    kaplan_meier_estimator,
    li_watkins_yu_estimator,
    turnbull_estimator,
    zheng_klein_estimator,
)


class TestComputeEmpiricalCDF(unittest.TestCase):
    def test1(self):
        a = np.array([1, 2, 2, 4, 3])
        dist = empirical_cdf_estimator(a, y_min=0, y_max=5)

        self.assertEqual(dist.b.shape, (6,))
        self.assertEqual(dist.cum_p.shape, (5,))
        self.assertTrue(np.allclose(dist.b, np.array([0, 1, 2, 3, 4, 5])))
        self.assertTrue(np.allclose(dist.cum_p, np.array([0.0, 0.2, 0.6, 0.8, 1.0])))

        ret = dist.cdf(np.array([0, 1, 2, 2.5, 3, 4, 5]))
        self.assertEqual(ret.shape, (7,))
        self.assertAlmostEqual(ret[0].item(), 0.0)
        self.assertAlmostEqual(ret[1].item(), 0.2)
        self.assertAlmostEqual(ret[2].item(), 0.6)
        self.assertAlmostEqual(ret[3].item(), 0.6)
        self.assertAlmostEqual(ret[4].item(), 0.8)
        self.assertAlmostEqual(ret[5].item(), 1.0)
        self.assertAlmostEqual(ret[6].item(), 1.0)

        ret = dist.icdf(np.array([0.0, 0.1, 0.5, 0.7, 0.9, 1.0]))
        self.assertEqual(ret.shape, (6,))
        self.assertAlmostEqual(ret[0].item(), 1.0)
        self.assertAlmostEqual(ret[1].item(), 1.0)
        self.assertAlmostEqual(ret[2].item(), 2.0)
        self.assertAlmostEqual(ret[3].item(), 3.0)
        self.assertAlmostEqual(ret[4].item(), 4.0)
        self.assertAlmostEqual(ret[5].item(), 4.0)

    def test2(self):
        a = np.array([5, 5, 5, 5])
        dist = empirical_cdf_estimator(a)

        self.assertEqual(dist.b.shape, (3,))
        self.assertEqual(dist.cum_p.shape, (2,))
        self.assertTrue(np.allclose(dist.b, np.array([4.5, 5, 5.5])))
        self.assertTrue(np.allclose(dist.cum_p, np.array([0.0, 1.0])))
        self.assertTrue(
            np.allclose(
                dist.cdf(np.array([4, 5, 6])),
                [0.0, 1.0, 1.0],
            )
        )
        self.assertTrue(
            np.allclose(
                dist.icdf(np.array([0.0, 0.5, 1.0])),
                [5, 5, 5],
            )
        )


class TestKaplanMeierEstimator(unittest.TestCase):
    def test1(self):
        observed_times = np.array([1, 2, 2, 4, 3])
        uncensored = np.array([True, True, False, True, True])
        dist = kaplan_meier_estimator(observed_times, uncensored)

        self.assertEqual(dist.confidence_interval, None)
        self.assertEqual(dist.b.shape, (5,))
        self.assertEqual(dist.cum_p.shape, (4,))
        self.assertTrue(np.allclose(dist.b, np.array([0, 1, 2, 3, 4])))
        self.assertTrue(np.allclose(dist.cum_p, np.array([0.0, 0.2, 0.4, 0.7])))
        self.assertTrue(
            np.allclose(
                dist.cdf(np.array([0, 1, 2, 2.5, 3, 4, 5])),
                [0.0, 0.2, 0.4, 0.4, 0.7, 1.0, 1.0],
            )
        )
        ret = dist.icdf(np.array([0.0, 0.1, 0.5, 0.7, 0.9, 1.0]))
        self.assertEqual(ret.shape, (6,))
        self.assertAlmostEqual(ret[0].item(), 1.0)
        self.assertAlmostEqual(ret[1].item(), 1.0)
        self.assertAlmostEqual(ret[2].item(), 3.0)
        self.assertAlmostEqual(ret[3].item(), 3.0)
        self.assertAlmostEqual(ret[4].item(), 4.0)
        self.assertAlmostEqual(ret[5].item(), 4.0)

    def test2(self):
        observed_times = np.array([5, 5, 5, 5])
        uncensored = np.array([False, False, True, False])
        dist = kaplan_meier_estimator(observed_times, uncensored, y_max=10)

        self.assertEqual(dist.confidence_interval, None)
        self.assertEqual(dist.b.shape, (3,))
        self.assertEqual(dist.cum_p.shape, (2,))
        self.assertTrue(np.allclose(dist.b, np.array([0, 5, 10])))
        self.assertTrue(np.allclose(dist.cum_p, np.array([0.0, 0.25])))
        self.assertTrue(
            np.allclose(
                dist.cdf(np.array([4, 5, 6, 10, 11])),
                [0.0, 0.25, 0.25, 1.0, 1.0],
            )
        )
        self.assertTrue(
            np.allclose(
                dist.icdf(np.array([0.0, 0.5, 1.0])),
                [5, 10.0, 10.0],
            )
        )

    def test3(self):
        observed_times = np.array([5, 5, 5, 6])
        uncensored = np.array([True, True, True, False])
        dist = kaplan_meier_estimator(observed_times, uncensored)

        self.assertEqual(dist.confidence_interval, None)
        self.assertEqual(dist.b.shape, (3,))
        self.assertEqual(dist.cum_p.shape, (2,))
        self.assertTrue(np.allclose(dist.b, np.array([0, 5, 6])))
        self.assertTrue(np.allclose(dist.cum_p, np.array([0.0, 0.75])))
        self.assertTrue(
            np.allclose(
                dist.cdf(np.array([4, 5, 6])),
                [0.0, 0.75, 1.0],
            )
        )
        self.assertTrue(
            np.allclose(
                dist.icdf(np.array([0.0, 0.5, 1.0])),
                [5, 5, 6],
            )
        )


class ZhengKleinEstimator(unittest.TestCase):
    def test1(self):
        observed_times = np.array([1, 2, 2, 4, 3])
        uncensored = np.array([True, True, False, True, True])
        copula = IndependenceCopula()
        dist = zheng_klein_estimator(observed_times, uncensored, copula, y_max=5)

        self.assertEqual(dist.confidence_interval, None)
        self.assertEqual(dist.b.shape, (6,))
        self.assertEqual(dist.cum_p.shape, (5,))
        self.assertTrue(np.allclose(dist.b, np.array([0, 1, 2, 3, 4, 5])))
        self.assertTrue(np.allclose(dist.cum_p, np.array([0.0, 0.2, 0.4, 0.7, 1.0]), rtol=0.01))
        self.assertTrue(
            np.allclose(
                dist.cdf(np.array([0, 1, 2, 2.5, 3, 4, 5])),
                [0.0, 0.2, 0.4, 0.4, 0.7, 1.0, 1.0],
                rtol=0.01,
            )
        )
        ret = dist.icdf(np.array([0.0, 0.1, 0.5, 0.7, 0.9, 1.0]))
        self.assertEqual(ret.shape, (6,))
        self.assertAlmostEqual(ret[0].item(), 1.0)
        self.assertAlmostEqual(ret[1].item(), 1.0)
        self.assertAlmostEqual(ret[2].item(), 3.0)
        self.assertAlmostEqual(ret[3].item(), 3.0)
        self.assertAlmostEqual(ret[4].item(), 4.0)
        self.assertAlmostEqual(ret[5].item(), 5.0)

    def test2(self):
        observed_times = np.array([5, 5, 5, 5])
        uncensored = np.array([False, False, True, False])
        copula = IndependenceCopula()
        dist = zheng_klein_estimator(observed_times, uncensored, copula, y_max=10)

        self.assertEqual(dist.confidence_interval, None)
        self.assertEqual(dist.b.shape, (3,))
        self.assertEqual(dist.cum_p.shape, (2,))
        self.assertTrue(np.allclose(dist.b, np.array([0, 5, 10])))
        self.assertTrue(np.allclose(dist.cum_p, np.array([0.0, 0.25]), rtol=0.01))
        self.assertTrue(
            np.allclose(
                dist.cdf(np.array([4, 5, 6, 10, 11])),
                [0.0, 0.25, 0.25, 1.0, 1.0],
                rtol=0.01,
            )
        )
        self.assertTrue(
            np.allclose(
                dist.icdf(np.array([0.0, 0.5, 1.0])),
                [5, 10.0, 10.0],
            )
        )

    def test3(self):
        observed_times = np.array([5, 5, 5, 6])
        uncensored = np.array([True, True, True, False])
        copula = IndependenceCopula()
        dist = zheng_klein_estimator(observed_times, uncensored, copula)

        self.assertEqual(dist.confidence_interval, None)
        self.assertEqual(dist.b.shape, (3,))
        self.assertEqual(dist.cum_p.shape, (2,))
        self.assertTrue(np.allclose(dist.b, np.array([0, 5, 6])))
        self.assertTrue(np.allclose(dist.cum_p, np.array([0.0, 0.75]), rtol=0.01))
        self.assertTrue(
            np.allclose(
                dist.cdf(np.array([4, 5, 6])),
                [0.0, 0.75, 1.0],
                rtol=0.01,
            )
        )
        self.assertTrue(
            np.allclose(
                dist.icdf(np.array([0.0, 0.5, 1.0])),
                [5, 5, 6],
            )
        )


class TestTurnbullEstimator(unittest.TestCase):
    def test1(self):
        lb = np.array([1, 2, 2, 4, 3])
        ub = np.array([1, 2, np.inf, 4, 3])
        cdf = turnbull_estimator(lb, ub, y_min=0.0, y_max=5.0)

        self.assertEqual(cdf.b.shape, (10,))
        self.assertEqual(cdf.cum_p.shape, (9,))
        self.assertTrue(
            np.allclose(
                cdf.b,
                np.array([0.0, 0.99999, 1, 1.99999, 2, 2.99999, 3, 3.99999, 4, 5.0]),
            )
        )
        self.assertTrue(
            np.allclose(
                cdf.cum_p,
                np.array([0.0, 0.2, 0.2, 0.4, 0.4, 0.7, 0.7, 1.0, 1.0]),
                rtol=0.01,
            )
        )
        self.assertTrue(
            np.allclose(
                cdf.cdf(np.array([0, 1, 2, 2.5, 3, 4, 5])),
                [0.0, 0.2, 0.4, 0.4, 0.7, 1.0, 1.0],
                rtol=0.01,
            )
        )

        ret = cdf.icdf(np.array([0.0, 0.1, 0.5, 0.9, 1.0]))
        self.assertEqual(ret.shape, (5,))
        self.assertAlmostEqual(ret[0].item(), 0.99999)
        self.assertAlmostEqual(ret[1].item(), 0.99999)
        self.assertAlmostEqual(ret[2].item(), 2.99999)
        self.assertAlmostEqual(ret[3].item(), 3.99999)
        self.assertAlmostEqual(ret[4].item(), 4.0)


class TestLiWatkinsYuEstimator(unittest.TestCase):
    def test1(self):
        lb = np.array([1, 2, 2, 4, 3])
        ub = np.array([1, 2, np.inf, 4, 3])
        cdf = li_watkins_yu_estimator(lb, ub, y_min=0.0, y_max=5.0)

        self.assertEqual(cdf.b.shape, (10,))
        self.assertEqual(cdf.cum_p.shape, (9,))
        self.assertTrue(
            np.allclose(
                cdf.b,
                np.array([0.0, 0.99999, 1, 1.99999, 2, 2.99999, 3, 3.99999, 4, 5.0]),
            )
        )
        self.assertTrue(
            np.allclose(
                cdf.cum_p,
                np.array([0.0, 0.2, 0.2, 0.4, 0.4, 0.7, 0.7, 1.0, 1.0]),
                rtol=0.01,
            )
        )
        self.assertTrue(
            np.allclose(
                cdf.cdf(np.array([0, 1, 2, 2.5, 3, 4, 5])),
                [0.0, 0.2, 0.4, 0.4, 0.7, 1.0, 1.0],
                rtol=0.01,
            )
        )

        ret = cdf.icdf(np.array([0.0, 0.1, 0.5, 0.9, 1.0]))
        self.assertEqual(ret.shape, (5,))
        self.assertAlmostEqual(ret[0].item(), 0.99999)
        self.assertAlmostEqual(ret[1].item(), 0.99999)
        self.assertAlmostEqual(ret[2].item(), 2.99999)
        self.assertAlmostEqual(ret[3].item(), 3.99999)
        self.assertAlmostEqual(ret[4].item(), 4.0)


if __name__ == "__main__":
    unittest.main()
