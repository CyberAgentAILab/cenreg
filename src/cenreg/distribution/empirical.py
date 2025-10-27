import numpy as np


class EmpiricalCDF:
    """
    Empirical Cumulative distribution functions (CDF).
    """

    def __init__(
        self,
        bins: np.ndarray,
        cum_p: np.ndarray,
        side: str = "left",
        confidence_interval: np.ndarray | None = None,
    ):
        """
        Initialization.

        Parameters
        -------
        bins : np.ndarray
            Array containing distinct observed values.
            bins must be strictly increasing.
            bins[0] != -np.inf and bins[-1] != np.inf must hold.
        cum_p : np.ndarray
            Cumulative probability for each bin.
            The probability corresponding to the i-th bin (between bins[i-1] and bins[i]) is computed as cum_p[i+1] - cum_p[i], where we implicitly assume bins[-1] = -inf and bins[len(bins)] = inf.
            cum_p must be non-decreasing, starting from 0.0 to 1.0 (i.e., cum_p[0] == 0.0 and cum_p[-1] == 1.0).
        side : str
            'left' or 'right' indicating the side for CDF step function.
        confidence_interval : np.ndarray | None
            Confidence interval for the empirical CDF.
            If not None, confidence_interval.shape == (len(cum_p), 2) must hold.
        """

        assert len(bins.shape) == 1
        assert len(cum_p.shape) == 1
        assert bins.shape[0] + 1 == cum_p.shape[0]
        assert cum_p[0] == 0.0
        assert cum_p[-1] == 1.0
        assert side in ("left", "right")
        if confidence_interval is not None:
            assert confidence_interval.shape == (len(cum_p), 2)

        self.bins = bins
        self.cum_p = cum_p
        self.side = side
        self.confidence_interval = confidence_interval

    def cdf(self, y: float | np.ndarray):
        """
        Cumulative distribution function (i.e., inverse of quantile function).

        Parameters
        -------
        y : np.ndarray | float
            Values for which the CDF is computed.

        Returns
        -------
        cum_p : np.ndarray
            CDF values for each value in y.
            Array shape is equal to the shape of y.
        """
        if isinstance(y, float):
            y = np.array([y])

        ret = np.zeros((len(y),), dtype=float)
        if self.side == "left":
            ret[y <= self.bins[0]] = 0.0
            ret[y > self.bins[-1]] = 1.0
            mask = (y > self.bins[0]) & (y <= self.bins[-1])
        else:  # self.side == "right"
            ret[y < self.bins[0]] = 0.0
            ret[y >= self.bins[-1]] = 1.0
            mask = (y >= self.bins[0]) & (y < self.bins[-1])
        idx = np.searchsorted(self.bins, y, side=self.side)
        idx = np.clip(idx, 1, len(self.bins) - 1)
        ret[mask] = self.cum_p[idx[mask]]
        return ret

    def icdf(self, quantiles: float | np.ndarray) -> np.ndarray:
        """
        Inverse cumulative distribution function (i.e., quantile function).

        If the input is 0.0, return self.bins[0].
        If the input is 1.0, return self.bins[-1].
        If the input is between 0.0 and 1.0, return the corresponding bin value.

        Note:
        For any alpha in [0.0, 1.0], we usually assume that self.cdf(self.icdf(alpha)) == alpha holds.
        However, the inverse CDF of empirical distribution defined here does not always satisfy this property.

        Parameters
        -------
        quantiles : float | np.ndarray
            Quantiles for which the inverse CDF is computed.

        Returns
        -------
        icdf_values : np.ndarray
            Compute inverse CDF values for each value in quantiles.
            Array shape is equal to the shape of quantiles.
        """

        if isinstance(quantiles, float):
            quantiles = np.array([quantiles])
        if np.any(quantiles < 0.0):
            raise ValueError("quantiles must be non-negative.")
        if np.any(quantiles > 1.0):
            raise ValueError("quantiles must be less than or equal to 1.0.")

        ret = np.zeros((len(quantiles),), dtype=float)
        ret[quantiles <= 0.0] = self.bins[0]
        ret[quantiles >= 1.0] = self.bins[-1]
        mask = (quantiles > 0.0) & (quantiles < 1.0)
        idx = np.searchsorted(self.cum_p, quantiles, side=self.side)
        idx = np.clip(idx - 1, 0, len(self.bins) - 1)
        ret[mask] = self.bins[idx[mask]]
        return ret
