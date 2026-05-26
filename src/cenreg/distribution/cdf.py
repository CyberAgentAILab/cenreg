import numpy as np

from typing import Literal
from cenreg.distribution.interpolate import linear_interpolation


class CumulativeDist:
    """
    Cumulative distribution function.
    """

    def __init__(
        self,
        b: np.ndarray,
        p: np.ndarray | None = None,
        cum_p: np.ndarray | None = None,
        interpolate: Literal["linear", "left", "right"] = "linear",
        confidence_interval: np.ndarray | None = None,
    ):
        """
        Initialization.

        Parameters
        -------
        b : np.ndarray
            Array containing boundaries of bins.
            b must be strictly increasing, and b[0] != -np.inf and b[-1] != np.inf must hold.
        p : np.ndarray | None
            Probability distribution for each bin.
            p must be one-dimensional or two-dimensional.
        cum_p : np.ndarray | None
            Cumulative probability distribution.
            cum_p must be one-dimensional or two-dimensional.
            If both p and cum_p are given, cum_p is used.
        interpolate :  Literal["linear", "left", "right"]
            'linear', 'left', or 'right' indicating the interpolation method.
            If 'linear' is set, linear interpolation is used.
            If 'left' is set, the CDF value at the left edge of each bin is used.
            If 'right' is set, the CDF value at the right edge of each bin is used.
        confidence_interval : np.ndarray | None
            Confidence interval for the empirical CDF.
            If not None, confidence_interval.shape == (len(cum_p), 2) must hold.
        """

        b = np.array(b)
        assert len(b.shape) == 1
        if cum_p is not None:
            cum_p = np.array(cum_p)
            assert len(cum_p.shape) <= 2
            if len(cum_p.shape) == 1:
                assert b.shape[0] - 1 == cum_p.shape[0]
                assert np.all(np.diff(cum_p) >= 0.0), "cum_p must be non-decreasing"
            else:
                assert b.shape[0] - 1 == cum_p.shape[1]
                assert np.all(np.diff(cum_p, axis=1) >= 0.0), "cum_p must be non-decreasing"
        elif p is not None:
            p = np.array(p)
            assert len(p.shape) <= 2
            assert np.all(p >= 0.0)
            if p.ndim == 1:
                assert b.shape[0] - 1 == p.shape[0]
                cum_p = np.cumsum(p)
            else:
                assert b.shape[0] - 1 == p.shape[1]
                cum_p = np.cumsum(p, axis=1)
        else:
            raise ValueError("Either p or cum_p must be provided.")
        assert interpolate in ["linear", "left", "right"]
        if confidence_interval is not None:
            assert confidence_interval.shape == (len(cum_p), 2)

        self.b = b
        self.cum_p = cum_p
        self.interpolate = interpolate
        self.confidence_interval = confidence_interval

    def cdf(self, y: int | float | np.ndarray):
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
        """
        if isinstance(y, int | float):
            y = np.array([y])

        if self.cum_p.ndim == 1:
            assert y.ndim == 1
        elif self.cum_p.ndim == 2:
            assert y.ndim == 1 or y.ndim == 2
        else:
            raise ValueError("cum_p must be one-dimensional or two-dimensional.")

        if self.cum_p.ndim == 2 and y.ndim == 1:
            y = np.tile(y, (self.cum_p.shape[0], 1))

        if self.interpolate == "linear":
            # linear interpolation implementation
            ret = np.zeros_like(y)
            mask_low = y <= self.b[0]
            ret[mask_low] = 0.0
            mask_high = y > self.b[-1]
            ret[mask_high] = 1.0
            mask = np.logical_not(mask_low | mask_high)
            if self.cum_p.ndim == 1:
                cum_p = np.append([0.0], self.cum_p)
                ret[mask] = linear_interpolation(self.b, cum_p, y[mask])
            else:  # self.cum_p.ndim == 2
                zeros = np.zeros((self.cum_p.shape[0], 1))
                temp = np.concatenate((zeros, self.cum_p), axis=1)
                temp2 = linear_interpolation(self.b, temp, y)
                ret[mask] = temp2[mask]
                # ret[mask] = linear_interpolation(boundaries, temp, y)[mask]
            return ret
        else:
            # step function implementation
            idx = np.searchsorted(self.b, y, side=self.interpolate)
            idx = np.clip(idx, 1, len(self.b) - 1)
            ret = self.cum_p[idx - 1]
            if self.interpolate == "left":
                ret[y <= self.b[0]] = 0.0
                ret[y > self.b[-1]] = 1.0
            else:
                ret[y < self.b[0]] = 0.0
                ret[y >= self.b[-1]] = 1.0
            return ret

    def icdf(self, quantiles: float | np.ndarray) -> np.ndarray:
        """
        Inverse cumulative distribution function (i.e., quantile function).

        If the input is 0.0, return self.b[0].
        If the input is 1.0, return self.b[-1].
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
        """

        if isinstance(quantiles, int | float):
            quantiles = np.array([quantiles])
        if np.any(quantiles < 0.0):
            raise ValueError("quantiles must be non-negative.")
        if np.any(quantiles > 1.0):
            raise ValueError("quantiles must be at most 1.0.")
        if self.cum_p.ndim == 2 and quantiles.ndim == 1:
            quantiles = np.tile(quantiles, (self.cum_p.shape[0], 1))

        if self.interpolate == "linear":
            # linear interpolation implementation
            ret = np.zeros_like(quantiles)
            if self.cum_p.ndim == 1:
                mask_low = quantiles <= 0.0
                mask_high = quantiles >= self.cum_p[-1]
            else:
                mask_low = quantiles <= 0.0
                mask_high = quantiles >= self.cum_p[:, -1].reshape(-1, 1)
            ret[mask_low] = self.b[0]
            ret[mask_high] = self.b[-1]
            mask = np.logical_not(mask_low | mask_high)
            if self.cum_p.ndim == 1:
                cum_p = np.append([0.0], self.cum_p)
                ret[mask] = linear_interpolation(cum_p, self.b, quantiles[mask])
            else:
                zeros = np.zeros((self.cum_p.shape[0], 1))
                temp = np.concatenate((zeros, self.cum_p), axis=1)
                ret[mask] = linear_interpolation(temp, self.b, quantiles)[mask]
            return ret
        else:
            # step function implementation
            ret = np.zeros_like(quantiles)
            idx = np.searchsorted(self.cum_p, quantiles, side="left")
            if self.cum_p[0] > 0.0:
                mask_low = idx <= 0
            else:
                idx = np.maximum(idx, 1)
                mask_low = np.full(idx.shape, False)
            ret[mask_low] = self.b[0]
            if self.cum_p[-1] < 1.0:
                mask_high = idx >= len(self.cum_p)
            else:
                idx = np.minimum(idx, len(self.cum_p) - 1)
                mask_high = np.full(idx.shape, False)
            ret[mask_high] = self.b[-1]
            mask = ~(mask_low | mask_high)
            ret[mask] = self.b[idx[mask]]
            return ret

    def survival_function(self, y: float | np.ndarray) -> np.ndarray:
        """
        Survival function (i.e., 1 - CDF).

        Parameters
        -------
        y : np.ndarray | float
            Values for which the survival function is computed.

        Returns
        -------
        survival_values : np.ndarray
            Survival function values for each value in y.
            Array shape is equal to the shape of y.
        """

        cdf_values = self.cdf(y)
        return 1.0 - cdf_values
