import numpy as np


class EmpiricalCDF:
    """
    Empirical Cumulative distribution functions (CDF).
    """

    def __init__(
        self, y: np.ndarray | None = None, sample_weight: np.ndarray | None = None
    ):
        """
        CDF initialization.

        Parameters
        -------
        y : np.ndarray
            Array containing observed values.
        sample_weight : np.ndarray
            Weight for each sample
        """

        if y is not None:
            self.set(y, sample_weight)

    def average_cdf(self, y: np.ndarray) -> np.ndarray:
        """
        Compute the average CDF values.

        Parameters
        -------
        y : np.ndarray
            CDF values are computed for values y.

        Returns
        -------
        cdf_values : np.ndarray
            Compute CDF values for each value in y.
        """
        values = self.cdf(y)
        if values.ndim > 1:
            return np.mean(values, 0)
        else:
            return values

    def cdf(self, y: np.ndarray):
        """
        Cumulative distribution function (i.e., inverse of quantile function).

        Parameters
        -------
        y : np.ndarray
            CDF values are computed for values y.

        Returns
        -------
        cdf_values : np.ndarray
            Compute CDF values for each value in y.
            Array shape is equal to the shape of y.
        """

        if self.boundaries is None:
            return np.zeros_like(y, dtype=float)

        idx = np.searchsorted(self.boundaries, y, side="right")
        idx_valid = idx > 0
        if len(self.cdf_values.shape) == 1:
            ret = np.zeros_like(y, dtype=float)
            ret[idx_valid] = self.cdf_values[idx[idx_valid] - 1]
        else:
            ret = np.zeros((self.cdf_values.shape[0], y.shape[0]), dtype=float)
            ret[:, idx_valid] = self.cdf_values[:, idx[idx_valid] - 1]
        return ret

    def icdf(self, quantiles: float | np.ndarray) -> np.ndarray:
        """
        Inverse cumulative distribution function (i.e., quantile function).

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

        idx = np.searchsorted(self.cdf_values, quantiles, side="right")
        idx = np.clip(idx, 0, len(self.boundaries) - 1)
        if len(self.cdf_values.shape) == 1:
            ret = np.zeros((len(quantiles),), dtype=float)
            for i in range(len(quantiles)):
                ret[i] = self.boundaries[idx[i]]
        else:
            raise NotImplementedError("Multi-dimensional CDF is not implemented yet.")
        return ret

    def set(self, y: np.ndarray, sample_weight: np.ndarray | None = None):
        """
        Compute CDF based on y and sample_weight.
        The observed values y are weightd by sample_weight.

        Parameters
        -------
        y : np.ndarray
            One-dimensional ndarray containing observed values.
        sample_weight : np.ndarray
            One or two-dimensional ndarray containing weight for each value.
        """

        if len(y.shape) != 1:
            raise ValueError("y must be one-dimensional array.")

        if y.size == 0:
            self.boundaries = None
            return
        if sample_weight is None:
            sample_weight = np.ones_like(y)
        else:
            assert len(sample_weight.shape) <= 2

        unq, inv = np.unique(y, return_inverse=True)
        self.boundaries = unq
        if len(sample_weight.shape) == 1:
            counts = np.bincount(inv, sample_weight)
            s = sample_weight.sum()
            s = np.clip(s, 0.0001, np.inf)
            self.cdf_values = np.cumsum(counts) / s
            self.cdf_values[-1] = 1.0
        else:
            list_counts = []
            for i in range(sample_weight.shape[0]):
                counts = np.bincount(inv, sample_weight[i, :])
                list_counts.append(counts)
            counts = np.stack(list_counts, 0)
            s = np.sum(sample_weight, axis=1).reshape(-1, 1)
            s = np.clip(s, 0.0001, np.inf)
            self.cdf_values = np.cumsum(counts, axis=1) / s
            self.cdf_values[:, -1] = 1.0
