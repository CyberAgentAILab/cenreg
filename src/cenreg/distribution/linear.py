import numpy as np


def _linear_interpolation(kx: np.ndarray, ky: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Linear interpolation.

    Parameters
    ----------
    kx : np.ndarray
        One-dimensional or two-dimensional array containing the x-coordinates
        of the data points.
    ky : np.ndarray
        One-dimensional or two-dimensional array containing the y-coordinates
        of the data points.
    x : np.ndarray
        One-dimensional or two-dimensional array containing the x-coordinates
        where to evaluate the interpolated values.

    Returns
    -------
    ret : np.ndarray
        One-dimensional or two-dimensional array containing the interpolated values.
    """

    # compute idx and ratio
    if kx.ndim == 1:
        idx = np.searchsorted(kx, x, side="right")
        idx = np.clip(idx, 1, len(kx) - 1)
        lb = kx[idx - 1]
        ub = kx[idx]
    else:
        # @note this part may be improved by using np.apply_along_axis (not verified)
        # idx = np.diag(np.apply_along_axis(np.searchsorted, 1, kx, x.reshape(-1))).reshape(-1,1)
        list_lb = []
        list_ub = []
        list_idx = []
        if x.ndim == 1:
            for i in range(kx.shape[0]):
                idx = np.searchsorted(kx[i], x, side="right")
                idx = np.clip(idx, 1, kx.shape[1] - 1)
                lb = kx[i, idx - 1]
                ub = kx[i, idx]
                list_lb.append(lb.reshape(1, -1))
                list_ub.append(ub.reshape(1, -1))
                list_idx.append(idx.reshape(1, -1))
        else:
            for i in range(kx.shape[0]):
                idx = np.searchsorted(kx[i], x[i], side="right")
                idx = np.clip(idx, 1, kx.shape[1] - 1)
                lb = kx[i, idx - 1]
                ub = kx[i, idx]
                list_lb.append(lb.reshape(1, -1))
                list_ub.append(ub.reshape(1, -1))
                list_idx.append(idx.reshape(1, -1))
        lb = np.concatenate(list_lb, 0)
        ub = np.concatenate(list_ub, 0)
        idx = np.concatenate(list_idx, 0)
    denominator = np.clip(ub - lb, 0.0001, np.inf)
    numerator = np.clip(x - lb, 0.0, ub - lb)
    ratio = numerator / denominator

    # linear interpolation
    if ky.ndim == 1:
        left = ky[idx - 1]
        right = ky[idx]
    elif idx.ndim == 1:
        left = np.take(ky, idx - 1, -1)
        right = np.take(ky, idx, -1)
    else:
        left = np.take_along_axis(ky, idx - 1, -1)
        right = np.take_along_axis(ky, idx, -1)
    return left + ratio * (right - left)


class LinearCDF:
    """
    Cumulative distribution functions (CDF) with linear interpolation.

    CDF values are stored for pre-defined boundaries,
    and the values between the boundaries are linear interpolated.
    """

    def __init__(
        self,
        boundaries: np.ndarray,
        values: np.ndarray = None,
        apply_cumsum: bool = True,
    ):
        """
        Cumulative distribution function initialization.

        Parameters
        ----------
        boundaries : np.ndarray
            One-dimensional array containing the boundaries used to represent CDF.

        values : np.ndarray
            Array containing the CDF values.
            The length of the last dimension must be equal to the length of boundaries.

        apply_cumsum : bool
            If True, then values is assumed to be the probablity distribution functions (PDFs)
            and the cumulative sum of values is computed.
        """

        if len(boundaries.shape) > 1:
            raise ValueError("boundaries must be one-dimensional array")
        if len(boundaries) < 2:
            raise ValueError("the length of boundaries must be at least 2")

        self.boundaries = np.array(boundaries)
        if values is not None:
            self.set_knot_values(values, apply_cumsum=apply_cumsum)

    def average_cdf(
        self, y: np.ndarray, mask: np.ndarray | None = None, add_edge: bool = False
    ) -> np.ndarray:
        """
        Compute the average CDF values.

        Parameters
        ----------
        y : np.ndarray
            CDF values are computed for values y.
            If dimension of y is one, then cdf(y) is computed for all CDFs.
            If dimension of y is two, then cdf(y) is computed for each corresponding CDF.

        mask : np.ndarray
            Mask to compute CDF for a subset of CDFs.

        add_edge : bool
            If True, then the output is ensured to have output[0]=0.0 and output[-1]=1.0.
            If you need output satisfies this condition, it is better to set add_edges to True and remove the first and last elements from y.

        Returns
        -------
        cdf_values : np.ndarray
            Compute CDF values for each value in y.
        """
        values = self.cdf(y, mask, add_edge)
        if values.ndim > 1:
            return np.mean(values, 0)
        else:
            return values

    def cdf(
        self, y: np.ndarray, mask: np.ndarray | None = None, add_edges: bool = False
    ):
        """
        Cumulative distribution function (i.e., inverse of quantile function).

        Parameters
        ----------
        y : np.ndarray
            CDF values are computed for values y.
            If dimension of y is one, then cdf(y) is computed for all CDFs.
            If dimension of y is two, then cdf(y) is computed for each corresponding CDF.
        mask : np.ndarray
            Mask to compute CDF for a subset of CDFs.
            Array must be one-dimensional and its length must be equal to
            the number of CDFs.
        add_edges : bool
            If True, then the output is ensured to have output[0]=0.0 and output[-1]=1.0.
            If you need output satisfies this condition, it is better to set add_edges to True and remove the first and last elements from y.

        Returns
        -------
        cdf_values : np.ndarray
            Compute CDF values for each value in y.
            Array shape is equal to the shape of y.
        """
        if mask is None:
            values = self.cdf_values
        else:
            values = self.cdf_values[mask]
        ret = _linear_interpolation(self.boundaries, values, y)
        if add_edges:
            s = np.array(ret.shape)
            s[-1] = 1
            zeros = np.zeros(s)
            ones = np.ones(s)
            ret = np.concatenate([zeros, ret, ones], -1)
        return ret

    def icdf(
        self,
        alpha: np.ndarray | float,
        mask: np.ndarray | None = None,
        add_edges: bool = False,
    ) -> np.ndarray:
        """
        Quantile function (i.e., inverse of cumulative distribution function).

        Parameters
        ----------
        alpha : np.ndarray or float
            Quantile values are computed for quantile levels alpha.
            If dimension of alpha is one, then icdf(alpha) is computed for all CDFs.
            If dimension of alpha is two, then icdf(alpha) is computed for each corresponding CDF.
        mask : np.ndarray
            Mask to compute CDF for a subset of CDFs.
            Array must be one-dimensional and its length must be equal to
            the number of CDFs.
        add_edges : bool
            If True, then the inverse of the CDF values at quantiles 0.0 and 1.0 are added.
            If you need quantiles at 0.0 and 1.0, it is better to set add_edges to True and remove 0.0 and 1.0 from alpha.

        Returns
        -------
        y : np.ndarray
            Compute y.
            Array shape is equal to the shape of alpha.
        """
        if isinstance(alpha, float):
            alpha = np.full((self.cdf_values.shape[0], 1), alpha)
        elif alpha.ndim == 1:
            if self.cdf_values.ndim > 1:
                alpha = np.tile(alpha, (self.cdf_values.shape[0], 1))
        if mask is None:
            values = self.cdf_values
        else:
            values = self.cdf_values[mask]
        ret = _linear_interpolation(values, self.boundaries, alpha)
        if add_edges:
            if self.cdf_values.ndim == 1:
                first = np.tile(self.cdf_values[0], (ret.shape[0], 1))
                last = np.tile(self.cdf_values[-1], (ret.shape[0], 1))
            else:
                first = self.cdf_values[:, 0].reshape(-1, 1)
                last = self.cdf_values[:, -1].reshape(-1, 1)
            ret = np.concatenate([first, ret, last], 1)
        return ret

    def get_boundary_lengths(self):
        return self.boundaries[1:] - self.boundaries[:-1]

    def set_knot_values(self, values: np.ndarray, apply_cumsum: bool = True):
        """
        Set CDF values.

        Parameters
        ----------
        values : np.ndarray
            Array containing the CDF values.
            The length of the last dimension must be equal to the length of boundaries.

        apply_cumsum : bool
            If True, then values is assumed to be the probablity distribution functions (PDFs) and
            the cumulative sum of values is computed.
        """

        # set values
        if apply_cumsum:
            s = np.array(values.shape)
            s[-1] = 1
            cum_values = np.cumsum(values, axis=-1)
            values = np.concatenate([np.zeros(s), cum_values], -1)
        else:
            values = np.array(values)
        self.cdf_values = values

        # verify dimension
        if values.shape[-1] != len(self.boundaries):
            raise ValueError(
                "The last dimension of values {0} must be equal to the length of boundaries {1}".format(
                    values.shape[-1], len(self.boundaries)
                )
            )


class LinearQuantileFunction:
    """
    Quantile functions with linear interpolation.

    A quantile function is defined by a set of quantile values (qk_values)
    at pre-defined quantile levels (qk_levels).
    The values between quantile values are computed by using linear interpolation.

    If qk_values are two-dimensional array, then each row corresponds
    to a quantile function.
    """

    def __init__(
        self,
        qk_levels: np.ndarray,
        qk_values: np.ndarray | None = None,
        apply_cumsum: bool = True,
    ):
        """
        Quantile function initialization.

        Parameters
        ----------
        qk_levels : np.ndarray
            One-dimensional array containing the positions (in quantile levels)
            of quantile knots in increasing order such that
                qk_levels[0] = 0.0
                qk_levels[-1] = 1.0

        qk_values : np.ndarray
            One or two-dimensional array containing the values of quantile knots.
            If qk_values is two-dimensional array, then
                each row corresponds to a quantile function and
                qk_values[:,j] stores the value of quantile function at qk_levels[j].
                Array shape must be [num_quantile_function, len(qk_levels)].
        """

        self.qk_levels = qk_levels
        self.set_knot_values(qk_values, apply_cumsum=apply_cumsum)

    def average_cdf(
        self, y: np.ndarray, mask: np.ndarray | None = None, add_edge: bool = False
    ):
        values = self.cdf(y, mask, add_edge)
        if values.ndim > 1:
            return np.mean(values, 0)
        else:
            return values

    def cdf(
        self, y: np.ndarray, mask: np.ndarray | None = None, add_edges: bool = False
    ):
        """
        Cumulative distribution function (i.e., inverse of quantile function).

        Parameters
        ----------
        y : np.ndarray
            Quantile levels are computed for quantile values y.
            If dimension of y is one, then cdf(y) is computed for all quantile functions.
            If dimension of y is two, then cdf(y) is computed for each corresponding quantile function.
        mask : np.ndarray
            Mask to compute quantile function for a subset of quantile functions.
            Array must be one-dimensional and its length must be equal to
            the number of quantile functions.
        add_edges : bool
            If True, then the CDF values at the boundaries are added.

        Returns
        -------
        q_levels : np.ndarray
            Compute quantile levels for each value in y.
            Array shape is equal to the shape of y.
        """
        if y.ndim == 1:
            if self.qk_values.ndim > 1:
                y = np.tile(y, (self.qk_values.shape[0], 1))
        if mask is None:
            values = self.qk_values
        else:
            values = self.qk_values[mask]
        ret = _linear_interpolation(values, self.qk_levels, y)
        if add_edges:
            if ret.ndim == 1:
                zero = np.zeros((1))
                one = np.ones((1))
                ret = np.concatenate([zero, ret, one])
            else:
                zeros = np.zeros((ret.shape[0], 1))
                ones = np.ones((ret.shape[0], 1))
                ret = np.concatenate([zeros, ret, ones], 1)
        return ret

    def icdf(
        self,
        alpha: np.ndarray | float,
        mask: np.ndarray | None = None,
        add_edges: bool = False,
    ):
        """
        Quantile function (i.e., inverse of cumulative distribution function).

        Parameters
        ----------
        alpha : np.ndarray or float
            Quantile values are computed for quantile levels alpha.
            If dimension of alpha is one, then quantile values in alpha are computed for all quantile functions.
            If dimension of alpha is two, then quantile values in alpha are computed for each corresponding quantile function.
        mask : np.ndarray
            Mask to compute quantile function for a subset of quantile functions.
            Array must be one-dimensional and its length must be equal to
            the number of quantile functions.
        add_edges : bool
            If True, then the inverse of the CDF values at the boundaries are added.

        Returns
        -------
        y : np.ndarray
            Array shape is equal to the shape of alpha.
        """
        if isinstance(alpha, float):
            alpha = np.full((self.qk_values.shape[0], 1), alpha)
        elif alpha.ndim == 1:
            if self.qk_values.ndim > 1:
                alpha = np.tile(alpha, (self.qk_values.shape[0], 1))
        if mask is None:
            values = self.qk_values
        else:
            values = self.qk_values[mask]
        ret = _linear_interpolation(self.qk_levels, values, alpha)
        if add_edges:
            if ret.ndim == 1:
                first = self.qk_values[0].reshape(-1)
                last = self.qk_values[-1].reshape(-1)
                ret = np.concatenate([first, ret, last])
            else:
                if self.qk_values.ndim == 1:
                    first = np.tile(self.qk_values[0], (ret.shape[0], 1))
                    last = np.tile(self.qk_values[-1], (ret.shape[0], 1))
                else:
                    first = self.qk_values[:, 0].reshape(-1, 1)
                    last = self.qk_values[:, -1].reshape(-1, 1)
                ret = np.concatenate([first, ret, last], 1)
        return ret

    def get_qk_lengths(self):
        return self.qk_levels[1:] - self.qk_levels[:-1]

    def set_knot_values(self, qk_values: np.ndarray, apply_cumsum: bool = True):
        """
        Set values of quantile knots.

        Parameters
        ----------
        qk_values : np.ndarray
            One or two-dimensional array containing the values of quantile knots.
            If qk_values is two-dimensional array, then each row corresponds to a quantile function
            and qk_values[:,j] stores the value of quantile function at qk_levels[j].
            Array shape must be [num_quantile_function, len(qk_levels)].
        apply_cumsum : bool
            If True, then qk_values is assumed to be the differences of quantile values and
            the cumulative sum of qk_values is computed.
        """

        if qk_values is None:
            return

        # set values
        if apply_cumsum:
            if qk_values.ndim == 1:
                cum_values = np.cumsum(qk_values)
                qk_values = np.concatenate([np.array([0.0]), cum_values], 0)
            else:
                cum_values = np.cumsum(qk_values, axis=1)
                zeros = np.zeros((cum_values.shape[0], 1))
                qk_values = np.concatenate([zeros, cum_values], 1)
        self.qk_values = qk_values

        # check validity of values
        if qk_values.ndim == 1:
            if qk_values.shape[0] != len(self.qk_levels):
                raise ValueError("qk_values.shape[0] != len(qk_levels)")
        else:
            if qk_values.shape[1] != len(self.qk_levels):
                raise ValueError("qk_values.shape[1] != len(qk_levels)")
