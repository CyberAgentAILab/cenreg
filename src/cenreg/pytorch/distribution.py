import torch


def _linear_interpolation(kx, ky, x):
    # compute idx and ratio
    idx = torch.searchsorted(kx, x, right=True)
    if kx.dim() == 1:
        idx = torch.clamp(idx, min=1, max=len(kx) - 1)
        lb = kx[idx - 1]
        ub = kx[idx]
    else:
        idx = torch.clamp(idx, min=1, max=kx.shape[1] - 1)
        lb = torch.gather(kx, 1, idx - 1)
        ub = torch.gather(kx, 1, idx)
    denominator = torch.clamp(ub - lb, min=0.0001)
    ratio = (x - lb) / denominator

    # linear interpolation
    if ky.dim() == 1:
        left = ky[idx - 1]
        right = ky[idx]
    else:
        left = torch.gather(ky, 1, idx - 1)
        right = torch.gather(ky, 1, idx)
    return torch.lerp(left, right, ratio)


class LinearCDF:
    """
    Distribution functions with linear interpolation.

    A distribution function is represented as a discrete cumulative
    distribution function (CDF) at pre-defined quantile levels (boundaries).
    The values between probabilities are computed by using linear interpolation.

    If qk_values are two-dimensional tensor, then each row corresponds to a CDF.
    """

    def __init__(
        self,
        boundaries: torch.Tensor,
        values: torch.Tensor = None,
        apply_cumsum: bool = True,
    ):
        """
        Distribution function initialization.

        Parameters
        ----------
        boundaries : Tensor
            One-dimensional tensor containing the boundaries used to represent CDF.

        values : Tensor
            One or two-dimensional tensor containing the values of CDFs or PDFs.
            If cdf_values is two-dimensional tensor, then
                Each row corresponds to a CDF and
                Tensor shape must be [num_CDF, len(boundaries)].
                If apply_cumsum is True, then
                    cdf_values[:,j] stores the value of PDF at boundaries[j].
                If apply_cumsum is False, then
                    cdf_values[:,j] stores the value of CDF at boundaries[j].
        """

        self.boundaries = boundaries
        self.set_knot_values(values, apply_cumsum=apply_cumsum)

    def average_cdf(self, y, mask=None, add_edge=False):
        values = self.cdf(y, mask, add_edge)
        if values.dim() > 1:
            return torch.mean(values, 0)
        else:
            return values

    def cdf(self, y, mask=None, add_edges=False):
        """
        Cumulative distribution function (i.e., inverse of quantile function).

        Parameters
        ----------
        y : Tensor
            CDF values are computed for values y.
            If dimension of y is one, then cdf(y) is computed for all CDFs.
            If dimension of y is two, then cdf(y) is computed for each corresponding CDF.
        mask : Tensor
            Mask to compute CDF for a subset of CDFs.
            Tensor must be one-dimensional and its length must be equal to
            the number of CDFs.
        add_edges : bool
            If True, then the CDF values at the boundaries are added.

        Returns
        -------
        cdf_values : Tensor
            Compute CDF values for each value in y.
            Tensor shape is equal to the shape of y.
        """
        if y.dim() == 1:
            if self.cdf_values.dim() > 1:
                y = torch.tile(y, (self.cdf_values.shape[0], 1))
        if mask is None:
            values = self.cdf_values
        else:
            values = self.cdf_values[mask]
        ret = _linear_interpolation(self.boundaries, values, y)
        if add_edges:
            if ret.dim() == 1:
                zero = torch.zeros(1, device=ret.device)
                one = torch.ones(1, device=ret.device)
                ret = torch.cat([zero, ret, one])
            else:
                zeros = torch.zeros(ret.shape[0], 1, device=ret.device)
                ones = torch.ones(ret.shape[0], 1, device=ret.device)
                ret = torch.cat([zeros, ret, ones], 1)
        return ret

    def icdf(self, alpha, mask=None, add_edges=False):
        """
        Quantile function (i.e., inverse of cumulative distribution function).

        Parameters
        ----------
        alpha : Tensor
            Quantile values are computed for quantile levels alpha.
            If dimension of alpha is one, then icdf(alpha) is computed for all CDFs.
            If dimension of alpha is two, then icdf(alpha) is computed for each corresponding CDF.
        mask : Tensor
            Mask to compute CDF for a subset of CDFs.
            Tensor must be one-dimensional and its length must be equal to
            the number of CDFs.
        add_edges : bool
            If True, then the inverse of the CDF values at the boundaries are added.

        Returns
        -------
        y : Tensor
            Compute y.
            Tensor shape is equal to the shape of alpha.
        """
        if alpha.dim() == 1:
            if self.cdf_values.dim() > 1:
                alpha = torch.tile(alpha, (self.cdf_values.shape[0], 1))
        if mask is None:
            values = self.cdf_values
        else:
            values = self.cdf_values[mask]
        ret = _linear_interpolation(values, self.boundaries, alpha)
        if add_edges:
            if self.cdf_values.dim() == 1:
                first = torch.tile(self.cdf_values[0], (ret.shape[0], 1))
                last = torch.tile(self.cdf_values[-1], (ret.shape[0], 1))
            else:
                first = self.cdf_values[:, 0].view(-1, 1)
                last = self.cdf_values[:, -1].view(-1, 1)
            ret = torch.cat([first, ret, last], 1)
        return ret

    def get_boundary_lengths(self):
        return self.boundaries[1:] - self.boundaries[:-1]

    def set_knot_values(self, values, apply_cumsum=True):
        """
        Set values of CDF values.

        Parameters
        ----------
        values : Tensor
            One or two-dimensional tensor containing the values of CDFs.
            If cdf_values is two-dimensional tensor, then
                each row corresponds to a CDF and
                cdf_values[:,j] stores the value of CDF at boundries[j].
                Tensor shape must be [num_CDF, len(boundaries)].
        apply_cumsum : bool
            If True, then cdf_values is assumed to be the probablity distribution functions (PDFs) and
            the cumulative sum of cdf_values is computed.
        """

        if values is None:
            return

        # set values
        if apply_cumsum:
            if values.dim() == 1:
                cum_values = torch.cumsum(values)
                values = torch.cat([torch.tensor([0.0]), cum_values], 0)
            else:
                cum_values = torch.cumsum(values, dim=1)
                zeros = torch.zeros(cum_values.shape[0], 1, device=values.device)
                values = torch.cat([zeros, cum_values], 1)
        self.cdf_values = values

        # verify values
        if values.dim() == 1:
            if values.shape[0] != len(self.boundaries):
                raise ValueError("cdf_values.shape[0] != len(boundaries)")
        else:
            if values.shape[1] != len(self.boundaries):
                raise ValueError("cdf_values.shape[1] != len(boundaries)")


class LinearQuantileFunction:
    """
    Quantile functions with linear interpolation.

    A quantile function is defined by a set of quantile values (qk_values)
    at pre-defined quantile levels (qk_levels).
    The values between quantile values are computed by using linear interpolation.

    If qk_values are two-dimensional tensor, then each row corresponds
    to a quantile function.
    """

    def __init__(
        self,
        qk_levels: torch.Tensor,
        qk_values: torch.Tensor = None,
        apply_cumsum: bool = True,
    ):
        """
        Quantile function initialization.

        Parameters
        ----------
        qk_levels : Tensor
            One-dimensional tensor containing the positions (in quantile levels)
            of quantile knots in increasing order such that
                qk_levels[0] = 0.0
                qk_levels[-1] = 1.0

        qk_values : Tensor
            One or two-dimensional tensor containing the values of quantile knots.
            If qk_values is two-dimensional tensor, then
                each row corresponds to a quantile function and
                qk_values[:,j] stores the value of quantile function at qk_levels[j].
                Tensor shape must be [num_quantile_function, len(qk_levels)].
        """

        self.qk_levels = qk_levels
        self.set_knot_values(qk_values, apply_cumsum=apply_cumsum)

    def average_cdf(self, y, mask=None, add_edge=False):
        values = self.cdf(y, mask, add_edge)
        if values.dim() > 1:
            return torch.mean(values, 0)
        else:
            return values

    def cdf(self, y, mask=None, add_edges=False):
        """
        Cumulative distribution function (i.e., inverse of quantile function).

        Parameters
        ----------
        y : Tensor
            Quantile levels are computed for quantile values y.
            If dimension of y is one, then cdf(y) is computed for all quantile functions.
            If dimension of y is two, then cdf(y) is computed for each corresponding quantile function.
        mask : Tensor
            Mask to compute quantile function for a subset of quantile functions.
            Tensor must be one-dimensional and its length must be equal to
            the number of quantile functions.
        add_edges : bool
            If True, then the CDF values at the boundaries are added.

        Returns
        -------
        q_levels : Tensor
            Compute quantile levels for each value in y.
            Tensor shape is equal to the shape of y.
        """
        if y.dim() == 1:
            if self.qk_values.dim() > 1:
                y = torch.tile(y, (self.qk_values.shape[0], 1))
        if mask is None:
            values = self.qk_values
        else:
            values = self.qk_values[mask]
        ret = _linear_interpolation(values, self.qk_levels, y)
        if add_edges:
            if ret.dim() == 1:
                zero = torch.zeros(1, device=ret.device)
                one = torch.ones(1, device=ret.device)
                ret = torch.cat([zero, ret, one])
            else:
                zeros = torch.zeros(ret.shape[0], 1, device=ret.device)
                ones = torch.ones(ret.shape[0], 1, device=ret.device)
                ret = torch.cat([zeros, ret, ones], 1)
        return ret

    def icdf(self, alpha, mask=None, add_edges=False):
        """
        Quantile function (i.e., inverse of cumulative distribution function).

        Parameters
        ----------
        alpha : Tensor
            Quantile values are computed for quantile levels alpha.
            If dimension of alpha is one, then icdf(alpha) is computed for all quantile functions.
            If dimension of alpha is two, then icdf(alpha) is computed for each corresponding quantile function.
        mask : Tensor
            Mask to compute quantile function for a subset of quantile functions.
            Tensor must be one-dimensional and its length must be equal to
            the number of quantile functions.
        add_edges : bool
            If True, then the inverse of the CDF values at the boundaries are added.

        Returns
        -------
        y : Tensor
            Compute y.
            Tensor shape is equal to the shape of alpha.
        """
        if alpha.dim() == 1:
            if self.qk_values.dim() > 1:
                alpha = torch.tile(alpha, (self.qk_values.shape[0], 1))
        if mask is None:
            values = self.qk_values
        else:
            values = self.qk_values[mask]
        ret = _linear_interpolation(self.qk_levels, values, alpha)
        if add_edges:
            if ret.dim() == 1:
                first = self.qk_values[0].view(-1)
                last = self.qk_values[-1].view(-1)
                ret = torch.cat([first, ret, last])
            else:
                if self.qk_values.dim() == 1:
                    first = torch.tile(self.qk_values[0], (ret.shape[0], 1))
                    last = torch.tile(self.qk_values[-1], (ret.shape[0], 1))
                else:
                    first = self.qk_values[:, 0].view(-1, 1)
                    last = self.qk_values[:, -1].view(-1, 1)
                ret = torch.cat([first, ret, last], 1)
        return ret

    def get_qk_lengths(self):
        return self.qk_levels[1:] - self.qk_levels[:-1]

    def set_knot_values(self, qk_values, apply_cumsum=True):
        """
        Set values of quantile knots.

        Parameters
        ----------
        qk_values : Tensor
            One or two-dimensional tensor containing the values of quantile knots.
            If qk_values is two-dimensional tensor, then
                each row corresponds to a quantile function and
                qk_values[:,j] stores the value of quantile function at qk_levels[j].
                Tensor shape must be [num_quantile_function, len(qk_levels)].
        apply_cumsum : bool
            If True, then qk_values is assumed to be the differences of quantile values and
            the cumulative sum of qk_values is computed.
        """

        if qk_values is None:
            return

        # set values
        if apply_cumsum:
            if qk_values.dim() == 1:
                cum_values = torch.cumsum(qk_values)
                qk_values = torch.cat([torch.tensor([0.0]), cum_values], 0)
            else:
                cum_values = torch.cumsum(qk_values, dim=1)
                zeros = torch.zeros(cum_values.shape[0], 1, device=qk_values.device)
                qk_values = torch.cat([zeros, cum_values], 1)
        self.qk_values = qk_values

        # check validity of values
        if qk_values.dim() == 1:
            if qk_values.shape[0] != len(self.qk_levels):
                raise ValueError("qk_values.shape[0] != len(qk_levels)")
        else:
            if qk_values.shape[1] != len(self.qk_levels):
                raise ValueError("qk_values.shape[1] != len(qk_levels)")
