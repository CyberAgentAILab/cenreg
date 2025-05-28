import numpy as np


def _binary_search_F_single(
    F_lb: np.ndarray,
    F_ub: np.ndarray,
    G_cur: np.ndarray,
    copula,
    target: float,
    EPS: float = 0.00001,
):
    F_cur = (F_lb + F_ub) / 2.0
    if F_ub - F_lb < EPS:
        return F_cur
    u = np.array([[F_cur, G_cur]])
    temp = 1.0 - F_cur - G_cur + copula.cdf(u)
    if temp > target:
        F_lb = F_cur
    else:
        F_ub = F_cur
    return _binary_search_F_single(F_lb, F_ub, G_cur, copula, target)


def _binary_search_G_single(
    F_cur: np.ndarray,
    G_lb: np.ndarray,
    G_ub: np.ndarray,
    copula,
    target: float,
    EPS: float = 0.00001,
):
    G_cur = (G_lb + G_ub) / 2
    if G_ub - G_lb < EPS:
        return G_cur
    u = np.array([[F_cur, G_cur]])
    temp = 1.0 - F_cur - G_cur + copula.cdf(u)
    if temp > target:
        G_lb = G_cur
    else:
        G_ub = G_cur
    return _binary_search_G_single(F_cur, G_lb, G_cur, copula, target)


def estimate(
    observed_times: np.ndarray,
    uncensored: np.ndarray,
    copula,
    weights: np.ndarray = None,
):
    """
    Copula-Graphic estimator.
    This method receives any copula.

    Parameters
    ----------
    observed_times : ndarray (float)
        One-dimensional ndarray containing observed times.

    uncensored : ndarray (bool)
        One-dimensional ndarray containing censored (False)
        or uncensored (True).

    copula : object
        Copula function.

    weights : ndarray (float) or None
        One-dimensional ndarray containing weights.
        If None, all weights are set to 1.

    Returns
    -------
    times : ndarray (float)
        One-dimensional ndarray containing time points.

    values : ndarray (float)
        One-dimensional ndarray containing survival rates.
    """

    if weights is None:
        weights = np.ones_like(observed_times)

    # sort values
    l_zip = list(zip(observed_times, uncensored.astype(bool), weights))
    l_sorted = sorted(l_zip, key=lambda y: (y[0], ~y[1]))
    z, e, w = zip(*l_sorted)
    observed_times = np.array(z)
    uncensored = np.array(e)
    weights = np.array(w)

    # compute
    f = 0.0
    g = 0.0
    times = []
    survival_rates = []
    total_weight = np.sum(weights)
    cum_weight = 0.0
    for i in range(len(observed_times)):
        cum_weight += weights[i]
        target = 1.0 - cum_weight / total_weight
        if uncensored[i]:
            f = _binary_search_F_single(f, 1.0, g, copula, target)
            times.append(observed_times[i])
            survival_rates.append(1.0 - f)
        else:
            g = _binary_search_G_single(f, g, 1.0, copula, target)
    return np.array(times), np.array(survival_rates)


class CopulaGraphicDistribution:
    """
    Distribution class of Copula-Graphic estimator.

    Notes:
        The first observation time must be strictly greater than 0.
        At least one data point is uncensored.
        CDF values can be strictly less than 1.0 after the last uncensored data point.
        Survival rates can be strictly greater than 0.0 after the last uncensored data point.
        Inverse CDF values may not be correct for large quantiles (due to the above notes).
    """

    def __init__(
        self,
        observed_times: np.ndarray,
        uncensored: np.ndarray,
        copula,
        weights: np.ndarray | None = None,
    ):
        """
        Initialization.

        Parameters
        ----------
        observed_times : ndarray (float)
            One-dimensional ndarray containing observed times.

        uncensored : ndarray (bool)
            One-dimensional ndarray containing censored (False)
            or uncensored (True).

        copula : object
            Copula function.

        weights : ndarray (float) or None
            One-dimensional ndarray containing weights.
            If None, all weights are set to 1.
        """

        # estimate survival rates
        self._time_points, self.survival_rates = estimate(
            observed_times, uncensored, copula, weights
        )
        if len(self._time_points) == 0:  # if all data points are censored
            raise ValueError("All data points are censored.")

        # store statistics
        self.last_uncensored_time = self._time_points[-1]
        self.cure_rate = self.survival_rates[-1]

        # add the survival rate 1.0 at the start time
        start_time = np.min(observed_times)
        if start_time >= 0.0:
            start_time = 0.0
        else:
            print("WARNING: minimum observed time is negative {}".format(start_time))
        if start_time < self._time_points[0]:
            self.boundaries = np.append(start_time, self._time_points)
            self.survival_rates = np.append(1.0, self.survival_rates)
        else:
            self.boundaries = self._time_points

        # add the survival rate 0.0 at the last observed time
        if self.cure_rate > 0.0:
            self.boundaries = np.append(self.boundaries, np.max(observed_times))
            self.survival_rates = np.append(self.survival_rates, self.cure_rate)

    def survival_function(self, t: np.ndarray):
        """
        Compute survival function.

        Parameters
        ----------
        t : ndarray (float)
            ndarray of shape [batch_size] containing time points.

        Returns
        -------
        survival_rates : ndarray (float)
            ndarray of shape [batch_size] containing survival rates.
        """

        t = t.reshape(-1)
        idx = np.searchsorted(self.boundaries, t, side="right") - 1

        sr = np.zeros(t.shape[0], dtype=float)
        sr[idx < 0] = 1.0
        mask = (idx >= 0) & (idx < len(self.survival_rates))
        sr[mask] = self.survival_rates[idx[mask]]
        return sr

    def average_cdf(self, t: np.ndarray):
        return self.cdf(t)

    def cdf(self, t: np.ndarray):
        """
        Compute cumulative distribution function.

        Parameters
        ----------
        t : ndarray (float)
            ndarray of shape [batch_size] containing time points.

        Returns
        -------
        cumulative probability : ndarray (float)
            ndarray of shape [batch_size] containing cumulative probability of event occurrence.
        """

        return 1.0 - self.survival_function(t)

    def icdf(self, quantiles: np.ndarray):
        """
        Compute cumulative distribution function.

        Parameters
        ----------
        quantiles : ndarray (float)
            ndarray of shape [batch_size] containing quantiles.

        Returns
        -------
        cumulative probability : ndarray (float)
            ndarray of shape [batch_size] containing cumulative probability of event occurrence.
        """

        p = 1.0 - self.survival_rates
        idx = np.searchsorted(p, quantiles, side="left")
        idx = np.clip(idx, 0, len(self.boundaries) - 1)
        return self.boundaries[idx]
