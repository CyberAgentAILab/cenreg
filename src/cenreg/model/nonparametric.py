import numpy as np
import scipy

from cenreg.distribution.empirical import EmpiricalCDF


def compute_empirical_cdf(
    y: np.ndarray,
    weights: np.ndarray | None = None,
    min_y: float | None = None,
    max_y: float | None = None,
) -> EmpiricalCDF:
    """
    Compute CDF based on y and weight.
    The observed values y are weighted by weight.

    Note:
    The created EmpiricalCDF object ecdf does not satisfy ecdf.cdf(ecdf.icdf(0.0)) == 0.0.
    If this behavior is not desired, please consider setting some min_y such that min_y < y.min().

    Parameters
    -------
    y : np.ndarray
        One-dimensional ndarray containing observed values.
    weights : np.ndarray | None
        One-dimensional ndarray containing non-negative weight for each value in y.
        If None, all weights are set to 1.0.
    min_y : float | None
        Minimum value for the EmpiricalCDF.  If None, min_y is set to y.min().
    max_y : float | None
        Maximum value for the EmpiricalCDF.  If None, max_y is set to y.max().

    Returns
    -------
    empirical_cdf : EmpiricalCDF
        Empirical CDF object.
    """

    # Input validation
    if len(y.shape) != 1:
        raise ValueError("y must be one-dimensional array.")
    if y.size == 0:
        raise ValueError("y must not be empty.")
    if weights is None:
        weights = np.ones_like(y)
    else:
        if len(weights.shape) != 1:
            raise ValueError("weight must be one-dimensional array.")
        if weights.shape[0] != y.shape[0]:
            raise ValueError("weight and y must have the same length.")
        if np.any(weights < 0.0):
            raise ValueError("weight must be non-negative.")
    if min_y is not None:
        assert min_y <= np.min(y), "min_y must be less than or equal to min(y)."
    if max_y is not None:
        assert max_y >= np.max(y), "max_y must be greater than or equal to max(y)."

    # Compute empirical CDF
    bins, inv = np.unique(y, return_inverse=True)
    counts = np.bincount(inv, weights)
    s = weights.sum()
    if s <= 0.0:
        raise ValueError("Sum of weights must be positive.")
    p = np.clip(counts / s, 0.0, 1.0)
    cum_p = np.cumsum(p)
    cum_p = np.append(0.0, cum_p)  # CDF starts from 0.0
    cum_p[-1] = 1.0  # Ensure last value is exactly 1.0
    if min_y is not None and min_y < bins[0]:
        bins = np.append(min_y, bins)
        cum_p = np.append(0.0, cum_p)
    if max_y is not None and max_y > bins[-1]:
        bins = np.append(bins, max_y)
        cum_p = np.append(cum_p, 1.0)
    return EmpiricalCDF(bins=bins, cum_p=cum_p, side="right")


def _create_empirical_cdf(
    time_points: np.ndarray,
    survival_rates: np.ndarray,
    observed_times: np.ndarray,
    min_y: float | None,
    max_y: float | None,
    survival_rates_lb: np.ndarray | None = None,
    survival_rates_ub: np.ndarray | None = None,
):
    assert len(time_points.shape) == 1
    assert len(survival_rates.shape) == 1
    assert time_points.shape[0] + 1 == survival_rates.shape[0]

    # set survival rate 1.0 at min_y
    if min_y is None:
        min_y = 0.0
    else:
        if min_y > np.min(observed_times):
            raise ValueError(
                "WARNING: min_y {} is greater than the minimum observed time {}".format(
                    min_y, np.min(observed_times)
                )
            )
    if min_y < time_points[0]:
        bins = np.append(min_y, time_points)
        survival_rates = np.append(1.0, survival_rates)
        if survival_rates_lb is not None:
            survival_rates_lb = np.append(1.0, survival_rates_lb)
            survival_rates_ub = np.append(1.0, survival_rates_ub)
    else:
        bins = time_points

    # set survival rate 0.0 at max_y
    if max_y is None:
        max_y = np.max(observed_times)
    else:
        if max_y < np.max(observed_times):
            raise ValueError(
                "WARNING: max_y {} is less than the maximum observed time {}".format(
                    max_y, np.max(observed_times)
                )
            )
    if max_y > bins[-1]:
        bins = np.append(bins, max_y)
        survival_rates = np.append(survival_rates, 0.0)
        if survival_rates_lb is not None:
            survival_rates_lb = np.append(survival_rates_lb, 0.0)
            survival_rates_ub = np.append(survival_rates_ub, 0.0)
    else:
        survival_rates[-1] = 0.0

    # return empirical CDF
    ecdf = EmpiricalCDF(bins=bins, cum_p=1.0 - survival_rates, side="right")
    if survival_rates_lb is not None:
        ecdf.confidence_interval = np.concatenate(
            [
                1.0 - survival_rates_ub.reshape(-1, 1),
                1.0 - survival_rates_lb.reshape(-1, 1),
            ],
            axis=1,
        )
    return ecdf


def kaplan_meier_estimator(
    observed_times: np.ndarray,
    uncensored: np.ndarray,
    weights: np.ndarray | None = None,
    conf_level: float | None = None,
    min_y: float | None = None,
    max_y: float | None = None,
):
    """
    Compute Kaplan-Meier estimator.

    Parameters
    ----------
    observed_times : np.ndarray
        Observed times (both censored and uncensored).
    uncensored : np.ndarray
        Indicator for uncensored data (1: uncensored, 0: censored).
    weights : np.ndarray | None
        Weights for each data point.
    conf_level : float
        Significance level for confidence intervals (e.g., 0.95).
    min_y : float | None
        Minimum value for the EmpiricalCDF.  If None, min_y is set to 0.0.
    max_y : float | None
        Maximum value for the EmpiricalCDF.  If None, max_y is set to observed_times.max().

    Returns
    -------
    empirical_cdf : EmpiricalCDF
        Empirical CDF object.
    """

    def _compute_variance(
        survival_rates: np.ndarray,
        num_alive: np.ndarray,
        num_death: np.ndarray,
        alpha: float,
    ):
        """
        Compute confidence interval of the survival function using the exponential Greenwood formula (i.e., log-log transformation).

        Parameters
        ----------
        survival_rates : np.ndarray
            Survival rates at each time point.
        num_alive : np.ndarray
            Number of alive individuals at each time point.
        num_death : np.ndarray
            Number of death individuals at each time point.
        alpha : float
            Significance level for confidence intervals (e.g., 0.05 for 95% confidence interval).
        """

        last_rate_is_zero = survival_rates[-1] == 0.0
        if last_rate_is_zero:
            survival_rates = survival_rates[:-1]
            num_alive = num_alive[:-1]
            num_death = num_death[:-1]
        else:
            survival_rates = survival_rates
            num_alive = num_alive
            num_death = num_death

        s = np.log(survival_rates)
        z = np.log(-s)

        rate = num_death / (num_alive * (num_alive - num_death))
        std = np.sqrt(np.cumsum(rate) / (s * s))
        std *= scipy.stats.norm.ppf(alpha)
        survival_rates_lb = np.exp(-np.exp(z - std))
        survival_rates_ub = np.exp(-np.exp(z + std))

        if last_rate_is_zero:
            survival_rates_lb = np.append(survival_rates_lb, 0.0)
            survival_rates_ub = np.append(survival_rates_ub, 0.0)
        return survival_rates_lb, survival_rates_ub

    assert len(observed_times.shape) == 1
    assert len(uncensored.shape) == 1
    assert observed_times.shape[0] == uncensored.shape[0]
    uncensored = uncensored.astype(int)
    if np.sum(uncensored) == 0:
        raise ValueError("At least one data point must be uncensored.")
    if weights is None:
        weights = np.ones_like(observed_times)
    else:
        assert len(weights.shape) == 1
        assert observed_times.shape[0] == weights.shape[0]

    # sort based on uncensored and observed_times
    temp = np.concatenate(
        [
            observed_times.reshape(-1, 1),
            uncensored.reshape(-1, 1),
            weights.reshape(-1, 1),
        ],
        axis=1,
    )
    temp = temp[np.argsort(temp[:, 1])[::-1]]
    temp = temp[np.argsort(temp[:, 0])]

    # count alive and dead
    num_alive = np.sum(temp[:, 2]) - np.cumsum(temp[:, 2]) + temp[:, 2]
    temp = np.concatenate([temp, num_alive.reshape(-1, 1)], axis=1)
    dead = temp[temp[:, 1] == 1]
    cumsum_death = np.concatenate([[0.0], np.cumsum(dead[:, 2])])
    cut_index = np.concatenate(([True], dead[1:, 0] != dead[:-1, 0], [True])).nonzero()[
        0
    ]
    num_death = cumsum_death[cut_index[1:]] - cumsum_death[cut_index[:-1]]
    num_alive = dead[cut_index[:-1], 3]
    time_points = dead[cut_index[:-1], 0]

    # compute survival rates
    rate = 1.0 - num_death / num_alive
    survival_rates = np.cumprod(rate)
    survival_rates = np.append(1.0, survival_rates)

    # compute variance
    if conf_level is not None:
        if conf_level <= 0.0 or conf_level >= 1.0:
            raise ValueError("conf_level must be in (0.0, 1.0).")
        survival_rates_lb, survival_rates_ub = _compute_variance(
            survival_rates, num_alive, num_death, 1.0 - conf_level
        )
    else:
        survival_rates_lb = None
        survival_rates_ub = None

    return _create_empirical_cdf(
        time_points,
        survival_rates,
        observed_times,
        min_y,
        max_y,
        survival_rates_lb,
        survival_rates_ub,
    )


def zheng_klein_estimator(
    observed_times: np.ndarray,
    uncensored: np.ndarray,
    copula,
    weights: np.ndarray = None,
    min_y: float | None = None,
    max_y: float | None = None,
):
    """
    Compute copula-graphic estimator proposed by Zheng and Klein.
    This method receives any copula.

    Parameters
    ----------
    observed_times : np.ndarray
        Observed times (both censored and uncensored).
    uncensored : np.ndarray
        Indicator for uncensored data (1: uncensored, 0: censored).
    copula : object
        Copula function.
    weights : np.ndarray | None
        Weights for each data point.
    min_y : float | None
        Minimum value for the EmpiricalCDF.  If None, min_y is set to 0.0.
    max_y : float | None
        Maximum value for the EmpiricalCDF.  If None, max_y is set to observed_times.max().

    Returns
    -------
    empirical_cdf : EmpiricalCDF
        Empirical CDF object.
    """

    def _binary_search_F_single(
        F_lb: float, F_ub: float, G_cur: float, copula, target: float, EPS=0.0001
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
        G_lb: float, G_ub: float, F_cur: float, copula, target: float, EPS=0.0001
    ):
        G_cur = (G_lb + G_ub) / 2.0
        if G_ub - G_lb < EPS:
            return G_cur
        u = np.array([[F_cur, G_cur]])
        temp = 1.0 - F_cur - G_cur + copula.cdf(u)
        if temp > target:
            G_lb = G_cur
        else:
            G_ub = G_cur
        return _binary_search_G_single(G_lb, G_ub, F_cur, copula, target)

    assert len(observed_times.shape) == 1
    assert len(uncensored.shape) == 1
    assert observed_times.shape[0] == uncensored.shape[0]
    uncensored = uncensored.astype(int)
    if np.sum(uncensored) == 0:
        raise ValueError("At least one data point must be uncensored.")
    if weights is None:
        weights = np.ones_like(observed_times)
    else:
        assert len(weights.shape) == 1
        assert observed_times.shape[0] == weights.shape[0]

    # sort based on uncensored and observed_times
    temp = np.concatenate(
        [
            observed_times.reshape(-1, 1),
            uncensored.reshape(-1, 1),
            weights.reshape(-1, 1),
        ],
        axis=1,
    )
    temp = temp[np.argsort(temp[:, 1])[::-1]]
    temp = temp[np.argsort(temp[:, 0])]

    # compute
    f = 0.0
    g = 0.0
    times = []
    survival_rates = []
    total_weight = np.sum(weights)
    cum_weight = 0.0
    for i in range(temp.shape[0]):
        cum_weight += temp[i, 2]
        if (
            i + 1 < temp.shape[0]
            and temp[i, 0] == temp[i + 1, 0]
            and temp[i, 1] == temp[i + 1, 1]
        ):
            continue  # skip duplicates
        target = 1.0 - cum_weight / total_weight
        if temp[i, 1] > 0:  # uncensored
            f = _binary_search_F_single(f, 1.0, g, copula, target)
            times.append(temp[i, 0])
            survival_rates.append(1.0 - f)
        else:
            g = _binary_search_G_single(g, 1.0, f, copula, target)
    times = np.array(times)
    survival_rates = np.append(1.0, np.array(survival_rates))

    return _create_empirical_cdf(
        times,
        survival_rates,
        observed_times,
        min_y,
        max_y,
    )
