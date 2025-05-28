import numpy as np

import cenreg.model.kaplan_meier as kaplan_meier


def negative_loglikelihood(
    dist,
    observed_times: np.ndarray,
    uncensored: np.ndarray | None = None,
    boundaries: np.ndarray | None = None,
    simplified: bool = True,
    EPS: float = 0.0001,
) -> np.ndarray:
    """
    Compute negative log-likelihood (NLL).

    Parameters
    ----------
    dist: distribution object
        Prediction results.
    observed_times : ndarray
        Observed times (event times or censoring times).
        Array shape is [batch_size].
    uncensored : ndarray(bool) or None
        Indicator (censored (False) or uncensored (True))
        Array shape is [batch_size].
    boundaries : ndarray or None
        Boundaries of the prediction to be evaluated.
        The first element must be at most the smallest observed time.
        The last element must be strictly larger than the largest observed time.
        If None, dist.boundaries is used.
    simplified : bool
        Use simplified version for censored data points.
    EPS : float
        Small positive value for numerical stability.

    Returns
    -------
    NLL : ndarray
        Value of negative log-likelihood.
    """

    if not simplified:
        raise NotImplementedError("Only simplified version is implemented.")

    # set default values
    if uncensored is None:
        uncensored = np.ones(observed_times.shape, dtype=bool)
    else:
        uncensored = uncensored.astype(bool)
    if boundaries is None:
        try:
            boundaries = dist.boundaries
        except AttributeError:
            raise ValueError(
                "boundaries must be provided if pred does not have boundaries."
            )

    # check idx
    idx = np.searchsorted(boundaries, observed_times.reshape(-1, 1), side="right")
    idx = idx.reshape(-1) - 1
    if np.any(idx < 0):
        raise ValueError("observed_times must be at least boundaries[0]")
    if np.any(idx >= len(boundaries) - 1):
        raise ValueError("observed_times must be strictly smaller than boundaries[-1]")

    # compute loss
    loss = np.zeros(observed_times.shape)
    b_lb = boundaries[idx]
    b_ub = boundaries[idx + 1]
    F_lb = dist.cdf(b_lb.reshape(-1, 1)).reshape(-1)
    F_ub = dist.cdf(b_ub.reshape(-1, 1)).reshape(-1)
    loss[uncensored] = -np.log(F_ub[uncensored] - F_lb[uncensored] + EPS)
    last_idx = idx == len(boundaries) - 2
    loss[~uncensored & ~last_idx] = -np.log(1.0 - F_ub[~uncensored & ~last_idx] + EPS)
    loss[~uncensored & last_idx] = -np.log(1.0 - F_lb[~uncensored & last_idx] + EPS)
    return loss


def brier(
    dist,
    y: np.ndarray,
    y_bins: np.ndarray,
) -> np.ndarray:
    """
    Compute the (original) Brier score.

    Parameters
    ----------
    dist: predicted distribution

    y: Array of shape [batch_size]

    y_bins: Array of shape [num_bins+1]

    Returns
    -------
    loss : Array of shape [batch_size]
    """

    assert len(y.shape) == 1
    assert len(y_bins.shape) == 1

    F_pred = dist.cdf(y_bins)
    if len(F_pred.shape) == 1:
        F_pred = F_pred.reshape(1, -1)
    pred = F_pred[:, 1:] - F_pred[:, :-1]
    y = y.reshape(-1, 1)
    idx = np.searchsorted(y_bins, y, side="right") - 1
    onehot = np.identity(len(y_bins) - 1)[idx.reshape(-1)]
    diff = onehot - pred
    return (diff * diff).sum(axis=1)


def ranked_probability_score(
    dist,
    y: np.ndarray,
    y_bins: np.ndarray,
) -> np.ndarray:
    """
    Compute the ranked probability score.

    Parameters
    ----------
    dist: predicted distribution

    y: Array of shape [batch_size]

    y_bins: Array of shape [num_col]

    Returns
    -------
    loss : Array of shape [batch_size]
    """

    assert len(y.shape) == 1
    assert len(y_bins.shape) == 1

    F_pred = dist.cdf(y_bins[1:-1])
    y = y.reshape(-1, 1)
    idx = np.searchsorted(y_bins, y, side="right") - 1
    num_cls = len(y_bins) - 1
    label = np.triu(np.ones((num_cls, num_cls)))[idx.reshape(-1)]
    diff = label[:, :-1] - F_pred
    return (diff * diff).sum(axis=1)


def nll_sc(
    list_dist,
    observed_times: np.ndarray,
    events: np.ndarray,
    survival_copula,
    EPS: float = 0.0001,
):
    """
    Compute Negative Log-Likelihood based on Survival Copula (NLL-SC).
    """

    assert len(observed_times.shape) == 1
    assert len(events.shape) == 1
    assert observed_times.shape[0] == events.shape[0]

    Sl_list = []
    Sr_list = []
    diff = np.zeros(observed_times.shape[0])
    for k in range(len(list_dist)):
        mask = events == k

        b = list_dist[k].boundaries
        idx = np.searchsorted(b, observed_times, side="right")
        idx = np.clip(idx, 1, len(b) - 1)
        left = b[idx - 1]
        right = b[idx]
        diff[mask] = np.clip(right[mask] - left[mask], EPS, None)

        Sl = np.zeros(observed_times.shape[0])
        Sr = np.zeros(observed_times.shape[0])
        Sl[mask] = 1.0 - list_dist[k].cdf(left.reshape(-1, 1)).reshape(-1)[mask]
        Sr[mask] = 1.0 - list_dist[k].cdf(right.reshape(-1, 1)).reshape(-1)[mask]
        S_pred = (
            1.0 - list_dist[k].cdf(observed_times.reshape(-1, 1)).reshape(-1)[~mask]
        )
        Sl[~mask] = S_pred
        Sr[~mask] = S_pred
        Sl_list.append(Sl.reshape(-1, 1))
        Sr_list.append(Sr.reshape(-1, 1))

    Sl = np.concatenate(Sl_list, axis=1)
    Sr = np.concatenate(Sr_list, axis=1)
    sl = survival_copula.cdf(Sl)
    sr = survival_copula.cdf(Sr)
    log_fl = np.log(np.clip((sl - sr) / diff, EPS, None))
    return -log_fl


def km_calibration(
    dist,
    observed_times: np.ndarray,
    uncensored: np.ndarray | None = None,
    boundaries: np.ndarray | None = None,
    EPS: float = 0.0001,
) -> float:
    """
    Compute KM-Calibration

    Parameters
    ----------
    dist: distribution object
        Prediction results.
    observed_times : ndarray
        Observation time (event time or censored time)
        Array shape is [batch_size].
    uncensored : ndarray
        Censored (False) or uncensored (True)
        Array shape is [batch_size].
    boundaries : ndarray
        Boundaries of F_pred
        Each element in z must be STRICTLY smaller than boundaries[-1]
    EPS : float
        Small positive value for numerical stability

    Returns
    -------
    KM-Calibration : float
        A non-negative float number.
    """

    # set default values
    if uncensored is None:
        uncensored = np.ones(observed_times.shape, dtype=bool)
    else:
        uncensored = uncensored.astype(bool)
    if boundaries is None:
        try:
            boundaries = dist.boundaries
        except AttributeError:
            raise ValueError(
                "boundaries must be provided if pred does not have boundaries."
            )

    # compute Kaplan-Meier distribution and prediction distribution
    km = kaplan_meier.KaplanMeierDistribution()
    km.fit(observed_times, uncensored)
    last_idx = np.searchsorted(boundaries, km.last_uncensored_time, side="right") - 1
    F_km = km.cdf(boundaries)
    F_km[-1] = 1.0
    f_km = F_km[1:] - F_km[:-1]
    F_pred = dist.cdf(boundaries.reshape(1, -1))
    f_pred_mean = np.mean(F_pred[:, 1:] - F_pred[:, :-1], 0)

    # compute logarithmic loss for KM valid region
    log_empirical = np.log(f_km[:last_idx] + EPS)
    log_mean_pred = np.log(f_pred_mean[:last_idx] + EPS)
    loss_valid = np.sum(f_km[:last_idx] * (log_empirical - log_mean_pred))

    # compute logarithmic loss for KM invalid region
    sum_empirical = np.sum(f_km[last_idx:])
    log_sum_empirical = np.log(sum_empirical + EPS)
    log_sum_pred = np.log(np.sum(f_pred_mean[last_idx:]) + EPS)
    loss_invalid = sum_empirical * (log_sum_empirical - log_sum_pred)

    return loss_valid + loss_invalid
