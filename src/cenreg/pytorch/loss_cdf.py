import torch
import torch.nn.functional as F

from cenreg.pytorch.distribution import LinearCDF


def negative_log_likelihood(
    dist,
    y: torch.Tensor,
    y_bins: torch.Tensor = None,
    uncensored: torch.Tensor = None,
    EPS: float = 0.0001,
) -> torch.Tensor:
    """
    Compute Negative log-likelihood.

    Parameters
    ----------
    dist: predicted distribution

    y: Tensor of shape [batch_size, 1]

    y_bins: Tensor of shape [num_bins+1]

    uncensored: Tensor of shape [batch_size]

    EPS: float

    Returns
    -------
    loss : Tensor of shape [batch_size]
    """

    if y_bins is None:
        y_bins = dist.boundaries
    if uncensored is not None:
        uncensored = uncensored.bool()

    idx = torch.searchsorted(y_bins, y.view(-1, 1), right=True)
    b_lb = y_bins[idx - 1]
    b_ub = y_bins[idx]
    F_lb = dist.cdf(b_lb.view(-1, 1))
    F_ub = dist.cdf(b_ub.view(-1, 1))
    if uncensored is None:
        F_lb_uncensored = F_lb
        F_ub_uncensored = F_ub
    else:
        uncensored = uncensored.bool()
        F_lb_uncensored = F_lb[uncensored]
        F_ub_uncensored = F_ub[uncensored]

    loss = torch.zeros(y.shape[0], 1, device=y.device)
    pu = F_ub_uncensored - F_lb_uncensored + EPS
    loss[uncensored] = -torch.log(pu)
    if uncensored is not None:
        F_lb_censored = F_lb[~uncensored]
        F_ub_censored = F_ub[~uncensored]
        c = y[~uncensored].view(-1, 1)
        F_c = dist.cdf(c, ~uncensored)
        denominator = torch.clamp(1.0 - F_c, min=EPS)
        w = torch.clamp((F_ub_censored - F_c) / denominator, min=0.0, max=1.0)
        w = w.detach()
        pc1 = F_ub_censored - F_lb_censored + EPS
        loss[~uncensored, :] -= w * torch.log(pc1)
        pc2 = 1.0 - F_ub_censored + EPS
        loss[~uncensored, :] -= (1.0 - w) * torch.log(pc2)
    return loss


class NegativeLogLikelihood:
    """
    Loss class for negative log-likelihood.
    """

    def __init__(self, y_bins: torch.Tensor, apply_cumsum: bool = True):
        self.distribution = LinearCDF(y_bins)
        self.y_bins = y_bins
        self.apply_cumsum = apply_cumsum
        self.EPS = 0.0001

    def loss(
        self,
        pred: torch.Tensor,
        y: torch.Tensor,
        uncensored: torch.Tensor = None,
    ) -> torch.Tensor:
        assert len(pred.shape) == 2
        assert len(y.shape) == 1

        self.distribution.set_knot_values(pred, apply_cumsum=self.apply_cumsum)
        return negative_log_likelihood(
            self.distribution, y, self.y_bins, uncensored, self.EPS
        )


class CNLL_CR:
    """
    Censored Negative Log Likelihood for Competing Risks
    """

    def __init__(self, boundaries: torch.Tensor, num_risks: int):
        self.max_time = boundaries[-1]
        self.list_distribution = []
        for i in range(num_risks):
            self.list_distribution.append(LinearCDF(boundaries))
        self.boundaries = boundaries
        self.EPS = 0.0001

    def loss(
        self, pred: torch.Tensor, observed_times: torch.Tensor, events: torch.Tensor
    ) -> torch.Tensor:
        num_risks = len(self.list_distribution)
        idx = torch.searchsorted(
            self.boundaries, observed_times.view(-1, 1), right=True
        ).view(-1)
        b_lb = self.boundaries[idx - 1]
        b_ub = self.boundaries[idx]

        loss = torch.zeros(observed_times.shape[0], device=observed_times.device)
        for i in range(num_risks):
            dist = self.list_distribution[i]
            dist.set_knot_values(pred[i, :, :])
            F_lb = dist.cdf(b_lb.view(-1, 1))
            F_ub = dist.cdf(b_ub.view(-1, 1))

            uncensored = events == i
            F_lb_uncensored = F_lb[uncensored]
            F_ub_uncensored = F_ub[uncensored]
            pu = torch.clamp(F_ub_uncensored - F_lb_uncensored, min=0.0)
            loss[uncensored] -= torch.log(pu + self.EPS).view(-1)

            F_lb_censored = F_lb[~uncensored]
            F_ub_censored = F_ub[~uncensored]
            c = observed_times[~uncensored].view(-1, 1)
            F_c = dist.cdf(c, ~uncensored)
            denominator = torch.clamp(1.0 - F_c, min=self.EPS)
            w = torch.clamp((F_ub_censored - F_c) / denominator, min=0.0, max=1.0)
            w = w.detach()
            pc1 = F_ub_censored - F_lb_censored + self.EPS
            loss[~uncensored] -= (w * torch.log(pc1)).view(-1)
            pc2 = 1.0 - F_ub_censored + self.EPS
            loss[~uncensored] -= ((1.0 - w) * torch.log(pc2)).view(-1)
        return loss


def brier(
    dist,
    y: torch.Tensor,
    y_bins: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute the Brier score.

    Parameters
    ----------
    dist: predicted distribution

    y: Tensor of shape [batch_size]

    y_bins: Tensor of shape [num_bin+1]

    Returns
    -------
    loss : Tensor of shape [batch_size]
    """

    assert len(y.shape) == 1
    assert len(y_bins.shape) == 1

    if y_bins is None:
        y_bins = dist.boundaries

    idx = torch.searchsorted(y_bins, y.view(-1, 1), right=True)
    F_pred = dist.cdf(y_bins)
    y_pred = F_pred[:, 1:] - F_pred[:, :-1]
    onehot = F.one_hot((idx - 1).view(-1), num_classes=len(y_bins) - 1).float()
    loss = F.mse_loss(y_pred, onehot, reduction="none")
    return loss.sum(dim=1)


class Brier:
    """
    Loss class for the Brier score.
    """

    def __init__(self, y_bins: torch.Tensor, apply_cumsum: bool = True):
        self.distribution = LinearCDF(y_bins)
        self.y_bins = y_bins
        self.apply_cumsum = apply_cumsum

    def loss(
        self,
        pred: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        assert len(pred.shape) == 2
        assert len(y.shape) == 1

        self.distribution.set_knot_values(pred, apply_cumsum=self.apply_cumsum)
        return brier(self.distribution, y, self.y_bins)


def ranked_probability_score(
    dist,
    y: torch.Tensor,
    y_bins: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the ranked probability score.

    Parameters
    ----------
    dist: predicted distribution

    y: Tensor of shape [batch_size]

    y_bins: Tensor of shape [num_bins+1]

    Returns
    -------
    loss : Tensor of shape [batch_size]
    """

    assert len(y.shape) == 1
    assert len(y_bins.shape) == 1

    if y_bins is None:
        y_bins = dist.boundaries

    F_pred = dist.cdf(y_bins[1:-1])
    idx = torch.searchsorted(y_bins, y.view(-1, 1), right=True) - 1
    num_cls = len(y_bins) - 1
    label = torch.triu(torch.ones(num_cls, num_cls, device=y.device))[idx.view(-1)]
    loss = F.mse_loss(F_pred, label[:, :-1], reduction="none")
    return loss.sum(dim=1)


class RankedProbabilityScore:
    """
    Loss class for the ranked probability score.
    """

    def __init__(self, y_bins: torch.Tensor, apply_cumsum: bool = True):
        self.distribution = LinearCDF(y_bins)
        self.y_bins = y_bins
        self.apply_cumsum = apply_cumsum

    def loss(
        self,
        pred: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        assert len(pred.shape) == 2
        assert len(y.shape) == 1

        self.distribution.set_knot_values(pred, apply_cumsum=self.apply_cumsum)
        return ranked_probability_score(self.distribution, y, self.y_bins)
