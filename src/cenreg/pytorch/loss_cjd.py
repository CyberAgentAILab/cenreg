import torch
import torch.nn.functional as F


class NegativeLogLikelihood:
    def __init__(self, y_bins: torch.Tensor, num_risks: int, EPS: float = 0.0001):
        self.y_bins = y_bins
        self.num_risks = num_risks
        self.EPS = EPS

    def loss(
        self,
        pred: torch.Tensor,
        observed_times: torch.Tensor,
        events: torch.Tensor,
    ) -> torch.Tensor:
        assert len(pred.shape) == 2
        assert len(observed_times.shape) == 1
        assert len(events.shape) == 1
        assert observed_times.max() < self.y_bins[-1], (
            "Observed times exceed y_bins range."
        )
        assert observed_times.min() >= self.y_bins[0], (
            "Observed times below y_bins range."
        )

        events = events.long().view(-1, 1)
        idx = torch.searchsorted(self.y_bins, observed_times.view(-1, 1), right=True)
        idx = (idx - 1) * self.num_risks + events
        p = torch.gather(pred, 1, idx)
        return -torch.log(p + self.EPS).view(-1)


class Brier:
    def __init__(self, y_bins: torch.Tensor, num_risks: int):
        self.y_bins = y_bins
        self.num_cls = (len(y_bins) - 1) * num_risks
        self.num_risks = num_risks

    def loss(
        self,
        pred: torch.Tensor,
        observed_times: torch.Tensor,
        events: torch.Tensor,
    ) -> torch.Tensor:
        assert len(pred.shape) == 2
        assert len(observed_times.shape) == 1
        assert len(events.shape) == 1
        assert observed_times.max() < self.y_bins[-1], (
            "Observed times exceed y_bins range."
        )
        assert observed_times.min() >= self.y_bins[0], (
            "Observed times below y_bins range."
        )

        events = events.long().view(-1, 1)
        idx = torch.searchsorted(self.y_bins, observed_times.view(-1, 1), right=True)
        idx = (idx - 1) * self.num_risks + events

        onehot = F.one_hot(idx.view(-1), num_classes=self.num_cls).float()
        diff = onehot - pred
        return (diff * diff).sum(dim=1)


class RankedProbabilityScore:
    def __init__(self, y_bins: torch.Tensor, num_risks: int):
        self.y_bins = y_bins
        self.num_cls = (len(y_bins) - 1) * num_risks
        self.num_risks = num_risks
        self.triu = torch.triu(
            torch.ones(self.num_cls, self.num_cls, device=y_bins.device)
        )

    def loss(
        self, f_pred: torch.Tensor, observed_times: torch.Tensor, events: torch.Tensor
    ) -> torch.Tensor:
        events = events.long().view(-1, 1)
        idx = torch.searchsorted(self.y_bins, observed_times.view(-1, 1), right=True)
        idx = (idx - 1) * self.num_risks + events
        label = self.triu[idx.view(-1)]

        F_pred = torch.cumsum(f_pred, dim=1)
        diff = label - F_pred
        return (diff * diff).sum(dim=1)
