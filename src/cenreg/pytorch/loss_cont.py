import torch

from src.cenreg.pytorch.copula_torch import IndependenceCopula, SurvivalCopula


class NegativeLogLikelihood:
    """
    Negative Log-Likelihood
    """

    def __init__(self, EPS=0.0001):
        self.EPS = EPS

    def loss(
        self,
        F_pred: torch.Tensor,
        observed_times: torch.Tensor,
        events: torch.Tensor = None,
    ) -> torch.Tensor:
        assert len(F_pred.shape) == 2
        assert F_pred.shape[0] == observed_times.shape[0]

        pred_sum = torch.sum(F_pred)
        df = torch.autograd.grad(pred_sum, observed_times, create_graph=True)[0]
        if events is None:
            return -torch.log(df + self.EPS)

        uncensored = events.to("cpu").detach().numpy().copy().astype(bool)
        ret = torch.zeros_like(observed_times)
        ret[uncensored] = -torch.log(df[uncensored] + self.EPS)
        ret[~uncensored] = -torch.log(1.0 - F_pred[~uncensored] + self.EPS)
        return ret


class CopulaNegativeLogLikelihood:
    """
    Negative Log Likelihood with survival copula

    return -log ((dC/dF) (dF/dt))
    """

    def __init__(self, copula=None, survival_copula=None, EPS=0.0001):
        self.EPS = EPS
        if copula is None:
            if survival_copula is None:
                self.survival_copula = IndependenceCopula()
            else:
                self.survival_copula = survival_copula
        else:
            if survival_copula is None:
                self.survival_copula = SurvivalCopula(copula)
            else:
                self.survival_copula = survival_copula
                print("Warning: survival_copula is not None. copula is ignored.")

    def loss(
        self, F_pred: torch.Tensor, observed_times: torch.Tensor, events: torch.Tensor
    ) -> torch.Tensor:
        assert len(F_pred.shape) == 2
        assert F_pred.shape[0] == observed_times.shape[0]

        df = torch.zeros_like(observed_times)
        num_risks = F_pred.shape[1]
        for k in range(num_risks):
            mask = (events == k).detach()
            pred_sum = torch.sum(F_pred[:, k])
            temp = torch.autograd.grad(pred_sum, observed_times, create_graph=True)[0]
            df[mask] = temp[mask]
        log_df = torch.log(df + self.EPS)

        s = 1.0 - F_pred
        if self.survival_copula is None:
            raise NotImplementedError()
            # TODO compute survival copula using normal copula
        else:
            c = self.survival_copula.cdf(s)
        c_sum = torch.sum(c)
        dc = torch.autograd.grad(c_sum, s, create_graph=True)[0]
        events = events.view(-1, 1).detach()
        log_dc = torch.log(torch.gather(dc, 1, events) + self.EPS)
        return -(log_df + log_dc)
