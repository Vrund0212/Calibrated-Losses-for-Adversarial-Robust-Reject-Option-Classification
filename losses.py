import torch
import torch.nn as nn
import torch.nn.functional as F


class ShiftedDoubleSigmoid(nn.Module):
    def __init__(self, cost: float, mu: float, beta: float) -> None:
        super().__init__()
        self.cost = cost
        self.mu = mu
        self.beta = beta

    def forward(
        self, f_x: torch.Tensor, rho_x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """f_x, rho_x, y should be (num_samples)"""
        return torch.mean(
            2 * self.cost * torch.sigmoid(-self.mu * (y * f_x - rho_x - self.beta))
            + 2
            * (1 - self.cost)
            * torch.sigmoid(-self.mu * (y * f_x + rho_x - self.beta))
        )


class ShiftedDoubleRamp(nn.Module):
    def __init__(self, cost: float, mu: float, beta: float) -> None:
        super().__init__()
        self.cost = cost
        self.mu = mu
        self.beta = beta

    def forward(
        self, f_x: torch.Tensor, rho_x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """f_x, rho_x, y should be (num_samples)"""
        return torch.mean(
            self.cost
            * torch.minimum(
                torch.full_like(f_x, 1 + self.mu),
                torch.clamp((self.mu + rho_x - y * f_x + self.beta) / self.mu, min=0),
            )
            + (1 - self.cost)
            * torch.minimum(
                torch.full_like(f_x, 1 + self.mu),
                torch.clamp((self.mu - rho_x - y * f_x + self.beta) / self.mu, min=0),
            )
        )


class SurrogateGeneralizedCrossEntropy(nn.Module):
    def __init__(self, cost: float, mu: float) -> None:
        """
        gamma should be between 0 and 1, 0 exclusive
        """
        super().__init__()
        self.mu = mu
        self.cost = cost

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x -> (batch_size, K+1)
        y -> int of (batch_size)
        """
        y = ((y + 1) / 2).long()
        x = F.softmax(x, dim=1)

        t = torch.gather(x, 1, torch.unsqueeze(y, 1))
        l1 = (1 - (t**self.mu)) / self.mu

        t = torch.gather(x, 1, torch.unsqueeze(torch.full_like(y, 2), 1))
        l2 = (1 - (t**self.mu)) / self.mu

        return torch.mean(l1 + (1 - self.cost) * l2)
