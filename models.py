import torch
import torch.nn as nn
import torch.nn.functional as F


class RISAN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.f = nn.LazyLinear(1)
        self.rho = nn.Parameter(torch.zeros(1))
        self.beta = 0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.f(x), F.softplus(self.rho)

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (batch_size, num_features)
        Output: (batch_size). -1, 1 -> Classes, 0 -> Rejection.
        """
        f, rho = self.__call__(x)
        return (f > rho + self.beta).flatten().int() - (f < -rho + self.beta).flatten().int()


class KP1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.f = nn.LazyLinear(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (batch_size, num_features)
        Output: (batch_size). -1, 1 -> Classes, 0 -> Rejection.
        """
        out = self.__call__(x)
        preds = torch.argmax(out, dim=1)
        preds = torch.where(preds == 0, -1, preds)
        preds = torch.where(preds == 2, 0, preds)
        return preds
