from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, hidden: Tuple[int, ...] = (64, 32)):
        super().__init__()
        enc_layers = []
        d = in_dim
        for h in hidden:
            enc_layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        enc_layers += [nn.Linear(d, latent_dim)]
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        d = latent_dim
        for h in reversed(hidden):
            dec_layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        dec_layers += [nn.Linear(d, in_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat


class VariationalAutoEncoder(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, hidden: Tuple[int, ...] = (64, 32)):
        super().__init__()
        enc_layers = []
        d = in_dim
        for h in hidden:
            enc_layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.enc_trunk = nn.Sequential(*enc_layers)
        self.mu_head = nn.Linear(d, latent_dim)
        self.logvar_head = nn.Linear(d, latent_dim)

        dec_layers = []
        d = latent_dim
        for h in reversed(hidden):
            dec_layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        dec_layers += [nn.Linear(d, in_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc_trunk(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar
