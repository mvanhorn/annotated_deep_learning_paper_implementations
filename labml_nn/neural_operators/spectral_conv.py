"""
---
title: Fourier Neural Operator
summary: >
    PyTorch implementation and tutorial of 1D Fourier layers for Burgers PDE surrogates.
---

# Fourier Neural Operator

This is an implementation of a 1D Fourier layer from the paper
[Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895).

A Fourier layer lifts the input to frequency space with a real FFT,
keeps the lowest $k$ modes, mixes those modes with a learned complex
weight tensor, and returns to physical space with an inverse FFT.
A residual $1 \times 1$ convolution runs in parallel.
Stacked layers form a Fourier neural operator (FNO) that maps an
initial field $u(x, 0)$ to a later field $u(x, T)$.

Here is the [training code](experiment.html) for an experiment that
trains an FNO on the 1D viscous Burgers equation.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def burgers_step(u: torch.Tensor, visc: float = 0.1, dt: float = 0.01) -> torch.Tensor:
    """
    ### One IMEX Euler step of 1D viscous Burgers

    The PDE on a periodic interval $[0, 2\\pi]$ is

    $$\\partial_t u + u \\partial_x u = \\nu \\partial_{xx} u$$

    The nonlinear advection is treated explicitly in Fourier space and the
    viscous term implicitly:

    $$\\hat u^{n+1} = \\frac{\\hat u^n - \\Delta t\\, \\mathcal{F}(u u_x)}{1 + \\Delta t\\, \\nu k^2}$$

    where $k = 2\\pi \\cdot \\mathrm{rfftfreq}(n, d=2\\pi/n)$.
    `u` may be `[n]` or `[batch, n]`.
    """
    squeezed = False
    if u.dim() == 1:
        u = u.unsqueeze(0)
        squeezed = True

    n = u.size(-1)
    # Wave numbers for a $2\pi$-periodic grid
    k = 2 * math.pi * torch.fft.rfftfreq(n, d=2 * math.pi / n, device=u.device, dtype=u.dtype)
    u_hat = torch.fft.rfft(u, dim=-1)
    # Spatial derivative via $i k$
    ux = torch.fft.irfft(1j * k * u_hat, n=n, dim=-1)
    nonlin = torch.fft.rfft(u * ux, dim=-1)
    denom = 1.0 + dt * visc * (k ** 2)
    u_hat_new = (u_hat - dt * nonlin) / denom
    u_new = torch.fft.irfft(u_hat_new, n=n, dim=-1)
    if squeezed:
        return u_new.squeeze(0)
    return u_new


class SpectralConv1d(nn.Module):
    """
    ### Spectral convolution

    Transform $x \\in \\mathbb{R}^{B \\times C_{in} \\times n}$ with a real FFT,
    multiply the lowest `modes` frequencies by a learned complex tensor
    $W \\in \\mathbb{C}^{C_{in} \\times C_{out} \\times k}$, and invert.

    Higher modes stay zero, so the layer is a low-pass mixer.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        """
        :param in_channels: is the number of input channels
        :param out_channels: is the number of output channels
        :param modes: is the number of Fourier modes kept
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # Scale like the paper's complex Glorot-style init
        scale = 1.0 / (in_channels * modes)
        w = torch.randn(in_channels, out_channels, modes, 2)
        self.weights = nn.Parameter(scale * torch.view_as_complex(w))

    def forward(self, x: torch.Tensor):
        """
        :param x: is `[batch, in_channels, n]`
        """
        n = x.size(-1)
        # Real FFT along the spatial axis
        x_ft = torch.fft.rfft(x, dim=-1)
        # Keep only the truncated modes; the rest stay $0$
        out_ft = torch.zeros(
            x.size(0), self.out_channels, n // 2 + 1,
            device=x.device, dtype=x_ft.dtype)
        out_ft[:, :, :self.modes] = torch.einsum(
            'bix,iox->box', x_ft[:, :, :self.modes], self.weights)
        return torch.fft.irfft(out_ft, n=n, dim=-1)


class FNO1d(nn.Module):
    """
    ### 1D Fourier neural operator

    Lift the scalar field to `width` channels with a $1 \\times 1$ convolution,
    stack `n_layers` Fourier residual blocks with GELU, then project back.
    """

    def __init__(self, width: int = 64, modes: int = 16, n_layers: int = 4,
                 in_channels: int = 1, out_channels: int = 1):
        """
        :param width: is the hidden channel count
        :param modes: is the number of Fourier modes per layer
        :param n_layers: is the number of Fourier residual blocks
        :param in_channels: is the number of input field channels
        :param out_channels: is the number of output field channels
        """
        super().__init__()
        self.width = width
        self.modes = modes
        self.n_layers = n_layers
        # Lift, mix, project
        self.lift = nn.Conv1d(in_channels, width, 1)
        self.spectral = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        self.pointwise = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.project = nn.Conv1d(width, out_channels, 1)

    def forward(self, x: torch.Tensor):
        """
        :param x: is `[batch, in_channels, n]`
        """
        x = self.lift(x)
        for spec, local in zip(self.spectral, self.pointwise):
            x = F.gelu(spec(x) + local(x))
        return self.project(x)
