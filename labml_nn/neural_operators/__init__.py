"""
---
title: Neural Operators
summary: >
 A set of PyTorch implementations/tutorials related to neural operators
---

# Neural Operators

Neural operators learn maps between function spaces.
A Fourier layer mixes modes in frequency space with an FFT, a truncated
complex multiply, and an inverse FFT.

* [Fourier Neural Operator](spectral_conv.html)
* [Training an FNO on Burgers' equation](experiment.html)
"""

from labml_nn.neural_operators.spectral_conv import FNO1d, SpectralConv1d, burgers_step

__all__ = ['SpectralConv1d', 'FNO1d', 'burgers_step']
