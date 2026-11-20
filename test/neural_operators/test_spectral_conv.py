"""Unit tests for 1D Fourier layers and a Burgers IMEX step."""
import math
import unittest

import torch

from labml_nn.neural_operators.spectral_conv import FNO1d, SpectralConv1d, burgers_step


class TestSpectralConv(unittest.TestCase):
    def test_rfft_irfft_roundtrip(self):
        n = 64
        x = torch.arange(n, dtype=torch.float32) * (2 * math.pi / n)
        u = torch.sin(x)
        rec = torch.fft.irfft(torch.fft.rfft(u), n=n)
        self.assertTrue(torch.allclose(rec, u, atol=1e-5))

    def test_mode_truncation_zeros_high_modes(self):
        conv = SpectralConv1d(1, 1, modes=2)
        n = 32
        x = torch.randn(1, 1, n)
        y = conv(x)
        y_ft = torch.fft.rfft(y, dim=-1)
        self.assertTrue(
            torch.allclose(y_ft[:, :, 8], torch.zeros_like(y_ft[:, :, 8]), atol=1e-5))

    def test_fno1d_shape(self):
        model = FNO1d(width=64, modes=16, n_layers=4)
        x = torch.randn(2, 1, 64)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, 1, 64))

    def test_burgers_periodic_sine_decays(self):
        n = 256
        x = torch.arange(n, dtype=torch.float32) * (2 * math.pi / n)
        u = torch.sin(x)
        u_next = burgers_step(u, visc=0.1, dt=0.01)
        energy_0 = (u ** 2).sum()
        energy_1 = (u_next ** 2).sum()
        self.assertLess(energy_1.item(), energy_0.item())


if __name__ == '__main__':
    unittest.main()
