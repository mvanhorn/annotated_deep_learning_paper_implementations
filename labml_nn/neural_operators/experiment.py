"""
---
title: Training a Fourier Neural Operator on Burgers' equation
summary: >
  Code for training a 1D Fourier Neural Operator on a generated viscous Burgers dataset.
---

# Training a [Fourier Neural Operator](spectral_conv.html)

This trains an [FNO](spectral_conv.html) to map the initial field of the
1D viscous Burgers equation to the field at a later time $T$.
The dataset is generated in `init` with a fixed seed: random Fourier
initial conditions on a periodic grid of $1{,}024$ points, evolved for
about $200$ IMEX Euler steps at viscosity $0.1$.
"""

import math

import torch
import torch.utils.data
from labml import tracker, experiment, monit
from labml.configs import BaseConfigs
from labml_nn.helpers.device import DeviceConfigs
from labml_nn.neural_operators.spectral_conv import FNO1d, burgers_step
from torch import nn


def _random_fourier_field(n: int, n_modes: int, generator: torch.Generator) -> torch.Tensor:
    """
    ### Random band-limited periodic field

    $$u(x) = \\sum_{k=1}^{K} a_k k^{-1} \\sin(k x + \\phi_k)$$
    on $x \\in [0, 2\\pi)$.
    """
    k = torch.arange(1, n_modes + 1, dtype=torch.float32)
    amp = torch.randn(n_modes, generator=generator) / k
    phase = torch.rand(n_modes, generator=generator) * 2 * math.pi
    x = torch.arange(n, dtype=torch.float32) * (2 * math.pi / n)
    u = torch.zeros(n)
    for i in range(n_modes):
        u = u + amp[i] * torch.sin(k[i] * x + phase[i])
    return u


class Configs(BaseConfigs):
    """
    ## Configurations
    """
    # Device to train the model on.
    # [`DeviceConfigs`](../helpers/device.html)
    #  picks up an available CUDA device or defaults to CPU.
    device: torch.device = DeviceConfigs()

    # [FNO](spectral_conv.html) model
    model: FNO1d

    # Hidden width
    width: int = 64
    # Fourier modes kept per layer
    modes: int = 16
    # Alias for the hidden channel count
    width_channels: int = 64
    # Number of Fourier residual blocks
    n_layers: int = 4

    # Spatial grid size
    n_grid: int = 1024
    # Samples generated in `init`
    n_samples: int = 32
    # Fourier modes in the random initial condition
    ic_modes: int = 8
    # Viscosity $\\nu$
    visc: float = 0.1
    # IMEX step size
    dt: float = 0.01
    # Number of Burgers steps from $t=0$ to $T$
    n_steps: int = 200
    # RNG seed for the generated dataset
    seed: int = 0

    # Batch size
    batch_size: int = 4
    # Learning rate
    learning_rate: float = 1e-3
    # Number of training epochs
    epochs: int = 4

    # Dataset
    dataset: torch.utils.data.TensorDataset
    # Dataloader
    data_loader: torch.utils.data.DataLoader

    # Loss function
    loss_func = nn.MSELoss()

    # Adam optimizer
    optimizer: torch.optim.Adam

    def init(self):
        # Generate Burgers pairs $(u(x,0), u(x,T))$ with a fixed seed
        generator = torch.Generator().manual_seed(self.seed)
        u0 = []
        uT = []
        for _ in range(self.n_samples):
            field = _random_fourier_field(self.n_grid, self.ic_modes, generator)
            evolved = field.clone()
            for _ in range(self.n_steps):
                evolved = burgers_step(evolved, visc=self.visc, dt=self.dt)
            u0.append(field)
            uT.append(evolved)
        u0 = torch.stack(u0).unsqueeze(1)
        uT = torch.stack(uT).unsqueeze(1)
        self.dataset = torch.utils.data.TensorDataset(u0, uT)

        # Initialize the model
        width = self.width_channels or self.width
        self.model = FNO1d(width=width, modes=self.modes, n_layers=self.n_layers).to(self.device)

        # Create dataloader
        self.data_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size,
                                                       shuffle=True, pin_memory=True)
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        """
        ### Train for an epoch
        """
        for image, target in monit.iterate('Train', self.data_loader):
            # Increment global step
            tracker.add_global_step()
            # Move data to device
            image, target = image.to(self.device), target.to(self.device)

            # Make the gradients zero
            self.optimizer.zero_grad()
            # Predict $u(x, T)$
            pred = self.model(image)
            # Calculate loss
            loss = self.loss_func(pred, target)
            # Compute gradients
            loss.backward()
            # Take an optimization step
            self.optimizer.step()
            # Track the loss
            tracker.save('loss', loss)

    def run(self):
        """
        ### Training loop
        """
        for _ in monit.loop(self.epochs):
            # Train the model
            self.train()
            # New line in the console
            tracker.new_line()


def main():
    # Create experiment
    experiment.create(name='fno_burgers', writers={'screen'})

    # Create configurations
    configs = Configs()

    # Set configurations. You can override the defaults by passing the values in the dictionary.
    experiment.configs(configs, {})

    # Initialize
    configs.init()

    # Set models for saving and loading
    experiment.add_pytorch_models({'model': configs.model})

    # Start and run the training loop
    with experiment.start():
        configs.run()


#
if __name__ == '__main__':
    main()
