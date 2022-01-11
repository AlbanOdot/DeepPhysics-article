import torch
import torch.nn as nn
from typing import List


class FCNN(nn.Module):
    """Class that describe the architecture and behavior of the neural network"""
    neurons_per_layer: int = 0      # number of neurons per layer
    layers: List[nn.Module]         # Ordered list of the network layers
    sequence: nn.Sequential         # Sequential object that reduces code complexity

    def __init__(self, neurons_per_layer: int = 0):
        super(FCNN, self).__init__()
        self.neurons_per_layer = neurons_per_layer

        self.layers = [nn.Linear(in_features=self.neurons_per_layer, out_features=self.neurons_per_layer),
                       nn.PReLU(num_parameters=self.neurons_per_layer),
                       nn.Linear(in_features=self.neurons_per_layer, out_features=self.neurons_per_layer),
                       nn.PReLU(num_parameters=self.neurons_per_layer),
                       nn.Linear(in_features=self.neurons_per_layer, out_features=self.neurons_per_layer),
                       nn.PReLU(num_parameters=self.neurons_per_layer),
                       nn.Linear(in_features=self.neurons_per_layer, out_features=self.neurons_per_layer)]

        self.sequence = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Override nn.Module's forward function.
            Take the NN input and return the predicted tensor."""
        return self.sequence(x)
