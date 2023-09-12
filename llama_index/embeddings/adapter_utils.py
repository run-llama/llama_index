"""Adapter utils."""

from typing import Dict, Optional, Callable
from abc import abstractmethod

import os
import json

from torch import nn, Tensor
import torch.nn.functional as F
import torch
import logging

logger = logging.getLogger(__name__)


class BaseAdapter(nn.Module):
    """Base adapter.

    Can be subclassed to implement custom adapters.
    To implement a custom adapter, subclass this class and implement the
    following methods:
        - get_config_dict
        - forward
    
    """
    
    @abstractmethod
    def get_config_dict(self) -> Dict:
        """Get config dict."""

    @abstractmethod
    def forward(self, embed: Tensor) -> Tensor:
        """Forward pass."""

    def save(self, output_path: str) -> None:
        """Save model."""
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut)
        torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    @classmethod
    def load(cls, input_path: str) -> "BaseAdapter":
        """Load model."""
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)
        model = cls(**config)
        model.load_state_dict(
            torch.load(
                os.path.join(input_path, "pytorch_model.bin"),
                map_location=torch.device("cpu"),
            )
        )
        return model


class LinearLayer(BaseAdapter):
    """Linear transformation.

    Args:
        in_features (int): Input dimension.
        out_features (int): Output dimension.
        bias (bool): Whether to use bias. Defaults to False.
    
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super(LinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # seed with identity matrix and 0 bias
        # only works for square matrices
        self.linear.weight.data.copy_(torch.eye(in_features, out_features))
        if bias:
            self.linear.bias.data.copy_(torch.zeros(out_features))

    def forward(self, embed: Tensor) -> Tensor:
        """Forward pass (Wv)."""
        return self.linear(embed)

    def get_config_dict(self) -> Dict:
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.bias,
        }


def get_activation_function(name: str):
    """Get activation function.

    Args:
        name (str): Name of activation function.
    
    """
    activations = {
        "relu": F.relu,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "leaky_relu": F.leaky_relu,
        # add more activations here as needed
    }
    if name not in activations:
        raise ValueError(f"Unknown activation function: {name}")
    return activations.get(name)


class TwoLayerNN(BaseAdapter):
    """Two-layer transformation.

    Args:
        in_features (int): Input dimension.
        hidden_features (int): Hidden dimension.
        out_features (int): Output dimension.
        bias (bool): Whether to use bias. Defaults to False.
        activation_fn_str (str): Name of activation function. Defaults to "relu".
    
    """

    def __init__(
        self, 
        in_features: int, 
        hidden_features: int, 
        out_features: int,
        bias: bool = False,
        activation_fn_str: str = "relu",
    ) -> None:
        super(TwoLayerNN, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.bias = bias
        self.activation_fn_str = activation_fn_str

        self.linear1 = nn.Linear(in_features, hidden_features, bias=True)
        self.linear2 = nn.Linear(hidden_features, out_features, bias=True)
        # seed with identity matrix and 0 bias
        # only works for square matrices
        self.linear1.weight.data.copy_(torch.eye(in_features, out_features))
        self.linear2.weight.data.copy_(torch.eye(hidden_features, out_features))
        if bias:
            self.linear1.bias.data.copy_(torch.zeros(out_features))
            self.linear2.bias.data.copy_(torch.zeros(out_features))

        self._activation_function = get_activation_function(activation_fn_str)

    def forward(self, embed: Tensor) -> Tensor:
        """Forward pass (Wv).

        Args:
            embed (Tensor): Input tensor.
        
        """
        output1 = self.linear1(embed)
        output1 = self._activation_function(output1)
        output2 = self.linear2(output1)
        return output2

    def get_config_dict(self) -> Dict:
        """Get config dict."""
        return {
            "in_features": self.in_features,
            "hidden_features": self.hidden_features,
            "out_features": self.out_features,
            "bias": self.bias,
            "activation_fn_str": self.activation_fn_str,
        }
