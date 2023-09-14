"""Adapter utils."""

from typing import Dict

import os
import json

from torch import nn, Tensor
import torch
import logging

logger = logging.getLogger(__name__)


class LinearLayer(nn.Module):
    """Linear transformation, no bias."""

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

    def save(self, output_path: str) -> None:
        """Save model."""
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut)
        torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    @staticmethod
    def load(input_path: str) -> "LinearLayer":
        """Load model."""
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)
        model = LinearLayer(**config)
        model.load_state_dict(
            torch.load(
                os.path.join(input_path, "pytorch_model.bin"),
                map_location=torch.device("cpu"),
            )
        )
        return model
