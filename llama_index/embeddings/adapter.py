"""Embedding adapter model."""

from typing import Any, List, Optional

from llama_index.bridge.pydantic import PrivateAttr

from llama_index.callbacks import CallbackManager
from llama_index.embeddings.base import DEFAULT_EMBED_BATCH_SIZE, BaseEmbedding
import os
import json

from torch import nn, Tensor
import torch
import logging

logger = logging.getLogger(__name__)


# class LinearLayer(nn.Module):
#     """Linear transformation, no bias."""

#     def __init__(self, in_features: int, out_features: int):
#         super(LinearLayer, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.linear = nn.Linear(in_features, out_features, bias=False)
#         # seed with identity matrix
#         # only works for square matrices
#         self.linear.weight.data.copy_(torch.eye(in_features, out_features))

#     def forward(self, embed: Tensor) -> Tensor:
#         """Forward pass (Wv)."""
#         return self.linear(embed)

#     def forward_transpose(self, embed: Tensor) -> Tensor:
#         """Forward pass (W^Tv) = (v^TW)^T."""
#         # return torch.matmul(embed, self.linear.weight)
#         return torch.matmul(self.linear.weight.transpose(0, 1), embed)

#     def get_config_dict(self):
#         return {
#             "in_features": self.in_features,
#             "out_features": self.out_features,
#         }

#     def save(self, output_path: str) -> None:
#         """Save model."""
#         os.makedirs(output_path, exist_ok=True)
#         with open(os.path.join(output_path, "config.json"), "w") as fOut:
#             json.dump(self.get_config_dict(), fOut)
#         torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

#     @staticmethod
#     def load(input_path: str) -> "LinearLayer":
#         """Load model."""
#         with open(os.path.join(input_path, "config.json")) as fIn:
#             config = json.load(fIn)
#         model = LinearLayer(**config)
#         model.load_state_dict(
#             torch.load(
#                 os.path.join(input_path, "pytorch_model.bin"),
#                 map_location=torch.device("cpu"),
#             )
#         )
#         return model


class LinearLayer(nn.Module):
    """Linear transformation, no bias."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
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

    def get_config_dict(self):
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


class LinearAdapterEmbeddingModel(BaseEmbedding):
    """Linear adapter for any embedding model."""

    _base_embed_model: BaseEmbedding = PrivateAttr()
    _adapter_path: str = PrivateAttr()
    _adapter: LinearLayer = PrivateAttr()
    _transform_query: bool = PrivateAttr()
    _device: Optional[str] = PrivateAttr()
    _target_device: Any = PrivateAttr()

    def __init__(
        self,
        base_embed_model: BaseEmbedding,
        adapter_path: str,
        transform_query: bool = True,
        device: Optional[str] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """Init params"""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Use pytorch device: {}".format(device))
        self._target_device = torch.device(device)

        self._base_embed_model = base_embed_model
        self._adapter_path = adapter_path

        adapter = LinearLayer.load(adapter_path)
        self._adapter = adapter
        self._adapter.to(self._target_device)

        self._transform_query = transform_query
        super().__init__(
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            model_name=f"Adapter for {base_embed_model.model_name}",
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "LinearAdapterEmbeddingModel"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        query_embedding = self._base_embed_model._get_query_embedding(query)
        if self._transform_query:
            query_embedding_t = torch.tensor(query_embedding).to(self._target_device)
            query_embedding_t = self._adapter.forward(query_embedding_t)
            query_embedding = query_embedding_t.tolist()

        return query_embedding

    async def _aget_query_embedding(self, query: str) -> List[float]:
        query_embedding = await self._base_embed_model._aget_query_embedding(query)
        if self._transform_query:
            query_embedding_t = torch.tensor(query_embedding).to(self._target_device)
            query_embedding_t = self._adapter.forward(query_embedding_t)
            query_embedding = query_embedding_t.tolist()

        return query_embedding

    def _get_text_embedding(self, text: str) -> List[float]:
        text_embedding = self._base_embed_model._get_text_embedding(text)

        return text_embedding

    async def _aget_text_embedding(self, text: str) -> List[float]:
        text_embedding = await self._base_embed_model._aget_text_embedding(text)

        return text_embedding
