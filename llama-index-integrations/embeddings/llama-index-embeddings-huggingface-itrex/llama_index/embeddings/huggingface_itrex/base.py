import os
from typing import Any, List, Optional

from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.embeddings.huggingface.utils import format_query, format_text
from transformers import AutoTokenizer, AutoConfig


class ItrexQuantizedBgeEmbedding(BaseEmbedding):
    folder_name: str = Field(description="Folder name to load from.")
    pooling: str = Field(description="Pooling strategy. One of ['cls', 'mean'].")
    max_length: int = Field(description="Maximum length of input.")
    normalize: str = Field(default=True, description="Normalize embeddings or not.")
    query_instruction: Optional[str] = Field(
        description="Instruction to prepend to query text."
    )
    text_instruction: Optional[str] = Field(
        description="Instruction to prepend to text."
    )
    onnx_file_name: Optional[str] = Field(
        description="File name of onnx optimized model which is exported by itrex."
    )

    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _hidden_size: Any = PrivateAttr()

    def __init__(
        self,
        folder_name: str,
        pooling: str = "cls",
        max_length: Optional[int] = None,
        normalize: bool = True,
        query_instruction: Optional[str] = None,
        text_instruction: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        onnx_file_name: Optional[str] = "int8-model.onnx",
    ):
        try:
            from intel_extension_for_transformers.transformers import AutoModel
        except ImportError:
            raise ImportError(
                "Itrex requires the following dependencies; please install with "
                "`pip install optimum[exporters] "
                "optimum-intel neural-compressor intel_extension_for_transformers`"
            )

        from huggingface_hub import hf_hub_download

        onnx_model_path = os.path.join(folder_name, onnx_file_name)
        if not os.path.exists(onnx_model_path):
            onnx_model_path = hf_hub_download(folder_name, filename=onnx_file_name)
        model = AutoModel.from_pretrained(onnx_model_path, use_embedding_runtime=True)
        config = AutoConfig.from_pretrained(folder_name)
        hidden_size = config.hidden_size

        tokenizer = tokenizer or AutoTokenizer.from_pretrained(folder_name)

        if max_length is None:
            try:
                max_length = int(config.max_position_embeddings)
            except Exception:
                raise ValueError(
                    "Unable to find max_length from model config. "
                    "Please provide max_length."
                )
            try:
                max_length = min(max_length, int(tokenizer.model_max_length))
            except Exception as exc:
                print(f"An error occurred while retrieving tokenizer max length: {exc}")

        if pooling not in ["cls", "mean"]:
            raise ValueError(f"Pooling {pooling} not supported.")

        super().__init__(
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            folder_name=folder_name,
            max_length=max_length,
            pooling=pooling,
            normalize=normalize,
            query_instruction=query_instruction,
            text_instruction=text_instruction,
        )
        self._model = model
        self._tokenizer = tokenizer
        self._hidden_size = hidden_size

    @classmethod
    def class_name(cls) -> str:
        return "ItrexQuantizedBgeEmbedding"

    def _mean_pooling(self, last_hidden_state: Any, attention_mask: Any) -> Any:
        """Mean Pooling - Take attention mask into account for correct averaging."""
        import torch

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _cls_pooling(self, last_hidden_state: list) -> Any:
        """Use the CLS token as the pooling token."""
        return last_hidden_state[:, 0]

    def _embed(self, sentences: List[str]) -> List[List[float]]:
        """Embed sentences."""
        encoded_input = self._tokenizer(
            sentences,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        import torch

        engine_input = list(encoded_input.values())
        outputs = self._model.generate(engine_input)
        if "last_hidden_state:0" in outputs:
            last_hidden_state = outputs["last_hidden_state:0"]
        else:
            last_hidden_state = next(iter(outputs.values()))
        last_hidden_state = torch.tensor(last_hidden_state).reshape(
            encoded_input["input_ids"].shape[0],
            encoded_input["input_ids"].shape[1],
            self._hidden_size,
        )
        if self.pooling == "mean":
            emb = self._mean_pooling(last_hidden_state, encoded_input["attention_mask"])
        elif self.pooling == "cls":
            emb = self._cls_pooling(last_hidden_state)
        else:
            raise ValueError("pooling method no supported")

        if self.normalize:
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        query = format_query(query, self.model_name, self.query_instruction)
        return self._embed([query])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding async."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding async."""
        return self._get_text_embedding(text)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        text = format_text(text, self.model_name, self.text_instruction)
        return self._embed([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        texts = [
            format_text(text, self.model_name, self.text_instruction) for text in texts
        ]
        return self._embed(texts)
