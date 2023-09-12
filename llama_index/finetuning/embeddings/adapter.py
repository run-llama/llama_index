"""Sentence Transformer Finetuning Engine."""

from llama_index.embeddings.base import BaseEmbedding

from typing import Any, List, Optional, Tuple, cast

from llama_index.finetuning.types import BaseEmbeddingFinetuneEngine
from llama_index.finetuning.embeddings.common import EmbeddingQAFinetuneDataset
from llama_index.embeddings.adapter import LinearAdapterEmbeddingModel
import logging

logger = logging.getLogger(__name__)


class EmbeddingAdapterFinetuneEngine(BaseEmbeddingFinetuneEngine):
    """Embedding adapter finetune engine.

    Args:
        dataset (EmbeddingQAFinetuneDataset): Dataset to finetune on.
        embed_model (BaseEmbedding): Embedding model to finetune.
        batch_size (Optional[int]): Batch size. Defaults to 10.
        epochs (Optional[int]): Number of epochs. Defaults to 1.
        dim (Optional[int]): Dimension of embedding. Defaults to None.
        adapter_model (Optional[BaseAdapter]): Adapter model. Defaults to None, in which
            case a linear adapter is used.
        device (Optional[str]): Device to use. Defaults to None.
        model_output_path (str): Path to save model output. Defaults to "model_output".
        verbose (bool): Whether to show progress bar. Defaults to False.
        bias (bool): Whether to use bias. Defaults to False.
    
    """

    def __init__(
        self,
        dataset: EmbeddingQAFinetuneDataset,
        embed_model: BaseEmbedding,
        batch_size: int = 10,
        epochs: int = 1,
        adapter_model: Optional[Any] = None,
        dim: Optional[int] = None,
        device: Optional[str] = None,
        model_output_path: str = "model_output",
        verbose: bool = False,
        bias: bool = False,
        **train_kwargs: Any,
    ) -> None:
        """Init params."""
        import torch
        from llama_index.embeddings.adapter_utils import BaseAdapter, LinearLayer

        self.dataset = dataset
        self.embed_model = embed_model

        # HACK: get dimension by passing text through it
        if dim is None:
            test_embedding = self.embed_model.get_text_embedding("hello world")
            self.dim = len(test_embedding)
        else:
            self.dim = dim

        # load in data, run embedding model, define data loader

        self.batch_size = batch_size
        self.loader = self._get_data_loader(dataset)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Use pytorch device: {}".format(device))
        self._target_device = torch.device(device)

        if adapter_model is not None:
            self.model = cast(BaseAdapter, adapter_model)
        else:
            self.model = LinearLayer(self.dim, self.dim, bias=bias)

        self._model_output_path = model_output_path
        self._epochs = epochs
        self._warmup_steps = int(len(self.loader) * epochs * 0.1)
        self._train_kwargs = train_kwargs

        self._verbose = verbose

    def smart_batching_collate(self, batch: List) -> Tuple[Any, Any]:
        """Smart batching collate."""
        from torch import Tensor
        import torch

        query_embeddings: List[Tensor] = []
        text_embeddings: List[Tensor] = []

        for query, text in batch:
            query_embedding = self.embed_model.get_query_embedding(query)
            text_embedding = self.embed_model.get_text_embedding(text)

            query_embeddings.append(torch.tensor(query_embedding))
            text_embeddings.append(torch.tensor(text_embedding))

        query_embeddings_t = torch.stack(query_embeddings)
        text_embeddings_t = torch.stack(text_embeddings)

        return query_embeddings_t, text_embeddings_t

    def _get_data_loader(self, dataset: EmbeddingQAFinetuneDataset) -> Any:
        """Get data loader."""
        from torch.utils.data import DataLoader

        examples: Any = []

        for query_id, query in dataset.queries.items():
            node_id = dataset.relevant_docs[query_id][0]
            text = dataset.corpus[node_id]

            examples.append((query, text))

        data_loader = DataLoader(examples, batch_size=self.batch_size)
        data_loader.collate_fn = self.smart_batching_collate

        return data_loader

    def finetune(self, **train_kwargs: Any) -> None:
        """Finetune."""
        from llama_index.finetuning.embeddings.adapter_utils import train_model

        # call model training
        train_model(
            self.model,
            self.loader,
            self._target_device,
            epochs=self._epochs,
            output_path=self._model_output_path,
            warmup_steps=self._warmup_steps,
            verbose=self._verbose,
            **self._train_kwargs,
        )

    def get_finetuned_model(self, **model_kwargs: Any) -> BaseEmbedding:
        """Get finetuned model."""

        return LinearAdapterEmbeddingModel(self.embed_model, self._model_output_path)
