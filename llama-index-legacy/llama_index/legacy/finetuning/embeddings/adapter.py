"""Sentence Transformer Finetuning Engine."""

import logging
from typing import Any, List, Optional, Tuple, Type, cast

from llama_index.legacy.embeddings.adapter import AdapterEmbeddingModel
from llama_index.legacy.embeddings.base import BaseEmbedding
from llama_index.legacy.finetuning.embeddings.common import EmbeddingQAFinetuneDataset
from llama_index.legacy.finetuning.types import BaseEmbeddingFinetuneEngine
from llama_index.legacy.utils import infer_torch_device

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
        model_checkpoint_path (Optional[str]): Path to save model checkpoints.
            Defaults to None (don't save checkpoints).
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
        model_checkpoint_path: Optional[str] = None,
        checkpoint_save_steps: int = 100,
        verbose: bool = False,
        bias: bool = False,
        **train_kwargs: Any,
    ) -> None:
        """Init params."""
        import torch

        from llama_index.legacy.embeddings.adapter_utils import BaseAdapter, LinearLayer

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
            device = infer_torch_device()
            logger.info(f"Use pytorch device: {device}")
        self._target_device = torch.device(device)

        if adapter_model is not None:
            self.model = cast(BaseAdapter, adapter_model)
        else:
            self.model = LinearLayer(self.dim, self.dim, bias=bias)

        self._model_output_path = model_output_path
        self._model_checkpoint_path = model_checkpoint_path
        self._checkpoint_save_steps = checkpoint_save_steps
        self._epochs = epochs
        self._warmup_steps = int(len(self.loader) * epochs * 0.1)
        self._train_kwargs = train_kwargs

        self._verbose = verbose

    @classmethod
    def from_model_path(
        cls,
        dataset: EmbeddingQAFinetuneDataset,
        embed_model: BaseEmbedding,
        model_path: str,
        model_cls: Optional[Type[Any]] = None,
        **kwargs: Any,
    ) -> "EmbeddingAdapterFinetuneEngine":
        """Load from model path.

        Args:
            dataset (EmbeddingQAFinetuneDataset): Dataset to finetune on.
            embed_model (BaseEmbedding): Embedding model to finetune.
            model_path (str): Path to model.
            model_cls (Optional[Type[Any]]): Adapter model class. Defaults to None.
            **kwargs (Any): Additional kwargs (see __init__)

        """
        from llama_index.legacy.embeddings.adapter_utils import LinearLayer

        model_cls = model_cls or LinearLayer
        model = model_cls.load(model_path)
        return cls(dataset, embed_model, adapter_model=model, **kwargs)

    def smart_batching_collate(self, batch: List) -> Tuple[Any, Any]:
        """Smart batching collate."""
        import torch
        from torch import Tensor

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
        from llama_index.legacy.finetuning.embeddings.adapter_utils import train_model

        # call model training
        train_model(
            self.model,
            self.loader,
            self._target_device,
            epochs=self._epochs,
            output_path=self._model_output_path,
            warmup_steps=self._warmup_steps,
            verbose=self._verbose,
            checkpoint_path=self._model_checkpoint_path,
            checkpoint_save_steps=self._checkpoint_save_steps,
            **self._train_kwargs,
        )

    def get_finetuned_model(self, **model_kwargs: Any) -> BaseEmbedding:
        """Get finetuned model."""
        return AdapterEmbeddingModel(
            self.embed_model, self._model_output_path, **model_kwargs
        )
