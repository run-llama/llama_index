import logging
from typing import Any, Optional

from tqdm import tqdm, trange

from llama_index.core.utils import print_text
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.utils import infer_torch_device
from llama_index.finetuning.embeddings.common import EmbeddingQAFinetuneDataset

logger = logging.getLogger(__name__)

IMPORT_ERROR_MSG = "PyTorch is not installed. Please install it with 'pip install torch' to use this functionality."


def multiclass_nll_loss(output, targets):
    return (-1 * output * targets).sum(axis=-1).mean()


class Nudge:
    """The algorithm implemented here and the current state of the art is called [NUDGE](https://www.arxiv.org/abs/2409.02343).
    If a validation dataset is provided, the best model is evaluated and saved based on the validation loss at the end of every epoch.

    Args:
        train_dataset (EmbeddingQAFinetuneDataset): Dataset to finetune on.
        embed_model (BaseEmbedding): Embedding model.
        val_dataset (Optional[EmbeddingQAFinetuneDataset]): Validation dataset. Defaults to None.
        train_batch_size (Optional[int]): Train batch size. Defaults to 10.
        val_batch_size (Optional[int]): Validation batch size. Defaults to 10.
        epochs (Optional[int]): Number of epochs. Defaults to 1.
        dim (Optional[int]): Dimension of embedding. Defaults to None.
        device (Optional[str]): Device to use. Defaults to None.
        model_output_path (str): Path to save model output. Defaults to "model_output".
        model_checkpoint_path (Optional[str]): Path to save model checkpoints.
            Defaults to None (don't save checkpoints).
        verbose (bool): Whether to show progress bar. Defaults to False.
        bias (bool): Whether to use bias. Defaults to False.
    """

    def __init__(
        self,
        train_dataset: EmbeddingQAFinetuneDataset,
        embed_model: BaseEmbedding,
        val_dataset: Optional[EmbeddingQAFinetuneDataset] = None,
        train_batch_size: int = 10,
        val_batch_size: int = 10,
        epochs: int = 1,
        device: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        """Init params."""
        try:
            import torch
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.embed_model = embed_model
        self.corpus_embeddings = self._get_corpus_embeddings(train_dataset)

        # load in data, run embedding model, define data loader

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_loader = self._get_data_loader(train_dataset, train_batch_size)
        self.val_loader = (
            self._get_data_loader(val_dataset, val_batch_size)
            if val_dataset is not None
            else None
        )

        if device is None:
            device = infer_torch_device()
            logger.info(f"Use pytorch device: {device}")
        self._target_device = torch.device(device)

        self._epochs = epochs

        self._verbose = verbose

    def _get_data_loader(
        self, dataset: EmbeddingQAFinetuneDataset, batch_size: int
    ) -> Any:
        """Get data loader."""
        try:
            import torch
            from torch.utils.data import DataLoader
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        examples: Any = []

        for query_id, query in dataset.queries.items():
            query_embedding = torch.tensor(self.embed_model.get_query_embedding(query))
            relevant_docs = dataset.relevant_docs[query_id]
            relevant_docs = torch.tensor(
                [1 if doc in relevant_docs else 0 for doc in dataset.corpus]
            )

            examples.append((query_embedding, relevant_docs))

        return DataLoader(examples, batch_size=batch_size)

    def _get_corpus_embeddings(self, dataset: EmbeddingQAFinetuneDataset):
        """Get corpus embeddings."""
        try:
            import torch
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        text_embeddings = [
            self.embed_model.get_text_embedding(text)
            for text in dataset.corpus.values()
        ]
        return torch.tensor(text_embeddings, requires_grad=False)

    def _evaluate_acc(self, model, loader):
        """Evaluate model."""
        try:
            import torch
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        model.eval()
        total_acc = 0
        total_records = 0
        with torch.no_grad():
            for query_embeddings_t, relevant_docs_t in loader:
                query_embeddings_t = query_embeddings_t.to(self._target_device)
                relevant_docs_t = relevant_docs_t.to(self._target_device)

                preds = model(query_embeddings_t)
                out = preds.max(1).indices.view(-1, 1)
                truths = torch.gather(relevant_docs_t, 1, out)

                total_acc += truths.sum().item()
                total_records += truths.shape[0]
        return total_acc / total_records

    def finetune(self):
        try:
            import torch
            from torch import nn
            from torch.nn import functional as F
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        # initialize the weights of a linear model with the normalized corpus embeddings
        w_init = F.normalize(self.corpus_embeddings)
        model = nn.Linear(w_init.shape[1], w_init.shape[0], bias=False)
        model.weight.data = w_init
        model.to(self._target_device)

        # train the model
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8
        )
        best_val_acc = self._evaluate_acc(model, self.val_loader)

        for epoch in trange(self._epochs, desc="Epoch"):
            model.train()
            for query_embeddings_t, relevant_docs_t in tqdm(
                self.train_loader, desc=f"Epoch {epoch+1}/{self._epochs}", leave=False
            ):
                query_embeddings_t = query_embeddings_t.to(self._target_device)
                relevant_docs_t = relevant_docs_t.to(self._target_device)

                loss = multiclass_nll_loss(model(query_embeddings_t), relevant_docs_t)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # normalize the weights
                with torch.no_grad():
                    model.weight.data = F.normalize(model.weight.data)

                if self._verbose:
                    print_text(
                        f"> [Epoch {epoch}] Current loss: {loss}\n", color="blue"
                    )
            if self.val_loader is not None:
                val_acc = self._evaluate_acc(model, self.val_loader)
                if self._verbose:
                    print_text(
                        f"> [Epoch {epoch}] validation acc: {val_acc} best validation acc: {best_val_acc} \n",
                        color="blue",
                    )
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.corpus_embeddings = model.weight.data.clone()
            else:
                self.corpus_embeddings = model.weight.data.clone()

    def get_finetuned_corpus_embeddings(self):
        return self.corpus_embeddings
