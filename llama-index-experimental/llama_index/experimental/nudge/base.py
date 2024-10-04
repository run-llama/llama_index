import logging
from typing import Dict, Optional

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.utils import infer_torch_device
from llama_index.finetuning.embeddings.common import EmbeddingQAFinetuneDataset

logger = logging.getLogger(__name__)

NUDGE_IMPORT_ERROR_MSG = "NUDGE is not installed. Please install it with 'pip install nudge-ft' to use this functionality."
NUMPY_IMPORT_ERROR_MSG = "Numpy is not installed. Please install it with 'pip install numpy' to use this functionality."
PYTORCH_IMPORT_ERROR_MSG = "Pytorch is not installed. Please install it with 'pip install torch' to use this functionality."


class Nudge:
    """The algorithm implemented here and the current state of the art is called [NUDGE](https://www.arxiv.org/abs/2409.02343).
    If a validation dataset is provided, the best model is evaluated and saved based on the validation loss at the end of every epoch.

    Args:
        train_dataset (EmbeddingQAFinetuneDataset): Dataset to finetune on.
        embed_model (BaseEmbedding): Embedding model.
        val_dataset (EmbeddingQAFinetuneDataset): Validation dataset.
        use_nudge_n (bool): Whether to use NUDGE-N or NUDGE-M. Defaults to True.
        device (Optional[str]): Device to use. Defaults to None.
    """

    def __init__(
        self,
        embed_model: BaseEmbedding,
        train_dataset: EmbeddingQAFinetuneDataset,
        val_dataset: EmbeddingQAFinetuneDataset,
        use_nudge_n: bool = True,
        device: Optional[str] = None,
    ) -> None:
        """Init params."""
        try:
            from nudge import NUDGEN, NUDGEM
        except ImportError:
            raise ImportError(NUDGE_IMPORT_ERROR_MSG)
        try:
            import torch
        except ImportError:
            raise ImportError(PYTORCH_IMPORT_ERROR_MSG)

        if device is None:
            device = infer_torch_device()
            logger.info(f"Use pytorch device: {device}")
        self._target_device = torch.device(device)

        self.embed_model = embed_model
        self.corpus = train_dataset.corpus
        self.corpus_embeddings = self._get_corpus_embeddings(self.corpus)
        self.train_dataset = self._format_dataset(train_dataset, self.corpus)
        self.val_dataset = self._format_dataset(val_dataset, self.corpus)

        self.nudge = (
            NUDGEN(device=self._target_device)
            if use_nudge_n
            else NUDGEM(device=self._target_device)
        )

    def _format_dataset(
        self, dataset: EmbeddingQAFinetuneDataset, corpus: Dict[str, str]
    ):
        """
        Convert the dataset into NUDGE format.

        Args:
            dataset (EmbeddingQAFinetuneDataset): Dataset to convert.
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError(NUMPY_IMPORT_ERROR_MSG)

        q_embs = []
        q_ans_indx = []
        corpus_keys = list(corpus.keys())

        for query_id, query in dataset.queries.items():
            query_embedding = self.embed_model.get_query_embedding(query)
            q_embs.append(query_embedding)

            relevant_docs = dataset.relevant_docs[query_id]
            relevant_doc_indices = [corpus_keys.index(doc) for doc in relevant_docs]
            q_ans_indx.append(relevant_doc_indices)

        return {"q_embs": np.array(q_embs), "q_ans_indx": q_ans_indx}

    def _get_corpus_embeddings(self, corpus: Dict[str, str]):
        """Get corpus embeddings."""
        try:
            import numpy as np
        except ImportError:
            raise ImportError(NUMPY_IMPORT_ERROR_MSG)

        text_embeddings = [
            self.embed_model.get_text_embedding(text) for text in corpus.values()
        ]
        return np.array(text_embeddings)

    def finetune(self):
        self.corpus_embeddings = self.nudge.finetune_embeddings(
            embeddings=self.corpus_embeddings,
            train_set=self.train_dataset,
            val_set=self.val_dataset,
            nontrain_embeddings=None,
            val_batch_size=256,
            gamma=None,
        )

    def insert_data_and_finetune(
        self,
        new_train_dataset_batch: EmbeddingQAFinetuneDataset,
        new_val_dataset_batch: Optional[EmbeddingQAFinetuneDataset] = None,
    ):
        """
        Insert data and finetune. This should only be done if the new data you are inserting does not conflict with the already existing data. It's important to not finetune multiple times as this can cause the embeddings to lose semantic meaning since they will become further from the original embeddings.
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError(NUMPY_IMPORT_ERROR_MSG)

        new_corpus_batch = new_train_dataset_batch.corpus
        # if any of the new ids are already in the existing corpus, raise an error
        if any(id in self.corpus for id in new_corpus_batch):
            raise ValueError(
                f"ID {id} already exists in the existing corpus. New IDs must be unique."
            )

        # get the embeddings for the new corpus
        new_corpus_initial_embeddings_batch = self._get_corpus_embeddings(
            new_corpus_batch
        )

        existing_corpus_embeddings = self.corpus_embeddings

        new_train_dataset = self._format_dataset(
            new_train_dataset_batch, new_corpus_batch
        )
        new_val_dataset = self._format_dataset(new_val_dataset_batch, new_corpus_batch)

        new_corpus_embeddings_batch = self.nudge.finetune_embeddings(
            embeddings=new_corpus_initial_embeddings_batch,
            train_set=new_train_dataset,
            val_set=new_val_dataset,
            # runs faster by filtering the embeddings which will not have any queries
            nontrain_embeddings=existing_corpus_embeddings,
            val_batch_size=256,
            gamma=None,
        )

        self.corpus_embeddings = np.concatenate(
            [existing_corpus_embeddings, new_corpus_embeddings_batch]
        )

    def get_finetuned_corpus_embeddings(self):
        return self.corpus_embeddings
