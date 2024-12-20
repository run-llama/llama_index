"""Sentence Transformer Finetuning Engine."""

from typing import Any, Optional

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.embeddings.utils import resolve_embed_model
from llama_index.finetuning.embeddings.common import (
    EmbeddingQAFinetuneDataset,
)
from llama_index.finetuning.types import BaseEmbeddingFinetuneEngine


class SentenceTransformersFinetuneEngine(BaseEmbeddingFinetuneEngine):
    """Sentence Transformers Finetune Engine."""

    def __init__(
        self,
        dataset: EmbeddingQAFinetuneDataset,
        model_id: str = "BAAI/bge-small-en",
        model_output_path: str = "exp_finetune",
        batch_size: int = 10,
        val_dataset: Optional[EmbeddingQAFinetuneDataset] = None,
        loss: Optional[Any] = None,
        epochs: int = 2,
        show_progress_bar: bool = True,
        evaluation_steps: int = 50,
        use_all_docs: bool = False,
        trust_remote_code: bool = False,
        device: Optional[Any] = None,
    ) -> None:
        """Init params."""
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader

        self.dataset = dataset

        self.model_id = model_id
        self.model_output_path = model_output_path
        self.model = SentenceTransformer(
            model_id, trust_remote_code=trust_remote_code, device=device
        )

        self.use_all_docs = use_all_docs

        examples: Any = []
        for query_id, query in dataset.queries.items():
            if use_all_docs:
                for node_id in dataset.relevant_docs[query_id]:
                    text = dataset.corpus[node_id]
                    example = InputExample(texts=[query, text])
                    examples.append(example)
            else:
                node_id = dataset.relevant_docs[query_id][0]
                text = dataset.corpus[node_id]
                example = InputExample(texts=[query, text])
                examples.append(example)

        self.examples = examples

        self.loader: DataLoader = DataLoader(examples, batch_size=batch_size)

        # define evaluator
        from sentence_transformers.evaluation import InformationRetrievalEvaluator

        evaluator: Optional[InformationRetrievalEvaluator] = None
        if val_dataset is not None:
            evaluator = InformationRetrievalEvaluator(
                val_dataset.queries, val_dataset.corpus, val_dataset.relevant_docs
            )
        self.evaluator = evaluator

        # define loss
        self.loss = loss or losses.MultipleNegativesRankingLoss(self.model)

        self.epochs = epochs
        self.show_progress_bar = show_progress_bar
        self.evaluation_steps = evaluation_steps
        self.warmup_steps = int(len(self.loader) * epochs * 0.1)

    def finetune(self, **train_kwargs: Any) -> None:
        """Finetune model."""
        self.model.fit(
            train_objectives=[(self.loader, self.loss)],
            epochs=self.epochs,
            warmup_steps=self.warmup_steps,
            output_path=self.model_output_path,
            show_progress_bar=self.show_progress_bar,
            evaluator=self.evaluator,
            evaluation_steps=self.evaluation_steps,
        )

    def get_finetuned_model(self, **model_kwargs: Any) -> BaseEmbedding:
        """Gets finetuned model."""
        embed_model_str = "local:" + self.model_output_path
        return resolve_embed_model(embed_model_str)
