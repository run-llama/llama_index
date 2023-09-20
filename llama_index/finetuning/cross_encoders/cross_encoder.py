"""Cross Encoder Finetuning Engine."""


from typing import Any, Optional, List, Union
from llama_index.finetuning.cross_encoders.dataset_gen import (
    CrossEncoderFinetuningDatasetSample,
)
from llama_index.finetuning.types import BaseCrossEncoderFinetuningEngine


class CrossEncoderFinetuneEngine(BaseCrossEncoderFinetuningEngine):
    """Cross-Encoders Finetune Engine."""

    def __init__(
        self,
        dataset: List[CrossEncoderFinetuningDatasetSample],
        model_id: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        model_output_path: str = "exp_finetune",
        batch_size: int = 10,
        val_dataset: Union[List[CrossEncoderFinetuningDatasetSample], None] = None,
        loss: Union[Any, None] = None,
        epochs: int = 2,
        show_progress_bar: bool = True,
        evaluation_steps: int = 50,
    ) -> None:
        """Init params."""
        try:
            from sentence_transformers import InputExample
            from sentence_transformers.cross_encoder import CrossEncoder
            from torch.utils.data import DataLoader
        except ImportError:
            raise ImportError(
                "Cannot import sentence-transformers package,",
                "please `pip install sentence-transformers`",
            )

        self.dataset = dataset

        self.model_id = model_id
        self.model_output_path = model_output_path
        self.model = CrossEncoder(self.model_id, num_labels=1)

        examples: Any = []
        for sample in dataset:
            query = sample.query
            text = sample.context
            score = sample.score
            example = InputExample(texts=[query, text], label=score)
            examples.append(example)
        self.examples = examples

        self.loader: DataLoader = DataLoader(examples, batch_size=batch_size)

        # define evaluator
        from sentence_transformers.cross_encoder.evaluation import (
            CEBinaryClassificationEvaluator,
        )

        # Todo: also add support for CERerankingEvaluator
        evaluator: Optional[CEBinaryClassificationEvaluator] = None

        if val_dataset is not None:
            dev_samples = []

            for val_sample in val_dataset:
                val_query = val_sample.query
                val_text = val_sample.context
                val_score = val_sample.score
                val_example = InputExample(texts=[val_query, val_text], label=val_score)
                dev_samples.append(val_example)

            evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev_samples)

        self.evaluator = evaluator

        # define loss
        self.loss = loss

        self.epochs = epochs
        self.show_progress_bar = show_progress_bar
        self.evaluation_steps = evaluation_steps
        self.warmup_steps = int(len(self.loader) * epochs * 0.1)

    def finetune(self, **train_kwargs: Any) -> None:
        """Finetune model."""
        self.model.fit(
            train_dataloader=self.loader,
            epochs=self.epochs,
            warmup_steps=self.warmup_steps,
            output_path=self.model_output_path,
            show_progress_bar=self.show_progress_bar,
            evaluator=self.evaluator,
            evaluation_steps=self.evaluation_steps,
        )

    def get_finetuned_model(self, **model_kwargs: Any) -> CrossEncoder:
        """Gets finetuned model."""

        from sentence_transformers.cross_encoder import CrossEncoder

        finetuned_model = CrossEncoder(self.model_output_path)

        return finetuned_model
