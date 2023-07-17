class SemanticAnswerSimilarity:
    def __init__(
        self,
        model: str = "cross-encoder/stsb-roberta-large",
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "Cannot import sentence-transformers package,",
                "please `pip install sentence-transformers`",
            )

        self._model = CrossEncoder(model, max_length=512)

    def evaluate(self, ground_truth: str, predicted: str) -> float:
        """
        Evaluates the semantic similarity between a ground truth answer and
        a predicted answer.
        """
        scores = self._model.predict([(ground_truth, predicted)])
        return scores[0]


class SemanticRelationMatch:
    def __init__(
        self,
        model: str = "cross-encoder/nli-deberta-v3-base",
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "Cannot import sentence-transformers package,",
                "please `pip install sentence-transformers`",
            )

        self._model = CrossEncoder(model, max_length=512)

    def evaluate(self, ground_truth: str, predicted: str) -> float:
        """
        Evaluates the semantic similarity between a ground truth answer and
        a predicted answer.
        """
        scores = self._model.predict([(ground_truth, predicted)])
        return scores.argmax(axis=1)[0] - 1.0
