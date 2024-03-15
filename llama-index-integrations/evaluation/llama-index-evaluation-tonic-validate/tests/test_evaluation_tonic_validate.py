from llama_index.core.evaluation.base import BaseEvaluator
from llama_index.evaluation.tonic_validate import (
    AnswerConsistencyEvaluator,
    AnswerConsistencyBinaryEvaluator,
    AnswerSimilarityEvaluator,
    AugmentationAccuracyEvaluator,
    AugmentationPrecisionEvaluator,
    RetrievalPrecisionEvaluator,
    TonicValidateEvaluator,
)


def test_evaluator_class():
    names_of_base_classes = [b.__name__ for b in AnswerConsistencyEvaluator.__mro__]
    assert BaseEvaluator.__name__ in names_of_base_classes

    names_of_base_classes = [
        b.__name__ for b in AnswerConsistencyBinaryEvaluator.__mro__
    ]
    assert BaseEvaluator.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in AnswerSimilarityEvaluator.__mro__]
    assert BaseEvaluator.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in AugmentationAccuracyEvaluator.__mro__]
    assert BaseEvaluator.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in AugmentationPrecisionEvaluator.__mro__]
    assert BaseEvaluator.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in RetrievalPrecisionEvaluator.__mro__]
    assert BaseEvaluator.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in TonicValidateEvaluator.__mro__]
    assert BaseEvaluator.__name__ in names_of_base_classes
