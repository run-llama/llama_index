from llama_index.core.question_gen.types import BaseQuestionGenerator
from llama_index.question_gen.guidance import GuidanceQuestionGenerator


def test_class():
    names_of_base_classes = [b.__name__ for b in GuidanceQuestionGenerator.__mro__]
    assert BaseQuestionGenerator.__name__ in names_of_base_classes
