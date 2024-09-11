from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.postprocessor.alibabacloud_aisearch_rerank import (
    AlibabaCloudAISearchRerank,
)


def test_class():
    names_of_base_classes = [b.__name__ for b in AlibabaCloudAISearchRerank.__mro__]
    assert BaseNodePostprocessor.__name__ in names_of_base_classes
