from llama_index.readers.alibabacloud_aisearch import (
    AlibabaCloudAISearchDocumentReader,
    AlibabaCloudAISearchImageReader,
)
from llama_index.core.readers.base import BasePydanticReader


def test_class():
    names_of_base_classes = [
        b.__name__ for b in AlibabaCloudAISearchDocumentReader.__mro__
    ]
    assert BasePydanticReader.__name__ in names_of_base_classes
    names_of_base_classes = [
        b.__name__ for b in AlibabaCloudAISearchImageReader.__mro__
    ]
    assert BasePydanticReader.__name__ in names_of_base_classes
