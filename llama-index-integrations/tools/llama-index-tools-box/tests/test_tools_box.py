from llama_index.core.tools.tool_spec.base import BaseToolSpec

from llama_index.tools.box import (
    BoxSearchToolSpec,
    BoxSearchByMetadataToolSpec,
    BoxTextExtractToolSpec,
)


def test_box_class():
    names_of_base_classes = [b.__name__ for b in BoxSearchToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in BoxSearchByMetadataToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in BoxTextExtractToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes
