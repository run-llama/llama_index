from llama_index.core.readers.base import BaseReader
from llama_index.readers.opendal import (
    OpendalAzblobReader,
    OpendalGcsReader,
    OpendalReader,
    OpendalS3Reader,
)


def test_class():
    names_of_base_classes = [b.__name__ for b in OpendalReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in OpendalGcsReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in OpendalAzblobReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in OpendalS3Reader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes
