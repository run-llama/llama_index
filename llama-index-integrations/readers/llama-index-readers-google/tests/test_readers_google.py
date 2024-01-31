from llama_index.core.readers.base import BasePydanticReader, BaseReader
from llama_index.readers.google import (
    GoogleDocsReader,
    GoogleSheetsReader,
    GoogleKeepReader,
    GmailReader,
    GoogleCalendarReader,
    GoogleDriveReader,
)


def test_class():
    names_of_base_classes = [b.__name__ for b in GoogleDocsReader.__mro__]
    assert BasePydanticReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in GoogleSheetsReader.__mro__]
    assert BasePydanticReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in GoogleKeepReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in GmailReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in GoogleCalendarReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in GoogleDriveReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes
