from llama_index.core.readers.base import BaseReader
from llama_index.readers.smart_pdf_loader import SmartPDFLoader


def test_class():
    names_of_base_classes = [b.__name__ for b in SmartPDFLoader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes
