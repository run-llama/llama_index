from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import NodeRelationship, TextNode
from llama_index.readers.file import (
    DocxReader,
    EpubReader,
    FlatReader,
    HWPReader,
    ImageCaptionReader,
    ImageReader,
    ImageVisionLLMReader,
    IPYNBReader,
    MarkdownReader,
    MboxReader,
    PandasCSVReader,
    PDFReader,
    PptxReader,
    UnstructuredReader,
    VideoAudioReader,
    XMLReader,
    ImageTabularChartReader,
)


def test_classes():
    names_of_base_classes = [b.__name__ for b in DocxReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in HWPReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in PDFReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in EpubReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in FlatReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in ImageReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in ImageCaptionReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in ImageVisionLLMReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in IPYNBReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in MarkdownReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in MboxReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in PptxReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in PandasCSVReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in VideoAudioReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in XMLReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in ImageTabularChartReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in UnstructuredReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


class _FakeMetadata:
    def __init__(self, data):
        self._data = data

    def to_dict(self):
        return dict(self._data)


class _FakeElement:
    def __init__(self, text, metadata, hash_id):
        self.text = text
        self.metadata = _FakeMetadata(metadata)
        self._hash_id = hash_id

    def __str__(self):
        return self.text

    def id_to_hash(self, sequence_number):
        return f"{self._hash_id}-{sequence_number}"


def test_unstructured_reader_split_documents_preserves_source_relationship():
    reader = object.__new__(UnstructuredReader)
    reader.allowed_metadata_types = (str, int, float, type(None))
    reader.excluded_metadata_keys = {"orig_elements"}

    docs = reader._create_documents(
        elements=[
            _FakeElement(
                text="chunk one",
                metadata={"filename": "sample.txt", "file_path": "sample.txt"},
                hash_id="chunk",
            )
        ],
        document_kwargs=None,
        extra_info=None,
        split_documents=True,
        excluded_metadata_keys=None,
    )

    assert len(docs) == 1
    assert isinstance(docs[0], TextNode)
    assert docs[0].id_ == "chunk-0"
    assert docs[0].relationships[NodeRelationship.SOURCE].node_id == "sample.txt"
    assert docs[0].ref_doc_id == "sample.txt"
