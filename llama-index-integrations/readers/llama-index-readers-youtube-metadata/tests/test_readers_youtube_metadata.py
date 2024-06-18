from llama_index.core.readers.base import BaseReader
from llama_index.readers.youtube_metadata import (
    YouTubeMetaData,
    YouTubeMetaDataAndTranscript,
)


def test_class():
    names_of_base_classes = [b.__name__ for b in YouTubeMetaData.__mro__]
    assert BaseReader.__name__ in names_of_base_classes
    names_of_base_classes = [b.__name__ for b in YouTubeMetaDataAndTranscript.__mro__]
    assert BaseReader.__name__ in names_of_base_classes
