from llama_index.core.readers.base import BaseReader
from llama_index.readers.funasr import FunASRReader


def test_class():
    names_of_base_classes = [b.__name__ for b in FunASRReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_init():
    reader = FunASRReader(model="iic/SenseVoiceSmall", device="cpu")
    assert reader.model_name == "iic/SenseVoiceSmall"
    assert reader.hub == "ms"
