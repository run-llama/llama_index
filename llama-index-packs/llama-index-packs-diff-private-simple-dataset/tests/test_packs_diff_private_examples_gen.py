from llama_index.core.llama_pack import BaseLlamaPack
from llama_index.packs.diff_private_simple_dataset import DiffPrivateSimpleDatasetPack


def test_class():
    names_of_base_classes = [b.__name__ for b in DiffPrivateSimpleDatasetPack.__mro__]
    assert BaseLlamaPack.__name__ in names_of_base_classes
