from typing import Any, Dict, Type

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.readers.string_iterable import StringIterableReader

ALL_READERS: Dict[str, Type[BasePydanticReader]] = {
    StringIterableReader.class_name(): StringIterableReader,
}


def load_reader(data: Dict[str, Any]) -> BasePydanticReader:
    if isinstance(data, BasePydanticReader):
        return data

    class_name = data.get("class_name")

    if class_name is None:
        raise ValueError("Must specify `class_name` in reader data.")

    if class_name not in ALL_READERS:
        raise ValueError(f"Reader class name {class_name} not found.")

    # remove static attribute
    data.pop("is_remote", None)

    return ALL_READERS[class_name].from_dict(data)
