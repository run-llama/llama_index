import json
from dataclasses import fields
from typing import Optional

from dataclasses_json import DataClassJsonMixin
from llama_index.core.constants import DATA_KEY, TYPE_KEY
from llama_index.core.data_structs.data_structs import IndexStruct
from llama_index.core.data_structs.registry import (
    INDEX_STRUCT_TYPE_TO_INDEX_STRUCT_CLASS,
)


def _fast_to_json(index_struct: IndexStruct) -> Optional[str]:
    """
    Serialize an index struct whose fields are already JSON-native.

    `DataClassJsonMixin.to_json` recursively walks and deep-copies every container
    entry, which dominates the cost of persisting large index structs on each
    insert/delete. Fall back (by returning None) whenever the generic path could
    produce a different result.
    """
    cls = type(index_struct)
    if (
        cls.to_json is not DataClassJsonMixin.to_json
        or cls.to_dict is not DataClassJsonMixin.to_dict
    ):
        return None

    struct_fields = fields(index_struct)
    if any(f.metadata for f in struct_fields):
        return None

    try:
        return json.dumps(
            {f.name: getattr(index_struct, f.name) for f in struct_fields}
        )
    except (TypeError, ValueError):
        return None


def index_struct_to_json(index_struct: IndexStruct) -> dict:
    data = _fast_to_json(index_struct)
    if data is None:
        data = index_struct.to_json()
    return {
        TYPE_KEY: index_struct.get_type(),
        DATA_KEY: data,
    }


def json_to_index_struct(struct_dict: dict) -> IndexStruct:
    type = struct_dict[TYPE_KEY]
    data_dict = struct_dict[DATA_KEY]
    cls = INDEX_STRUCT_TYPE_TO_INDEX_STRUCT_CLASS[type]
    try:
        return cls.from_json(data_dict)
    except TypeError:
        return cls.from_dict(data_dict)
