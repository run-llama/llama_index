"""Common structures for structured indices."""

from dataclasses import dataclass
from typing import Dict, Optional

from dataclasses_json import DataClassJsonMixin


# TODO: migrate this to be a data_struct
@dataclass
class SQLContextContainer(DataClassJsonMixin):
    """
    SQLContextContainer.

    A container interface to store context for a given table.
    Context can be built from unstructured documents (e.g. using SQLContextBuilder).
    Context can also be dumped to an underlying LlamaIndex data structure.

    Contains both the raw context_dict as well as any index_structure.

    Should be not be used directly - build one from SQLContextContainerBuilder instead.

    """

    context_dict: Optional[Dict[str, str]] = None
    context_str: Optional[str] = None
