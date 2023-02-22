from dataclasses import dataclass
from typing import Dict, Optional

from dataclasses_json import DataClassJsonMixin

from gpt_index.data_structs.data_structs import IndexStruct
from gpt_index.docstore import DocumentStore


# TODO: migrate this to be a data_struct
@dataclass
class SQLContextContainer(DataClassJsonMixin):
    """SQLContextContainer.

    A container interface to store context for a given table.
    Context can be built from unstructured documents (e.g. using SQLContextBuilder).
    Context can also be dumped to an underlying GPT Index data structure.

    Contains both the raw context_dict as well as any index_structure.

    Should be not be used directly - build one from SQLContextContainerBuilder instead.

    """

    context_dict: Optional[Dict[str, str]] = None
    context_str: Optional[str] = None

    # index_struct: Optional[IndexStruct] = None
    # index_docstore: Optional[DocumentStore] = None

    # def __init__(
    #     self,
    #     # sql_database: SQLDatabase,
    #     context_dict: Optional[Dict[str, str]] = None,
    #     # index: Optional[BaseGPTIndex] = None,
    #     index_struct: Optional[IndexStruct] = None,
    #     index_docstore: Optional[DocumentStore] = None,
    # ):
    #     """Initialize params."""
    #     self.context_dict = context_dict
    #     # TODO: translate
    #     # self.index = index

    #     self.index_struct = index_struct
    #     self.index_docstore = index_docstore

    # # def build_index(self, index_cls: Type[BaseGPTIndex]) -> None:
    # #     """Build index structure."""
