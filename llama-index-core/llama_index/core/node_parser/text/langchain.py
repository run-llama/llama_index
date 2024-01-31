from typing import TYPE_CHECKING, Callable, List, Optional

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.node_parser.interface import TextSplitter
from llama_index.core.node_parser.node_utils import default_id_func
from llama_index.core.schema import Document

if TYPE_CHECKING:
    from langchain.text_splitter import TextSplitter as LC_TextSplitter


class LangchainNodeParser(TextSplitter):
    """
    Basic wrapper around langchain's text splitter.

    TODO: Figure out how to make this metadata aware.
    """

    _lc_splitter: "LC_TextSplitter" = PrivateAttr()

    def __init__(
        self,
        lc_splitter: "LC_TextSplitter",
        callback_manager: Optional[CallbackManager] = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        id_func: Optional[Callable[[int, Document], str]] = None,
    ):
        """Initialize with parameters."""
        try:
            from langchain.text_splitter import TextSplitter as LC_TextSplitter  # noqa
        except ImportError:
            raise ImportError(
                "Could not run `from langchain.text_splitter import TextSplitter`, "
                "please run `pip install langchain`"
            )
        id_func = id_func or default_id_func

        super().__init__(
            callback_manager=callback_manager or CallbackManager(),
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            id_func=id_func,
        )
        self._lc_splitter = lc_splitter

    def split_text(self, text: str) -> List[str]:
        """Split text into sentences."""
        return self._lc_splitter.split_text(text)
