from typing import Optional
import logging
import textwrap
from jinja2 import Template

from llama_index.core.llms import ChatMessage
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.storage.chat_store.base import BaseChatStore

_logger = logging.getLogger(__name__)


IMPORT_ERROR_MESSAGE = """
Error: Gel Python package is not installed.
Please install it using 'pip install gel'.
"""

NO_PROJECT_MESSAGE = """
Error: it appears that the Gel project has not been initialized.
If that's the case, please run 'gel project init' to get started.
"""

MISSING_RECORD_TYPE_TEMPLATE = """
Error: Record type {{record_type}} is missing from the Gel schema.

In order to use the LangChain integration, ensure you put the following in dbschema/default.gel:

    using extension pgvector;

    module default {
        type {{record_type}} {
            required collection: str;
            text: str;
            embedding: ext::pgvector::vector<1536>;
            external_id: str {
                constraint exclusive;
            };
            metadata: json;

            index ext::pgvector::hnsw_cosine(m := 16, ef_construction := 128)
                on (.embedding)
        } 
    }

Remember that you also need to run a migration:

    $ gel migration create
    $ gel migrate

"""

try:
    import gel
except ImportError as e:
    _logger.error(IMPORT_ERROR_MESSAGE)
    raise e


def format_query(text: str) -> str:
    return textwrap.dedent(text.strip())


SET_MESSAGES_QUERY = format_query(
    """
    insert Record {
        key := <str>$key,
        value := <array<json>>$value
    } unless conflict on .key else (
        update Record set {
            value := <array<json>>$value
        }
    )
    """
)

GET_MESSAGES_QUERY = format_query(
    """
    with
        record := (select Record filter .key = <str>$key),
    select record.value;
    """
)

ADD_MESSAGE_QUERY = format_query(
    """
    insert Record {
        key := <str>$key,
        value := <array<json>>$value
    } unless conflict on .key else (
        update Record set {
            value := .value ++ <array<json>>$value
        }
    )
    """
)

DELETE_MESSAGES_QUERY = format_query(
    """
    delete Record filter .key = <str>$key
    """
)

DELETE_MESSAGE_QUERY = format_query(
    """
    with
        idx := <int64>$idx,
        value := (select Record filter .key = <str>$key).value,
        idx_item := value[idx],
        new_value := value[:idx] ++ value[idx+1:],
        updated_record := (
            update Record
            filter .key = <str>$key
            set {
                value := new_value
            }
        )
    select idx_item;
    """
)

DELETE_LAST_MESSAGE_QUERY = format_query(
    """
    with
        value := (select Record filter .key = <str>$key).value,
        last_item := value[len(value) - 1],
        new_value := value[:len(value) - 1],
        updated_record := (
            update Record
            filter .key = <str>$key
            set {
                value := new_value
            }
        )
    select last_item;
    """
)

GET_KEYS_QUERY = format_query(
    """
    select Record.key;
    """
)


class GelChatStore(BaseChatStore):
    record_type: str
    _sync_client: Optional[gel.Client] = PrivateAttr()
    _async_client: Optional[gel.AsyncIOClient] = PrivateAttr()

    def __init__(
        self,
        record_type: str = "Record",
    ):
        super().__init__(record_type=record_type)

        self._sync_client = gel.create_client()
        self._async_client = gel.create_async_client()

        try:
            self._sync_client.ensure_connected()
        except gel.errors.ClientConnectionError as e:
            _logger.error(NO_PROJECT_MESSAGE)
            raise e

        try:
            self._sync_client.query(f"select {self.record_type};")
        except gel.errors.InvalidReferenceError as e:
            _logger.error(
                Template(MISSING_RECORD_TYPE_TEMPLATE).render(record_type="Record")
            )
            raise e

    def set_messages(self, key: str, messages: list[ChatMessage]) -> None:
        """Set messages for a key."""
        self._sync_client.query(
            SET_MESSAGES_QUERY,
            key=key,
            value=[message.model_dump_json() for message in messages],
        )

    async def aset_messages(self, key: str, messages: list[ChatMessage]) -> None:
        """Async version of Get messages for a key."""
        await self._async_client.query(
            SET_MESSAGES_QUERY,
            key=key,
            value=[message.model_dump_json() for message in messages],
        )

    def get_messages(self, key: str) -> list[ChatMessage]:
        """Get messages for a key."""
        result = self._sync_client.query_single(GET_MESSAGES_QUERY, key=key) or []
        return [ChatMessage.model_validate_json(message) for message in result]

    async def aget_messages(self, key: str) -> list[ChatMessage]:
        """Async version of Get messages for a key."""
        result = await self._async_client.query_single(GET_MESSAGES_QUERY, key=key) or []
        return [ChatMessage.model_validate_json(message) for message in result]

    def add_message(self, key: str, message: ChatMessage) -> None:
        """Add a message for a key."""
        self._sync_client.query(ADD_MESSAGE_QUERY, key=key, value=[message.model_dump_json()])

    async def async_add_message(self, key: str, message: ChatMessage) -> None:
        """Async version of Add a message for a key."""
        await self._async_client.query(ADD_MESSAGE_QUERY, key=key, value=[message.model_dump_json()])

    def delete_messages(self, key: str) -> Optional[list[ChatMessage]]:
        """Delete messages for a key."""
        self._sync_client.query(DELETE_MESSAGES_QUERY, key=key)

    async def adelete_messages(self, key: str) -> Optional[list[ChatMessage]]:
        """Async version of Delete messages for a key."""
        await self._async_client.query(DELETE_MESSAGES_QUERY, key=key)

    def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """Delete specific message for a key."""
        result = self._sync_client.query_single(DELETE_MESSAGE_QUERY, key=key, idx=idx)
        return ChatMessage.model_validate_json(result) if result else None

    async def adelete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """Async version of Delete specific message for a key."""
        result = await self._async_client.query_single(DELETE_MESSAGE_QUERY, key=key, idx=idx)
        return ChatMessage.model_validate_json(result) if result else None

    def delete_last_message(self, key: str) -> Optional[ChatMessage]:
        """Delete last message for a key."""
        result = self._sync_client.query_single(DELETE_LAST_MESSAGE_QUERY, key=key)
        return ChatMessage.model_validate_json(result) if result else None

    async def adelete_last_message(self, key: str) -> Optional[ChatMessage]:
        """Async version of Delete last message for a key."""
        result = await self._async_client.query_single(DELETE_LAST_MESSAGE_QUERY, key=key)
        return ChatMessage.model_validate_json(result) if result else None

    def get_keys(self) -> list[str]:
        """Get all keys."""
        return self._sync_client.query(GET_KEYS_QUERY)

    async def aget_keys(self) -> list[str]:
        """Async version of Get all keys."""
        return await self._async_client.query(GET_KEYS_QUERY)
