import json
from typing import Any, Callable, Iterator, List, Mapping, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

RecordHandler = Callable[[Any, Optional[str]], Document]


class AirbyteCDKReader(BaseReader):
    """
    AirbyteCDKReader reader.

    Retrieve documents from an Airbyte source implemented using the CDK.

    Args:
        source_class: The Airbyte source class.
        config: The config object for the Airbyte source.

    """

    def __init__(
        self,
        source_class: Any,
        config: Mapping[str, Any],
        record_handler: Optional[RecordHandler] = None,
    ) -> None:
        """Initialize with parameters."""
        from airbyte_cdk.models.airbyte_protocol import AirbyteRecordMessage
        from airbyte_cdk.sources.embedded.base_integration import (
            BaseEmbeddedIntegration,
        )
        from airbyte_cdk.sources.embedded.runner import CDKRunner

        class CDKIntegration(BaseEmbeddedIntegration):
            def _handle_record(
                self, record: AirbyteRecordMessage, id: Optional[str]
            ) -> Document:
                if record_handler:
                    return record_handler(record, id)
                return Document(
                    doc_id=id, text=json.dumps(record.data), extra_info=record.data
                )

        self._integration = CDKIntegration(
            config=config,
            runner=CDKRunner(source=source_class(), name=source_class.__name__),
        )

    def load_data(self, *args: Any, **kwargs: Any) -> List[Document]:
        return list(self.lazy_load_data(*args, **kwargs))

    def lazy_load_data(self, *args: Any, **kwargs: Any) -> Iterator[Document]:
        return self._integration._load_data(*args, **kwargs)

    @property
    def last_state(self):
        return self._integration.last_state
