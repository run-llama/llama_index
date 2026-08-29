from typing import Any, Mapping, Optional

from llama_index.readers.airbyte_cdk.base import AirbyteCDKReader, RecordHandler


class AirbyteZendeskSupportReader(AirbyteCDKReader):
    """
    AirbyteZendeskSupportReader reader.

    Retrieve documents from ZendeskSupport

    Args:
        config: The config object for the zendesk_support source.

    """

    def __init__(
        self,
        config: Mapping[str, Any],
        record_handler: Optional[RecordHandler] = None,
    ) -> None:
        """Initialize with parameters."""
        import source_zendesk_support

        super().__init__(
            source_class=source_zendesk_support.SourceZendeskSupport,
            config=config,
            record_handler=record_handler,
        )
