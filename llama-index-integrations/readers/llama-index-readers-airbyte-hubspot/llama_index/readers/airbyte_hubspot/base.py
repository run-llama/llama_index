from typing import Any, Mapping, Optional

from llama_index.readers.airbyte_cdk.base import AirbyteCDKReader, RecordHandler


class AirbyteHubspotReader(AirbyteCDKReader):
    """
    AirbyteHubspotReader reader.

    Retrieve documents from Hubspot

    Args:
        config: The config object for the hubspot source.

    """

    def __init__(
        self,
        config: Mapping[str, Any],
        record_handler: Optional[RecordHandler] = None,
    ) -> None:
        """Initialize with parameters."""
        import source_hubspot

        super().__init__(
            source_class=source_hubspot.SourceHubspot,
            config=config,
            record_handler=record_handler,
        )
