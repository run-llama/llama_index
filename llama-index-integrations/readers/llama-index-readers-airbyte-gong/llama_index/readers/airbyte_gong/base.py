from typing import Any, Mapping, Optional

from llama_index.readers.airbyte_cdk.base import AirbyteCDKReader, RecordHandler


class AirbyteGongReader(AirbyteCDKReader):
    """
    AirbyteGongReader reader.

    Retrieve documents from Gong

    Args:
        config: The config object for the gong source.

    """

    def __init__(
        self,
        config: Mapping[str, Any],
        record_handler: Optional[RecordHandler] = None,
    ) -> None:
        """Initialize with parameters."""
        import source_gong

        super().__init__(
            source_class=source_gong.SourceGong,
            config=config,
            record_handler=record_handler,
        )
