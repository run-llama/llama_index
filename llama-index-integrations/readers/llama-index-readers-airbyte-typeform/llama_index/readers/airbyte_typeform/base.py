from typing import Any, Mapping, Optional

from llama_index.readers.airbyte_cdk.base import AirbyteCDKReader, RecordHandler


class AirbyteTypeformReader(AirbyteCDKReader):
    """AirbyteTypeformReader reader.

    Retrieve documents from Typeform

    Args:
        config: The config object for the typeform source.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        record_handler: Optional[RecordHandler] = None,
    ) -> None:
        """Initialize with parameters."""
        import source_typeform

        super().__init__(
            source_class=source_typeform.SourceTypeform,
            config=config,
            record_handler=record_handler,
        )
