from typing import Any, Mapping, Optional

from llama_index.readers.airbyte_cdk.base import AirbyteCDKReader, RecordHandler


class AirbyteSalesforceReader(AirbyteCDKReader):
    """AirbyteSalesforceReader reader.

    Retrieve documents from Salesforce

    Args:
        config: The config object for the salesforce source.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        record_handler: Optional[RecordHandler] = None,
    ) -> None:
        """Initialize with parameters."""
        import source_salesforce

        super().__init__(
            source_class=source_salesforce.SourceSalesforce,
            config=config,
            record_handler=record_handler,
        )
