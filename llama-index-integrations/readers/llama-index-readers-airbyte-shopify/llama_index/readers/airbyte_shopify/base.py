from typing import Any, Mapping, Optional

from llama_index.readers.airbyte_cdk.base import AirbyteCDKReader, RecordHandler


class AirbyteShopifyReader(AirbyteCDKReader):
    """AirbyteShopifyReader reader.

    Retrieve documents from Shopify

    Args:
        config: The config object for the shopify source.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        record_handler: Optional[RecordHandler] = None,
    ) -> None:
        """Initialize with parameters."""
        import source_shopify

        super().__init__(
            source_class=source_shopify.SourceShopify,
            config=config,
            record_handler=record_handler,
        )
