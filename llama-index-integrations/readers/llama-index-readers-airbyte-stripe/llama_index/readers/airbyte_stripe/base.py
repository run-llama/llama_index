from typing import Any, Mapping, Optional

from llama_index.readers.airbyte_cdk.base import AirbyteCDKReader, RecordHandler


class AirbyteStripeReader(AirbyteCDKReader):
    """
    AirbyteStripeReader reader.

    Retrieve documents from Stripe

    Args:
        config: The config object for the stripe source.

    """

    def __init__(
        self,
        config: Mapping[str, Any],
        record_handler: Optional[RecordHandler] = None,
    ) -> None:
        """Initialize with parameters."""
        import source_stripe

        super().__init__(
            source_class=source_stripe.SourceStripe,
            config=config,
            record_handler=record_handler,
        )
