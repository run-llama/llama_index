from typing import Any, Optional


class ServiceContext:
    """
    Service Context container.

    NOTE: Deprecated, use llama_index.settings.Settings instead or pass in
    modules to local functions/methods/interfaces.

    """

    def __init__(self, **kwargs: Any) -> None:
        raise ValueError(
            "ServiceContext is deprecated. Use llama_index.settings.Settings instead, "
            "or pass in modules to local functions/methods/interfaces.\n"
            "See the docs for updated usage/migration: \n"
            "https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/service_context_migration/"
        )

    @classmethod
    def from_defaults(
        cls,
        **kwargs: Any,
    ) -> "ServiceContext":
        """
        Create a ServiceContext from defaults.

        NOTE: Deprecated, use llama_index.settings.Settings instead or pass in
        modules to local functions/methods/interfaces.

        """
        raise ValueError(
            "ServiceContext is deprecated. Use llama_index.settings.Settings instead, "
            "or pass in modules to local functions/methods/interfaces.\n"
            "See the docs for updated usage/migration: \n"
            "https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/service_context_migration/"
        )


def set_global_service_context(service_context: Optional[ServiceContext]) -> None:
    """Helper function to set the global service context."""
    raise ValueError(
        "ServiceContext is deprecated. Use llama_index.settings.Settings instead, "
        "or pass in modules to local functions/methods/interfaces.\n"
        "See the docs for updated usage/migration: \n"
        "https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/service_context_migration/"
    )
