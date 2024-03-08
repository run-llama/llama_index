from llama_index.core.base.base_retriever import BaseRetriever


class NetworkRetriever(BaseRetriever):
    """..."""

    def __init__(
        self,
        contributors: List[ContributorClient],
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._contributors = contributors
        self._response_synthesizer = response_synthesizer or get_response_synthesizer(
            llm=Settings.llm, callback_manager=Settings.callback_manager
        )
        super().__init__(callback_manager=callback_manager)
