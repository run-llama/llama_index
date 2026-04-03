"""
TruLens-Eval LlamaPack.
"""

from typing import Any, Dict, List

from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.schema import TextNode


class TruLensRAGTriadPack(BaseLlamaPack):
    """
    The TruLens-Eval RAG Triad LlamaPack show how to instrument and evaluate your LlamaIndex query
    engine. It launches starts a logging database and launches a dashboard in the background,
    builds an index over an input list of nodes, and instantiates and instruments a query engine
    over that index. It also instantiates the RAG triad (groundedness, context relevance, answer relevance)'
    so that query is logged and evaluated by this triad for detecting hallucination.

    Note: Using this LlamaPack requires that your OpenAI API key is set via the
    OPENAI_API_KEY environment variable.
    """

    def __init__(
        self,
        nodes: List[TextNode],
        app_id: str,
        **kwargs: Any,
    ) -> None:
        """
        Initializes a new instance of TruLensEvalPack.

        Args:
            nodes (List[TextNode]): An input list of nodes over which the index
            will be built.
            app_id (str): The application ID for the TruLensEvalPack.

        """
        try:
            from trulens_eval import Feedback, Tru, TruLlama
            from trulens_eval.feedback import Groundedness
            from trulens_eval.feedback.provider.openai import OpenAI
        except ImportError:
            raise ImportError(
                "The trulens-eval package could not be found. "
                "Please install with `pip install trulens-eval`."
            )
        self.app_id = app_id
        self._tru = Tru()
        self._tru.run_dashboard()
        self._index = VectorStoreIndex(nodes, **kwargs)
        self._query_engine = self._index.as_query_engine()

        import numpy as np

        # Initialize provider class
        provider = OpenAI()

        grounded = Groundedness(groundedness_provider=provider)

        # Define a groundedness feedback function
        f_groundedness = (
            Feedback(
                grounded.groundedness_measure_with_cot_reasons, name="Groundedness"
            )
            .on(TruLlama.select_source_nodes().node.text.collect())
            .on_output()
            .aggregate(grounded.grounded_statements_aggregator)
        )

        # Question/answer relevance between overall question and answer.
        f_qa_relevance = Feedback(
            provider.relevance, name="Answer Relevance"
        ).on_input_output()

        # Question/statement relevance between question and each context chunk.
        f_context_relevance = (
            Feedback(provider.qs_relevance, name="Context Relevance")
            .on_input()
            .on(TruLlama.select_source_nodes().node.text.collect())
            .aggregate(np.mean)
        )

        feedbacks = [f_groundedness, f_qa_relevance, f_context_relevance]

        self._tru_query_engine = TruLlama(
            self._query_engine, app_id=app_id, feedbacks=feedbacks
        )

    def get_modules(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing the internals of the LlamaPack.

        Returns:
            Dict[str, Any]: A dictionary containing the internals of the
            LlamaPack.

        """
        return {
            "session": self._tru,
            "index": self._index,
            "tru_query_engine": self._tru_query_engine,
            "query_engine": self._query_engine,
        }

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """
        Runs queries against the index.

        Returns:
            Any: A response from the query engine.

        """
        with self._tru_query_engine as _:
            return self._query_engine.query(*args, **kwargs)


class TruLensHarmlessPack(BaseLlamaPack):
    """
    The TruLens-Eval Harmless LlamaPack show how to instrument and evaluate your LlamaIndex query
    engine. It launches starts a logging database and launches a dashboard in the background,
    builds an index over an input list of nodes, and instantiates and instruments a query engine
    over that index. It also instantiates the a suite of Harmless evals so that query is logged
    and evaluated for harmlessness.

    Note: Using this LlamaPack requires that your OpenAI and HuggingFace API keys are set via the
    OPENAI_API_KEY and HUGGINGFACE_API_KEY environment variable.
    """

    def __init__(
        self,
        nodes: List[TextNode],
        app_id: str,
        **kwargs: Any,
    ) -> None:
        """
        Initializes a new instance of TruLensEvalPack.

        Args:
            nodes (List[TextNode]): An input list of nodes over which the index
            will be built.
            app_id (str): The application ID for the TruLensEvalPack.

        """
        try:
            from trulens_eval import Feedback, Tru, TruLlama
            from trulens_eval.feedback.provider.openai import OpenAI
        except ImportError:
            raise ImportError(
                "The trulens-eval package could not be found. "
                "Please install with `pip install trulens-eval`."
            )
        self.app_id = app_id
        self._tru = Tru()
        self._tru.run_dashboard()
        self._index = VectorStoreIndex(nodes, **kwargs)
        self._query_engine = self._index.as_query_engine()

        # Initialize provider class
        provider = OpenAI()

        # LLM-based feedback functions
        f_controversiality = Feedback(
            provider.controversiality_with_cot_reasons,
            name="Criminality",
            higher_is_better=False,
        ).on_output()
        f_criminality = Feedback(
            provider.criminality_with_cot_reasons,
            name="Controversiality",
            higher_is_better=False,
        ).on_output()
        f_insensitivity = Feedback(
            provider.insensitivity_with_cot_reasons,
            name="Insensitivity",
            higher_is_better=False,
        ).on_output()
        f_maliciousness = Feedback(
            provider.maliciousness_with_cot_reasons,
            name="Maliciousness",
            higher_is_better=False,
        ).on_output()

        # Moderation feedback functions
        f_hate = Feedback(
            provider.moderation_hate, name="Hate", higher_is_better=False
        ).on_output()
        f_hatethreatening = Feedback(
            provider.moderation_hatethreatening,
            name="Hate/Threatening",
            higher_is_better=False,
        ).on_output()
        f_violent = Feedback(
            provider.moderation_violence, name="Violent", higher_is_better=False
        ).on_output()
        f_violentgraphic = Feedback(
            provider.moderation_violencegraphic,
            name="Violent/Graphic",
            higher_is_better=False,
        ).on_output()
        f_selfharm = Feedback(
            provider.moderation_selfharm, name="Self Harm", higher_is_better=False
        ).on_output()

        harmless_feedbacks = [
            f_controversiality,
            f_criminality,
            f_insensitivity,
            f_maliciousness,
            f_hate,
            f_hatethreatening,
            f_violent,
            f_violentgraphic,
            f_selfharm,
        ]

        self._tru_query_engine = TruLlama(
            self._query_engine, app_id=app_id, feedbacks=harmless_feedbacks
        )

    def get_modules(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing the internals of the LlamaPack.

        Returns:
            Dict[str, Any]: A dictionary containing the internals of the
            LlamaPack.

        """
        return {
            "session": self._tru,
            "index": self._index,
            "tru_query_engine": self._tru_query_engine,
            "query_engine": self._query_engine,
        }

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """
        Runs queries against the index.

        Returns:
            Any: A response from the query engine.

        """
        with self._tru_query_engine as _:
            return self._query_engine.query(*args, **kwargs)


class TruLensHelpfulPack(BaseLlamaPack):
    """
    The TruLens-Eval Helpful LlamaPack show how to instrument and evaluate your LlamaIndex query
    engine. It launches starts a logging database and launches a dashboard in the background,
    builds an index over an input list of nodes, and instantiates and instruments a query engine
    over that index. It also instantiates the a suite of Helpful evals so that query is logged
    and evaluated for helpfulness.

    Note: Using this LlamaPack requires that your OpenAI and HuggingFace API keys are set via the
    OPENAI_API_KEY and HUGGINGFACE_API_KEY environment variable.
    """

    def __init__(
        self,
        nodes: List[TextNode],
        app_id: str,
        **kwargs: Any,
    ) -> None:
        """
        Initializes a new instance of TruLensEvalPack.

        Args:
            nodes (List[TextNode]): An input list of nodes over which the index
            will be built.
            app_id (str): The application ID for the TruLensEvalPack.

        """
        try:
            from trulens_eval import Feedback, Tru, TruLlama
            from trulens_eval.feedback.provider.hugs import Huggingface
            from trulens_eval.feedback.provider.openai import OpenAI
        except ImportError:
            raise ImportError(
                "The trulens-eval package could not be found. "
                "Please install with `pip install trulens-eval`."
            )
        self.app_id = app_id
        self._tru = Tru()
        self._tru.run_dashboard()
        self._index = VectorStoreIndex(nodes, **kwargs)
        self._query_engine = self._index.as_query_engine()

        # Initialize provider class
        provider = OpenAI()

        hugs_provider = Huggingface()

        # LLM-based feedback functions
        f_coherence = Feedback(
            provider.coherence_with_cot_reasons, name="Coherence"
        ).on_output()
        f_input_sentiment = Feedback(
            provider.sentiment_with_cot_reasons, name="Input Sentiment"
        ).on_input()
        f_output_sentiment = Feedback(
            provider.sentiment_with_cot_reasons, name="Output Sentiment"
        ).on_output()
        f_langmatch = Feedback(
            hugs_provider.language_match, name="Language Match"
        ).on_input_output()

        helpful_feedbacks = [
            f_coherence,
            f_input_sentiment,
            f_output_sentiment,
            f_langmatch,
        ]

        self._tru_query_engine = TruLlama(
            self._query_engine, app_id=app_id, feedbacks=helpful_feedbacks
        )

    def get_modules(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing the internals of the LlamaPack.

        Returns:
            Dict[str, Any]: A dictionary containing the internals of the
            LlamaPack.

        """
        return {
            "session": self._tru,
            "index": self._index,
            "tru_query_engine": self._tru_query_engine,
            "query_engine": self._query_engine,
        }

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """
        Runs queries against the index.

        Returns:
            Any: A response from the query engine.

        """
        with self._tru_query_engine as _:
            return self._query_engine.query(*args, **kwargs)
