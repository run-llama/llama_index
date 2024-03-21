"""LLM-based reranker."""

from typing import Callable, List, Optional

class LLMRerank:
    """LLM-based reranker.

    This class represents an LLM-based reranker, which reorders a list of nodes based on their relevance to a given query using a Language Model (LLM).

    Attributes:
        top_n (int): The number of top-ranked nodes to return.
        choice_select_prompt (BasePromptTemplate): The prompt template used for choice selection.
        choice_batch_size (int): The batch size for choice selection.
        llm (LLM): The Language Model used for reranking.

    """

    top_n: int
    choice_select_prompt: "BasePromptTemplate"
    choice_batch_size: int
    llm: "LLM"

    def __init__(
        self,
        llm: Optional["LLM"] = None,
        choice_select_prompt: Optional["BasePromptTemplate"] = None,
        choice_batch_size: int = 10,
        format_node_batch_fn: Optional[Callable] = None,
        parse_choice_select_answer_fn: Optional[Callable] = None,
        service_context: Optional["ServiceContext"] = None,
        top_n: int = 10,
    ) -> None:
        """Initialize LLMRerank.

        Args:
            llm (Optional[LLM]): The LLM instance to use. Defaults to None.
            choice_select_prompt (Optional[BasePromptTemplate]): The prompt template for choice selection. Defaults to None.
            choice_batch_size (int): The batch size for choice selection. Defaults to 10.
            format_node_batch_fn (Optional[Callable]): Function to format node batch. Defaults to None.
            parse_choice_select_answer_fn (Optional[Callable]): Function to parse choice select answer. Defaults to None.
            service_context (Optional[ServiceContext]): Service context. Defaults to None.
            top_n (int): The number of top-ranked nodes to return. Defaults to 10.

        """
        choice_select_prompt = choice_select_prompt or DEFAULT_CHOICE_SELECT_PROMPT

        llm = llm or llm_from_settings_or_context(Settings, service_context)

        self._format_node_batch_fn = (
            format_node_batch_fn or default_format_node_batch_fn
        )
        self._parse_choice_select_answer_fn = (
            parse_choice_select_answer_fn or default_parse_choice_select_answer_fn
        )

        super().__init__(
            llm=llm,
            choice_select_prompt=choice_select_prompt,
            choice_batch_size=choice_batch_size,
            service_context=service_context,
            top_n=top_n,
        )

    def _get_prompts(self) -> "PromptDictType":
        """Get prompts."""
        pass

    def _update_prompts(self, prompts: "PromptDictType") -> None:
        """Update prompts."""
        pass

    @classmethod
    def class_name(cls) -> str:
        """Return the class name."""
        return "LLMRerank"

    def _postprocess_nodes(
        self,
        nodes: List["NodeWithScore"],
        query_bundle: Optional["QueryBundle"] = None,
    ) -> List["NodeWithScore"]:
        """Rerank nodes based on LLM predictions.

        Args:
            nodes (List[NodeWithScore]): The list of nodes to rerank.
            query_bundle (Optional[QueryBundle]): The query bundle. Defaults to None.

        Returns:
            List[NodeWithScore]: The reranked nodes.

        Raises:
            ValueError: If query bundle is not provided.

        """
        pass

    def compare_with_cohere_rerank(self):
        """Compare with Cohere Rerank.

        This method provides a comparison between LLMRerank and Cohere Rerank in terms of performance and pricing.

        Returns:
            str: Comparison between LLMRerank and Cohere Rerank.

        """
        comparison_details = """
        LLMRerank vs. Cohere Rerank:

        Performance:
        - LLMRerank leverages Language Models (LLMs) for reranking, offering flexibility in model selection and customization.
        - Cohere Rerank utilizes the capabilities of the Cohere API for reranking.

        Pricing:
        - Pricing for LLMRerank may vary based on the selected Language Model provider and usage.
        - Cohere Rerank's pricing is determined by the pricing plans of the Cohere API.

        Conclusion:
        - LLMRerank offers flexibility and potentially lower costs depending on the selected Language Model provider.
        - Cohere Rerank provides a convenient solution integrated with the Cohere API but may have specific pricing considerations.

        """

        return comparison_details
 